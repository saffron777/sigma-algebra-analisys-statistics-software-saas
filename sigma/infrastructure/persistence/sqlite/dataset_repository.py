"""SQLite-backed dataset store: metadata in ``datasets``, rows in one ``ds_<uuid>`` table each."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from sigma.domain.datasets import ColumnInfo, Dataset, DatasetInfo, describe_columns
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.frame_codec import decode_frame, encode_frame

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_WRITE_CHUNK = 20_000


class SqliteDatasetRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------ writes
    def add(self, name: str, frame: pd.DataFrame, source_type: str, source_meta: dict[str, Any]) -> DatasetInfo:
        name = self._valid_name(name)
        dataset_id = uuid.uuid4().hex
        with self._db.connection() as conn:
            self._ensure_name_free(conn, name)
            self._write_table(conn, dataset_id, frame)
            try:
                conn.execute(
                    "INSERT INTO datasets (id, name, source_type, source_meta, n_rows, n_cols, dtypes,"
                    " columns_info, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dataset_id,
                        name,
                        source_type,
                        json.dumps(source_meta, ensure_ascii=False, default=str),
                        len(frame),
                        frame.shape[1],
                        *self._frame_metadata(frame),
                        datetime.now(UTC).isoformat(timespec="seconds"),
                    ),
                )
            except sqlite3.Error:
                conn.execute(f'DROP TABLE IF EXISTS "{self._table(dataset_id)}"')
                raise
        return self.get_info(dataset_id)

    def replace_data(self, dataset_id: str, frame: pd.DataFrame) -> DatasetInfo:
        self.get_info(dataset_id)
        with self._db.connection() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{self._table(dataset_id)}"')
            self._write_table(conn, dataset_id, frame)
            conn.execute(
                "UPDATE datasets SET n_rows = ?, n_cols = ?, dtypes = ?, columns_info = ? WHERE id = ?",
                (len(frame), frame.shape[1], *self._frame_metadata(frame), dataset_id),
            )
        return self.get_info(dataset_id)

    def rename(self, dataset_id: str, name: str) -> DatasetInfo:
        name = self._valid_name(name)
        self.get_info(dataset_id)
        with self._db.connection() as conn:
            self._ensure_name_free(conn, name, except_id=dataset_id)
            conn.execute("UPDATE datasets SET name = ? WHERE id = ?", (name, dataset_id))
            conn.execute("UPDATE saved_results SET dataset_name = ? WHERE dataset_id = ?", (name, dataset_id))
        return self.get_info(dataset_id)

    def delete(self, dataset_id: str) -> None:
        self.get_info(dataset_id)
        with self._db.connection() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{self._table(dataset_id)}"')
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))

    # ------------------------------------------------------------------ reads
    def get_info(self, dataset_id: str) -> DatasetInfo:
        self._table(dataset_id)  # validates the id format
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise DatasetNotFoundError("El dataset no existe (puede haber sido eliminado).")
        return self._to_info(row)

    def list(self) -> list[DatasetInfo]:
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC, name").fetchall()
        return [self._to_info(row) for row in rows]

    def get(self, dataset_id: str) -> Dataset:
        info = self.get_info(dataset_id)
        return Dataset(info, self._read_table(dataset_id, None, None))

    def get_by_name(self, name: str) -> Dataset:
        with self._db.connection() as conn:
            row = conn.execute("SELECT id FROM datasets WHERE name = ? COLLATE NOCASE", (name.strip(),)).fetchone()
        if row is None:
            raise DatasetNotFoundError(f"No existe un dataset llamado «{name}».")
        return self.get(row["id"])

    def preview(self, dataset_id: str, limit: int, offset: int) -> pd.DataFrame:
        self.get_info(dataset_id)
        return self._read_table(dataset_id, limit, offset)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _table(dataset_id: str) -> str:
        if not _ID_PATTERN.match(dataset_id or ""):
            raise DatasetNotFoundError("Identificador de dataset inválido.")
        return f"ds_{dataset_id}"

    @staticmethod
    def _valid_name(name: str) -> str:
        name = " ".join((name or "").split())
        if not name:
            raise InvalidInputError("El nombre del dataset no puede estar vacío.")
        return name

    @staticmethod
    def _ensure_name_free(conn: sqlite3.Connection, name: str, except_id: str | None = None) -> None:
        row = conn.execute("SELECT id FROM datasets WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if row is not None and row["id"] != except_id:
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.")

    @staticmethod
    def _frame_metadata(frame: pd.DataFrame) -> tuple[str, str]:
        dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
        columns = [vars(c) for c in describe_columns(frame)]
        return json.dumps(dtypes), json.dumps(columns)

    def _write_table(self, conn: sqlite3.Connection, dataset_id: str, frame: pd.DataFrame) -> None:
        encoded, _ = encode_frame(frame)
        try:
            encoded.to_sql(self._table(dataset_id), conn, index=False, chunksize=_WRITE_CHUNK)
        except (sqlite3.Error, ValueError) as error:
            conn.execute(f'DROP TABLE IF EXISTS "{self._table(dataset_id)}"')
            raise InvalidInputError(f"No se pudo guardar el dataset: {error}") from error

    def _read_table(self, dataset_id: str, limit: int | None, offset: int | None) -> pd.DataFrame:
        table = self._table(dataset_id)
        sql = f'SELECT * FROM "{table}"'
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (int(limit), int(offset or 0))
        with self._db.connection() as conn:
            frame = pd.read_sql_query(sql, conn, params=params)
            row = conn.execute("SELECT dtypes FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return decode_frame(frame, json.loads(row["dtypes"]))

    @staticmethod
    def _to_info(row: sqlite3.Row) -> DatasetInfo:
        columns = tuple(ColumnInfo(**item) for item in json.loads(row["columns_info"]))
        return DatasetInfo(
            id=row["id"],
            name=row["name"],
            source_type=row["source_type"],
            n_rows=row["n_rows"],
            n_cols=row["n_cols"],
            columns=columns,
            created_at=row["created_at"],
            source_meta=json.loads(row["source_meta"]),
        )
