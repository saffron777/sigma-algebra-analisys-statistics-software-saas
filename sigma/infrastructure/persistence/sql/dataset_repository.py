"""Datasets on any SQL engine, scoped to one user.

Metadata lives in ``datasets`` (every query filters by ``user_id``, so another user's id behaves exactly like a missing
one); the rows live in a table of their own (see ``table_io.py``).
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Connection, Row
from sqlalchemy.exc import IntegrityError

from sigma.domain.datasets import ColumnInfo, Dataset, DatasetInfo, describe_columns
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError
from sigma.infrastructure.persistence.sql import schema, table_io
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import bump_sync, now_iso

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_d = schema.datasets
_r = schema.saved_results


class SqlDatasetRepository:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ writes
    def add(self, name: str, frame: pd.DataFrame, source_type: str, source_meta: dict[str, Any]) -> DatasetInfo:
        name = self._valid_name(name)
        dataset_id = uuid.uuid4().hex
        table_name = table_io.new_table_name("ds")
        try:
            with self._db.begin() as conn:
                self._ensure_name_free(conn, name)
                col_map = table_io.write_frame(conn, table_name, frame)
                dtypes, columns = self._frame_metadata(frame)
                conn.execute(
                    insert(_d).values(
                        id=dataset_id,
                        user_id=self._user,
                        name=name,
                        name_key=name.lower(),
                        source_type=source_type,
                        source_meta=json.dumps(source_meta, ensure_ascii=False, default=str),
                        n_rows=len(frame),
                        n_cols=frame.shape[1],
                        dtypes=dtypes,
                        columns_info=columns,
                        created_at=now_iso(),
                        table_name=table_name,
                        col_map=json.dumps(col_map),
                    )
                )
                bump_sync(conn, self._user, "datasets")
        except IntegrityError:
            self._drop_quietly(table_name)
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.") from None
        except BaseException:
            self._drop_quietly(table_name)  # engines whose DDL cannot be rolled back (MySQL) leave the table behind
            raise
        return self.get_info(dataset_id)

    def replace_data(self, dataset_id: str, frame: pd.DataFrame) -> DatasetInfo:
        new_table = table_io.new_table_name("ds")
        try:
            with self._db.begin() as conn:
                row = self._row(conn, dataset_id)
                col_map = table_io.write_frame(conn, new_table, frame)
                dtypes, columns = self._frame_metadata(frame)
                conn.execute(
                    update(_d)
                    .where(_d.c.id == dataset_id, _d.c.user_id == self._user)
                    .values(
                        n_rows=len(frame),
                        n_cols=frame.shape[1],
                        dtypes=dtypes,
                        columns_info=columns,
                        table_name=new_table,
                        col_map=json.dumps(col_map),
                    )
                )
                table_io.drop_table(conn, row.table_name)
                bump_sync(conn, self._user, "datasets")
        except BaseException:
            self._drop_quietly(new_table)
            raise
        return self.get_info(dataset_id)

    def rename(self, dataset_id: str, name: str) -> DatasetInfo:
        name = self._valid_name(name)
        with self._db.begin() as conn:
            self._row(conn, dataset_id)
            self._ensure_name_free(conn, name, except_id=dataset_id)
            try:
                conn.execute(
                    update(_d)
                    .where(_d.c.id == dataset_id, _d.c.user_id == self._user)
                    .values(name=name, name_key=name.lower())
                )
            except IntegrityError:
                raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.") from None
            conn.execute(
                update(_r).where(_r.c.dataset_id == dataset_id, _r.c.user_id == self._user).values(dataset_name=name)
            )
            bump_sync(conn, self._user, "datasets")
            bump_sync(conn, self._user, "results")
        return self.get_info(dataset_id)

    def delete(self, dataset_id: str) -> None:
        with self._db.begin() as conn:
            row = self._row(conn, dataset_id)
            self._release(conn, [dataset_id])
            table_io.drop_table(conn, row.table_name)
            conn.execute(delete(_d).where(_d.c.id == dataset_id, _d.c.user_id == self._user))
            bump_sync(conn, self._user, "datasets")

    def purge(self) -> int:
        """Delete every dataset (and its data table) of this user; returns how many were removed."""
        with self._db.begin() as conn:
            rows = conn.execute(select(_d.c.id, _d.c.table_name).where(_d.c.user_id == self._user)).all()
            self._release(conn, [row.id for row in rows])
            for row in rows:
                table_io.drop_table(conn, row.table_name)
            conn.execute(delete(_d).where(_d.c.user_id == self._user))
        return len(rows)

    # ------------------------------------------------------------------ reads
    def get_info(self, dataset_id: str) -> DatasetInfo:
        with self._db.begin() as conn:
            return self._to_info(self._row(conn, dataset_id))

    def list(self) -> list[DatasetInfo]:
        with self._db.begin() as conn:
            rows = conn.execute(
                select(_d).where(_d.c.user_id == self._user).order_by(_d.c.created_at.desc(), _d.c.name)
            ).all()
        return [self._to_info(row) for row in rows]

    def get(self, dataset_id: str) -> Dataset:
        with self._db.begin() as conn:
            row = self._row(conn, dataset_id)
            return Dataset(self._to_info(row), self._read(conn, row, None, None))

    def get_by_name(self, name: str) -> Dataset:
        with self._db.begin() as conn:
            row = conn.execute(
                select(_d).where(_d.c.name_key == (name or "").strip().lower(), _d.c.user_id == self._user)
            ).first()
            if row is None:
                raise DatasetNotFoundError(f"No existe un dataset llamado «{name}».")
            return Dataset(self._to_info(row), self._read(conn, row, None, None))

    def preview(self, dataset_id: str, limit: int, offset: int) -> pd.DataFrame:
        with self._db.begin() as conn:
            return self._read(conn, self._row(conn, dataset_id), limit, offset)

    # ------------------------------------------------------------------ internals
    def _row(self, conn: Connection, dataset_id: str) -> Row:
        if not _ID_PATTERN.match(dataset_id or ""):
            raise DatasetNotFoundError("Identificador de dataset inválido.")
        row = conn.execute(select(_d).where(_d.c.id == dataset_id, _d.c.user_id == self._user)).first()
        if row is None:
            raise DatasetNotFoundError("El dataset no existe (puede haber sido eliminado).")
        return row

    def _read(self, conn: Connection, row: Row, limit: int | None, offset: int | None) -> pd.DataFrame:
        info_columns = json.loads(row.columns_info)
        names = list(json.loads(row.dtypes))  # dtype tags are keyed by column name, in column order
        _ = info_columns
        col_map = json.loads(row.col_map) if row.col_map else None
        return table_io.read_frame(conn, row.table_name, names, col_map, json.loads(row.dtypes), limit, offset)

    def _release(self, conn: Connection, dataset_ids: list[str]) -> None:
        """Results keep their name but lose the link to a dataset that is going away."""
        if not dataset_ids:
            return
        conn.execute(
            update(_r).where(_r.c.dataset_id.in_(dataset_ids), _r.c.user_id == self._user).values(dataset_id=None)
        )
        bump_sync(conn, self._user, "results")

    def _drop_quietly(self, table_name: str) -> None:
        with contextlib.suppress(Exception), self._db.begin() as conn:
            table_io.drop_table(conn, table_name)

    @staticmethod
    def _valid_name(name: str) -> str:
        name = " ".join((name or "").split())
        if not name:
            raise InvalidInputError("El nombre del dataset no puede estar vacío.")
        return name

    def _ensure_name_free(self, conn: Connection, name: str, except_id: str | None = None) -> None:
        row = conn.execute(select(_d.c.id).where(_d.c.name_key == name.lower(), _d.c.user_id == self._user)).first()
        if row is not None and row.id != except_id:
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.")

    @staticmethod
    def _frame_metadata(frame: pd.DataFrame) -> tuple[str, str]:
        dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
        columns = [vars(c) for c in describe_columns(frame)]
        return json.dumps(dtypes), json.dumps(columns)

    @staticmethod
    def _to_info(row: Row) -> DatasetInfo:
        columns = tuple(ColumnInfo(**item) for item in json.loads(row.columns_info))
        return DatasetInfo(
            id=row.id,
            name=row.name,
            source_type=row.source_type,
            n_rows=row.n_rows,
            n_cols=row.n_cols,
            columns=columns,
            created_at=row.created_at,
            source_meta=json.loads(row.source_meta),
        )
