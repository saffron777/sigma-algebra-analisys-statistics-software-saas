"""SQLite store for pending imports (drafts), scoped to one user.

A draft keeps the uploaded file (as a BLOB) and, once parsed, the table awaiting confirmation (in a ``stg_<id>``
table, encoded like a dataset so dtypes survive). There is one draft per user and source type. Every query filters
by ``user_id``: somebody else's draft id behaves exactly like an expired one.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from sigma.application.ports import DraftInfo, StagedImport
from sigma.domain.datasets import ColumnInfo, describe_columns
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.frame_codec import decode_frame, encode_frame

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_WRITE_CHUNK = 20_000
EXPIRED = "La importación pendiente ya no existe (se descartó, expiró o ya fue guardada). Cargue la fuente otra vez."


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _table(draft_id: str) -> str:
    if not _ID_PATTERN.match(draft_id or ""):
        raise InvalidInputError(EXPIRED)
    return f"stg_{draft_id}"


class SqliteImportDraftStore:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ file
    def attach_file(self, source_type: str, file_name: str, content: bytes) -> DraftInfo:
        now = _now()
        with self._db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row_for_source(conn, source_type)
            if row is None:
                draft_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO import_drafts (id, user_id, source_type, file_name, file_blob, file_size, created_at,"
                    " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (draft_id, self._user, source_type, file_name, content, len(content), now, now),
                )
            else:  # a new file replaces the old one and makes its parsed preview stale
                draft_id = row["id"]
                self._drop_staged(conn, row["staged_table"])
                conn.execute(
                    "UPDATE import_drafts SET file_name = ?, file_blob = ?, file_size = ?, options = '{}',"
                    " staged_table = NULL, n_rows = NULL, n_cols = NULL, dtypes = NULL, columns_info = NULL,"
                    " source_meta = '{}', detected = '{}', suggested_name = NULL, updated_at = ? WHERE id = ?",
                    (file_name, content, len(content), now, draft_id),
                )
        return self.info(draft_id)

    def file(self, token: str) -> tuple[DraftInfo, bytes | None]:
        info = self.info(token)
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT file_blob FROM import_drafts WHERE id = ? AND user_id = ?", (token, self._user)
            ).fetchone()
        return info, (bytes(row["file_blob"]) if row and row["file_blob"] is not None else None)

    # ------------------------------------------------------------------ staging
    def put(self, staged: StagedImport, token: str | None = None, options: dict[str, Any] | None = None) -> str:
        encoded, dtypes = encode_frame(staged.frame)
        columns = [vars(c) for c in describe_columns(staged.frame)]
        now = _now()
        with self._db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row_by_id(conn, token) if token else self._row_for_source(conn, staged.source_type)
            if token and row is None:
                raise InvalidInputError(EXPIRED)
            if row is None:
                draft_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO import_drafts (id, user_id, source_type, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (draft_id, self._user, staged.source_type, now, now),
                )
            else:
                draft_id = row["id"]
                self._drop_staged(conn, row["staged_table"])
            table = _table(draft_id)
            try:
                encoded.to_sql(table, conn, index=False, chunksize=_WRITE_CHUNK)
            except (sqlite3.Error, ValueError) as error:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                raise InvalidInputError(f"No se pudo conservar la vista previa: {error}") from error
            conn.execute(
                "UPDATE import_drafts SET staged_table = ?, n_rows = ?, n_cols = ?, dtypes = ?, columns_info = ?,"
                " source_meta = ?, detected = ?, suggested_name = ?, options = ?, updated_at = ? WHERE id = ?",
                (
                    table,
                    len(staged.frame),
                    staged.frame.shape[1],
                    json.dumps(dtypes),
                    json.dumps(columns),
                    json.dumps(staged.source_meta, ensure_ascii=False, default=str),
                    json.dumps(staged.detected, ensure_ascii=False, default=str),
                    staged.suggested_name,
                    json.dumps(options or {}, ensure_ascii=False, default=str),
                    now,
                    draft_id,
                ),
            )
        return draft_id

    def take(self, token: str) -> StagedImport:
        with self._db.connection() as conn:
            row = self._row_by_id(conn, token)
            if row is None or not row["staged_table"]:
                raise InvalidInputError(EXPIRED)
            frame = pd.read_sql_query(f'SELECT * FROM "{_table(token)}"', conn)
        return StagedImport(
            frame=decode_frame(frame, json.loads(row["dtypes"])),
            source_type=row["source_type"],
            source_meta=json.loads(row["source_meta"]),
            suggested_name=row["suggested_name"] or "importado",
            detected=json.loads(row["detected"]),
        )

    def preview(self, token: str, limit: int) -> tuple[DraftInfo, tuple[ColumnInfo, ...], pd.DataFrame, dict[str, Any]]:
        """Info, columns, first rows and detected options of the staged table (to show it again after a reload)."""
        with self._db.connection() as conn:
            row = self._row_by_id(conn, token)
            if row is None or not row["staged_table"]:
                raise InvalidInputError(EXPIRED)
            frame = pd.read_sql_query(f'SELECT * FROM "{_table(token)}" LIMIT ?', conn, params=(int(limit),))
        columns = tuple(ColumnInfo(**item) for item in json.loads(row["columns_info"]))
        return (
            self._to_info(row),
            columns,
            decode_frame(frame, json.loads(row["dtypes"])),
            json.loads(row["detected"]),
        )

    # ------------------------------------------------------------------ reads
    def info(self, token: str) -> DraftInfo:
        with self._db.connection() as conn:
            row = self._row_by_id(conn, token)
        if row is None:
            raise InvalidInputError(EXPIRED)
        return self._to_info(row)

    def list(self) -> list[DraftInfo]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM import_drafts WHERE user_id = ? ORDER BY updated_at DESC", (self._user,)
            ).fetchall()
        return [self._to_info(row) for row in rows]

    # ------------------------------------------------------------------ discarding
    def discard(self, token: str) -> None:
        with self._db.connection() as conn:
            row = self._row_by_id(conn, token)
            if row is None:
                raise InvalidInputError(EXPIRED)
            self._drop_staged(conn, row["staged_table"])
            conn.execute("DELETE FROM import_drafts WHERE id = ? AND user_id = ?", (token, self._user))

    def discard_preview(self, token: str) -> None:
        with self._db.connection() as conn:
            row = self._row_by_id(conn, token)
            if row is None:
                raise InvalidInputError(EXPIRED)
            self._drop_staged(conn, row["staged_table"])
            conn.execute(
                "UPDATE import_drafts SET staged_table = NULL, n_rows = NULL, n_cols = NULL, dtypes = NULL,"
                " columns_info = NULL, detected = '{}', suggested_name = NULL, updated_at = ? WHERE id = ?",
                (_now(), token),
            )

    def purge(self) -> int:
        """Remove every draft of this user (account deletion); returns how many there were."""
        drafts = self.list()
        for draft in drafts:
            self.discard(draft.id)
        return len(drafts)

    # ------------------------------------------------------------------ internals
    def _row_by_id(self, conn: sqlite3.Connection, token: str | None) -> sqlite3.Row | None:
        if not _ID_PATTERN.match(token or ""):
            return None
        return conn.execute("SELECT * FROM import_drafts WHERE id = ? AND user_id = ?", (token, self._user)).fetchone()

    def _row_for_source(self, conn: sqlite3.Connection, source_type: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM import_drafts WHERE user_id = ? AND source_type = ?", (self._user, source_type)
        ).fetchone()

    @staticmethod
    def _drop_staged(conn: sqlite3.Connection, table: str | None) -> None:
        if table and re.fullmatch(r"stg_[0-9a-f]{32}", table):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    @staticmethod
    def _to_info(row: sqlite3.Row) -> DraftInfo:
        return DraftInfo(
            id=row["id"],
            source_type=row["source_type"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            options=json.loads(row["options"]),
            n_rows=row["n_rows"],
            n_cols=row["n_cols"],
            suggested_name=row["suggested_name"],
            has_preview=bool(row["staged_table"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def purge_stale_drafts(database: Database, max_age_days: int = 14) -> int:
    """Delete drafts nobody touched for ``max_age_days`` (every user); run once when the application starts."""
    cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
    removed = 0
    with database.connection() as conn:
        for row in conn.execute(
            "SELECT id, staged_table FROM import_drafts WHERE updated_at < ?", (cutoff,)
        ).fetchall():
            SqliteImportDraftStore._drop_staged(conn, row["staged_table"])
            conn.execute("DELETE FROM import_drafts WHERE id = ?", (row["id"],))
            removed += 1
    return removed
