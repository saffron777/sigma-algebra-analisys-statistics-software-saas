"""Pending imports (drafts) on any SQL engine, scoped to one user.

A draft keeps the uploaded file (in pieces, so no statement exceeds a server's packet limit) and, once parsed, the table
awaiting confirmation (a ``stg_<id>`` table, see ``table_io.py``). There is one draft per user and source type. Every
query filters by ``user_id``: somebody else's draft id behaves exactly like an expired one.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Connection, Row

from sigma.application.ports import DraftInfo, StagedImport
from sigma.domain.datasets import ColumnInfo, describe_columns
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.sql import schema, table_io
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import bump_sync, now_iso, upsert

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
EXPIRED = "La importación pendiente ya no existe (se descartó, expiró o ya fue guardada). Cargue la fuente otra vez."
CHUNK_BYTES = 8 * 1024 * 1024  # below the default packet limit of every supported server
_d = schema.import_drafts
_f = schema.import_draft_files


class SqlImportDraftStore:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ file
    def attach_file(self, source_type: str, file_name: str, content: bytes) -> DraftInfo:
        now = now_iso()
        with self._db.begin() as conn:
            previous = self._row_for_source(conn, source_type)
            if previous is not None:  # a new file makes the old parsed preview stale
                table_io.drop_table(conn, previous.staged_table)
            fresh = {
                "file_name": file_name,
                "file_size": len(content),
                "options": "{}",
                "staged_table": None,
                "n_rows": None,
                "n_cols": None,
                "dtypes": None,
                "columns_info": None,
                "col_map": None,
                "source_meta": "{}",
                "detected": "{}",
                "suggested_name": None,
                "updated_at": now,
            }
            upsert(
                conn,
                _d,
                {"user_id": self._user, "source_type": source_type},
                {"id": uuid.uuid4().hex, "created_at": now, **fresh},
                fresh,
            )
            draft = self._row_for_source(conn, source_type)
            conn.execute(delete(_f).where(_f.c.draft_id == draft.id))
            for seq, start in enumerate(range(0, max(len(content), 1), CHUNK_BYTES)):
                conn.execute(insert(_f).values(draft_id=draft.id, seq=seq, data=content[start : start + CHUNK_BYTES]))
            bump_sync(conn, self._user, "drafts")
            draft_id = draft.id
        return self.info(draft_id)

    def file(self, token: str) -> tuple[DraftInfo, bytes | None]:
        info = self.info(token)
        with self._db.begin() as conn:
            chunks = conn.execute(select(_f.c.data).where(_f.c.draft_id == token).order_by(_f.c.seq)).all()
        return info, (b"".join(bytes(chunk.data) for chunk in chunks) if chunks else None)

    # ------------------------------------------------------------------ staging
    def put(self, staged: StagedImport, token: str | None = None, options: dict[str, Any] | None = None) -> str:
        frame = staged.frame
        dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
        columns = [vars(c) for c in describe_columns(frame)]
        now = now_iso()
        table = table_io.new_table_name("stg")
        try:
            with self._db.begin() as conn:
                row = self._row_by_id(conn, token) if token else self._row_for_source(conn, staged.source_type)
                if token and row is None:
                    raise InvalidInputError(EXPIRED)
                if row is None:
                    upsert(
                        conn,
                        _d,
                        {"user_id": self._user, "source_type": staged.source_type},
                        {
                            "id": uuid.uuid4().hex,
                            "file_size": 0,
                            "options": "{}",
                            "source_meta": "{}",
                            "detected": "{}",
                            "created_at": now,
                            "updated_at": now,
                        },
                        {"updated_at": now},
                    )
                    row = self._row_for_source(conn, staged.source_type)
                table_io.drop_table(conn, row.staged_table)
                col_map = table_io.write_frame(conn, table, frame)
                conn.execute(
                    update(_d)
                    .where(_d.c.id == row.id)
                    .values(
                        staged_table=table,
                        n_rows=len(frame),
                        n_cols=frame.shape[1],
                        dtypes=json.dumps(dtypes),
                        columns_info=json.dumps(columns),
                        col_map=json.dumps(col_map),
                        source_meta=json.dumps(staged.source_meta, ensure_ascii=False, default=str),
                        detected=json.dumps(staged.detected, ensure_ascii=False, default=str),
                        suggested_name=staged.suggested_name,
                        options=json.dumps(options or {}, ensure_ascii=False, default=str),
                        updated_at=now,
                    )
                )
                bump_sync(conn, self._user, "drafts")
                draft_id = row.id
        except BaseException:
            self._drop_quietly(table)
            raise
        return draft_id

    def take(self, token: str) -> StagedImport:
        with self._db.begin() as conn:
            row = self._row_by_id(conn, token)
            if row is None or not row.staged_table:
                raise InvalidInputError(EXPIRED)
            frame = self._read(conn, row, None)
        return StagedImport(
            frame=frame,
            source_type=row.source_type,
            source_meta=json.loads(row.source_meta),
            suggested_name=row.suggested_name or "importado",
            detected=json.loads(row.detected),
        )

    def preview(self, token: str, limit: int) -> tuple[DraftInfo, tuple[ColumnInfo, ...], pd.DataFrame, dict[str, Any]]:
        """Info, columns, first rows and detected options of the staged table (to show it again after a reload)."""
        with self._db.begin() as conn:
            row = self._row_by_id(conn, token)
            if row is None or not row.staged_table:
                raise InvalidInputError(EXPIRED)
            frame = self._read(conn, row, limit)
        columns = tuple(ColumnInfo(**item) for item in json.loads(row.columns_info))
        return self._to_info(row), columns, frame, json.loads(row.detected)

    # ------------------------------------------------------------------ reads
    def info(self, token: str) -> DraftInfo:
        with self._db.begin() as conn:
            row = self._row_by_id(conn, token)
        if row is None:
            raise InvalidInputError(EXPIRED)
        return self._to_info(row)

    def list(self) -> list[DraftInfo]:
        with self._db.begin() as conn:
            rows = conn.execute(select(_d).where(_d.c.user_id == self._user).order_by(_d.c.updated_at.desc())).all()
        return [self._to_info(row) for row in rows]

    # ------------------------------------------------------------------ discarding
    def discard(self, token: str) -> None:
        with self._db.begin() as conn:
            row = self._row_by_id(conn, token)
            if row is None:
                raise InvalidInputError(EXPIRED)
            table_io.drop_table(conn, row.staged_table)
            conn.execute(delete(_f).where(_f.c.draft_id == token))
            conn.execute(delete(_d).where(_d.c.id == token, _d.c.user_id == self._user))
            bump_sync(conn, self._user, "drafts")

    def discard_preview(self, token: str) -> None:
        with self._db.begin() as conn:
            row = self._row_by_id(conn, token)
            if row is None:
                raise InvalidInputError(EXPIRED)
            table_io.drop_table(conn, row.staged_table)
            conn.execute(
                update(_d)
                .where(_d.c.id == token)
                .values(
                    staged_table=None,
                    n_rows=None,
                    n_cols=None,
                    dtypes=None,
                    columns_info=None,
                    col_map=None,
                    detected="{}",
                    suggested_name=None,
                    updated_at=now_iso(),
                )
            )
            bump_sync(conn, self._user, "drafts")

    def purge(self) -> int:
        """Remove every draft of this user (account deletion, «Limpiar»); returns how many there were."""
        drafts = self.list()
        for draft in drafts:
            self.discard(draft.id)
        return len(drafts)

    # ------------------------------------------------------------------ internals
    def _read(self, conn: Connection, row: Row, limit: int | None) -> pd.DataFrame:
        dtypes = json.loads(row.dtypes)
        return table_io.read_frame(conn, row.staged_table, list(dtypes), json.loads(row.col_map), dtypes, limit, 0)

    def _row_by_id(self, conn: Connection, token: str | None) -> Row | None:
        if not _ID_PATTERN.match(token or ""):
            return None
        return conn.execute(select(_d).where(_d.c.id == token, _d.c.user_id == self._user)).first()

    def _row_for_source(self, conn: Connection, source_type: str) -> Row | None:
        return conn.execute(select(_d).where(_d.c.user_id == self._user, _d.c.source_type == source_type)).first()

    def _drop_quietly(self, table: str) -> None:
        try:
            with self._db.begin() as conn:
                table_io.drop_table(conn, table)
        except Exception:  # noqa: BLE001 - cleanup of a table that may not exist
            pass

    @staticmethod
    def _to_info(row: Row) -> DraftInfo:
        return DraftInfo(
            id=row.id,
            source_type=row.source_type,
            file_name=row.file_name,
            file_size=row.file_size,
            options=json.loads(row.options),
            n_rows=row.n_rows,
            n_cols=row.n_cols,
            suggested_name=row.suggested_name,
            has_preview=bool(row.staged_table),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


def purge_stale_drafts(database: Database, max_age_days: int = 14) -> int:
    """Delete drafts nobody touched for ``max_age_days`` (every user); run once when the application starts."""
    cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
    removed = 0
    with database.begin() as conn:
        for row in conn.execute(select(_d.c.id, _d.c.staged_table).where(_d.c.updated_at < cutoff)).all():
            table_io.drop_table(conn, row.staged_table)
            conn.execute(delete(_f).where(_f.c.draft_id == row.id))
            conn.execute(delete(_d).where(_d.c.id == row.id))
            removed += 1
    return removed
