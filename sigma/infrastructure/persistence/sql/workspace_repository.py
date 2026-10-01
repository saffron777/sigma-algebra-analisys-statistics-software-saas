"""The per-module workspace and the sync revisions on any SQL engine."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from sigma.domain.workspace import UNSET, WorkspaceEntry
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import now_ms, upsert

_w = schema.workspace


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class SqlWorkspaceRepository:
    """One row per (user, module). Revisions only ever grow, so a client can tell any change from its own writes."""

    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def get(self, module: str) -> WorkspaceEntry:
        with self._db.begin() as conn:
            row = conn.execute(select(_w).where(_w.c.user_id == self._user, _w.c.module == module)).first()
        if row is None:
            return WorkspaceEntry(module)
        return WorkspaceEntry(module, _load(row.state), _load(row.payload), row.revision, row.updated_at)

    def save(self, module: str, state: Any = UNSET, payload: Any = UNSET) -> WorkspaceEntry:
        """Partial update: a field left as ``UNSET`` keeps its stored value; ``None`` clears it."""
        updated = now_ms()
        insert_values: dict[str, Any] = {"state": None, "payload": None, "revision": 1, "updated_at": updated}
        update_values: dict[str, Any] = {"revision": _w.c.revision + 1, "updated_at": updated}
        if state is not UNSET:
            insert_values["state"] = update_values["state"] = _dump(state)
        if payload is not UNSET:
            insert_values["payload"] = update_values["payload"] = _dump(payload)
        with self._db.begin() as conn:
            upsert(conn, _w, {"user_id": self._user, "module": module}, insert_values, update_values)
        return self.get(module)

    def clear(self, module: str) -> int:
        """Empty a module but keep the row, so its revision keeps growing (other windows notice the change)."""
        return self._clear([module])[module]

    def clear_all(self) -> dict[str, int]:
        with self._db.begin() as conn:
            modules = [r.module for r in conn.execute(select(_w.c.module).where(_w.c.user_id == self._user))]
        return self._clear(modules)

    def _clear(self, modules: list[str]) -> dict[str, int]:
        revisions: dict[str, int] = {}
        with self._db.begin() as conn:
            for module in modules:
                upsert(
                    conn,
                    _w,
                    {"user_id": self._user, "module": module},
                    {"state": None, "payload": None, "revision": 1, "updated_at": now_ms()},
                    {"state": None, "payload": None, "revision": _w.c.revision + 1, "updated_at": now_ms()},
                )
                revisions[module] = conn.execute(
                    select(_w.c.revision).where(_w.c.user_id == self._user, _w.c.module == module)
                ).scalar_one()
        return revisions

    def revisions(self) -> dict[str, int]:
        with self._db.begin() as conn:
            rows = conn.execute(select(_w.c.module, _w.c.revision).where(_w.c.user_id == self._user)).all()
        return {row.module: row.revision for row in rows}


class SqlSyncRepository:
    """Per-scope revisions raised by the repositories (datasets, results…): the cheap «did anything change?» check."""

    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def revisions(self) -> dict[str, int]:
        t = schema.sync_state
        with self._db.begin() as conn:
            rows = conn.execute(select(t.c.scope, t.c.revision).where(t.c.user_id == self._user)).all()
        return {row.scope: row.revision for row in rows}
