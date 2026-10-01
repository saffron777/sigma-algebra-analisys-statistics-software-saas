"""SQLite stores for the per-module workspace and the change revisions used for live synchronisation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sigma.domain.workspace import UNSET, WorkspaceEntry
from sigma.infrastructure.persistence.sqlite.database import Database


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class SqliteWorkspaceRepository:
    """One row per (user, module). Revisions only ever grow, so a client can tell any change from its own writes."""

    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def get(self, module: str) -> WorkspaceEntry:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT state, payload, revision, updated_at FROM workspace WHERE user_id = ? AND module = ?",
                (self._user, module),
            ).fetchone()
        if row is None:
            return WorkspaceEntry(module)
        return WorkspaceEntry(module, _load(row["state"]), _load(row["payload"]), row["revision"], row["updated_at"])

    def save(self, module: str, state: Any = UNSET, payload: Any = UNSET) -> WorkspaceEntry:
        """Partial update: a field left as ``UNSET`` keeps its stored value; ``None`` clears it."""
        with self._db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")  # read-modify-write must not interleave with another window's save
            row = conn.execute(
                "SELECT state, payload, revision FROM workspace WHERE user_id = ? AND module = ?",
                (self._user, module),
            ).fetchone()
            new_state = _dump(state) if state is not UNSET else (row["state"] if row else None)
            new_payload = _dump(payload) if payload is not UNSET else (row["payload"] if row else None)
            revision = (row["revision"] if row else 0) + 1
            updated = _now()
            conn.execute(
                "INSERT INTO workspace (user_id, module, state, payload, revision, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (user_id, module) DO UPDATE SET state = excluded.state, payload = excluded.payload,"
                " revision = excluded.revision, updated_at = excluded.updated_at",
                (self._user, module, new_state, new_payload, revision, updated),
            )
        return WorkspaceEntry(module, _load(new_state), _load(new_payload), revision, updated)

    def clear(self, module: str) -> int:
        """Empty a module but keep the row, so its revision keeps growing (other windows notice the change)."""
        return self._clear([module])[module]

    def clear_all(self) -> dict[str, int]:
        with self._db.connection() as conn:
            modules = [r[0] for r in conn.execute("SELECT module FROM workspace WHERE user_id = ?", (self._user,))]
        return self._clear(modules)

    def _clear(self, modules: list[str]) -> dict[str, int]:
        revisions: dict[str, int] = {}
        with self._db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for module in modules:
                conn.execute(
                    "INSERT INTO workspace (user_id, module, state, payload, revision, updated_at)"
                    " VALUES (?, ?, NULL, NULL, 1, ?)"
                    " ON CONFLICT (user_id, module) DO UPDATE SET state = NULL, payload = NULL,"
                    " revision = revision + 1, updated_at = excluded.updated_at",
                    (self._user, module, _now()),
                )
                revisions[module] = conn.execute(
                    "SELECT revision FROM workspace WHERE user_id = ? AND module = ?", (self._user, module)
                ).fetchone()[0]
        return revisions

    def revisions(self) -> dict[str, int]:
        with self._db.connection() as conn:
            rows = conn.execute("SELECT module, revision FROM workspace WHERE user_id = ?", (self._user,)).fetchall()
        return {row["module"]: row["revision"] for row in rows}


class SqliteSyncRepository:
    """Per-scope revisions bumped by database triggers (datasets, results…): the cheap «did anything change?» check."""

    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def revisions(self) -> dict[str, int]:
        with self._db.connection() as conn:
            rows = conn.execute("SELECT scope, revision FROM sync_state WHERE user_id = ?", (self._user,)).fetchall()
        return {row["scope"]: row["revision"] for row in rows}
