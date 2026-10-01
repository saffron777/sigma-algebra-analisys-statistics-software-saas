"""SQLite stores for the server-wide settings (``app_settings``) and the AI usage log (``ai_usage``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sigma.infrastructure.persistence.sqlite.database import Database


class SqliteSettingsRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._db.connection() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


class SqliteAiUsageRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, user_id: int, module: str, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO ai_usage (user_id, module, model, input_tokens, output_tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    module,
                    model,
                    int(input_tokens),
                    int(output_tokens),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )

    def count_since(self, user_id: int, since_iso: str) -> int:
        with self._db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM ai_usage WHERE user_id = ? AND created_at >= ?", (user_id, since_iso)
            ).fetchone()[0]

    def oldest_since(self, user_id: int, since_iso: str) -> str | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT MIN(created_at) FROM ai_usage WHERE user_id = ? AND created_at >= ?", (user_id, since_iso)
            ).fetchone()
        return row[0]

    def summary(self, since_iso: str) -> list[dict[str, Any]]:
        """Requests and tokens per account since ``since_iso`` (for the administrator's page)."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT u.id AS user_id, u.username, COUNT(a.id) AS requests,"
                " COALESCE(SUM(a.input_tokens), 0) AS input_tokens, COALESCE(SUM(a.output_tokens), 0) AS output_tokens"
                " FROM users u LEFT JOIN ai_usage a ON a.user_id = u.id AND a.created_at >= ?"
                " GROUP BY u.id ORDER BY requests DESC, u.username",
                (since_iso,),
            ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]  # noqa: SIM118
