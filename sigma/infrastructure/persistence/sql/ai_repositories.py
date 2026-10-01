"""The server-wide settings (``app_settings``) and the AI usage log (``ai_usage``) on any SQL engine."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, insert, select

from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import now_iso, upsert

_s = schema.app_settings
_a = schema.ai_usage
_u = schema.users


class SqlSettingsRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._db.begin() as conn:
            row = conn.execute(select(_s.c.value).where(_s.c.key == key)).first()
        return row.value if row else default

    def set(self, key: str, value: str) -> None:
        with self._db.begin() as conn:
            upsert(conn, _s, {"key": key}, {"value": value}, {"value": value})


class SqlAiUsageRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, user_id: int, module: str, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._db.begin() as conn:
            conn.execute(
                insert(_a).values(
                    user_id=user_id,
                    module=module,
                    model=model,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    created_at=now_iso(),
                )
            )

    def count_since(self, user_id: int, since_iso: str) -> int:
        with self._db.begin() as conn:
            return int(
                conn.execute(
                    select(func.count()).select_from(_a).where(_a.c.user_id == user_id, _a.c.created_at >= since_iso)
                ).scalar_one()
            )

    def oldest_since(self, user_id: int, since_iso: str) -> str | None:
        with self._db.begin() as conn:
            return conn.execute(
                select(func.min(_a.c.created_at)).where(_a.c.user_id == user_id, _a.c.created_at >= since_iso)
            ).scalar_one()

    def summary(self, since_iso: str) -> list[dict[str, Any]]:
        """Requests and tokens per account since ``since_iso`` (for the administrator's page)."""
        recent = (
            select(
                _a.c.user_id,
                func.count().label("requests"),
                func.sum(_a.c.input_tokens).label("input_tokens"),
                func.sum(_a.c.output_tokens).label("output_tokens"),
            )
            .where(_a.c.created_at >= since_iso)
            .group_by(_a.c.user_id)
            .subquery()
        )
        with self._db.begin() as conn:
            rows = conn.execute(
                select(
                    _u.c.id.label("user_id"),
                    _u.c.username,
                    func.coalesce(recent.c.requests, 0).label("requests"),
                    func.coalesce(recent.c.input_tokens, 0).label("input_tokens"),
                    func.coalesce(recent.c.output_tokens, 0).label("output_tokens"),
                )
                .select_from(_u.outerjoin(recent, recent.c.user_id == _u.c.id))
                .order_by(func.coalesce(recent.c.requests, 0).desc(), _u.c.username)
            ).all()
        return [
            {
                "user_id": row.user_id,
                "username": row.username,
                "requests": int(row.requests),
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
            }
            for row in rows
        ]
