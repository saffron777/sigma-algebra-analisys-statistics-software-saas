"""Helpers shared by tests that build repositories directly (without the web app)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect, text

from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.user_repository import SqlUserRepository


def add_user(database: Database, username: str = "tester", role: str = "user") -> int:
    """Insert an account (the password hash is irrelevant here) and return its id."""
    return SqlUserRepository(database).add(username, username.title(), "x", role, True, True).id


# ------------------------------------------------------------------ raw SQL for assertions (portable: named binds only)
def scalar(database: Database, sql: str, **params: Any) -> Any:
    with database.begin() as conn:
        return conn.execute(text(sql), params).scalar()


def fetch_all(database: Database, sql: str, **params: Any) -> list[tuple]:
    with database.begin() as conn:
        return [tuple(row) for row in conn.execute(text(sql), params)]


def execute(database: Database, sql: str, **params: Any) -> None:
    with database.begin() as conn:
        conn.execute(text(sql), params)


def table_names(database: Database, prefix: str = "") -> list[str]:
    return sorted(name for name in inspect(database.engine).get_table_names() if name.startswith(prefix))
