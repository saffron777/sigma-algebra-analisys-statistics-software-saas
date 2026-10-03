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


# ------------------------------------------------------------------ engine-neutral checks on a running container
def count(container: Any, table: str, **where: Any) -> int:
    """Rows of ``table`` matching the equality filters, on whichever engine the application runs."""
    if container.database is not None:
        clause = " AND ".join(f"{column} = :{column}" for column in where)
        return int(
            scalar(container.database, f"SELECT COUNT(*) FROM {table}" + (f" WHERE {clause}" if where else ""), **where)
        )
    return container.backend.core.col(table).count_documents(where)


def dump(container: Any, tables: tuple[str, ...]) -> str:
    """Every stored value of the given tables as one string (to prove a secret was never written)."""
    if container.database is not None:
        return " ".join(
            str(v) for t in tables for row in fetch_all(container.database, f"SELECT * FROM {t}") for v in row
        )
    return " ".join(str(list(container.backend.core.col(t).find())) for t in tables)


def data_tables(container: Any, prefix: str) -> list[str]:
    """Physical data tables (``ds_…`` / ``stg_…``) still present; for MongoDB, the row groups of that kind."""
    if container.database is not None:
        return table_names(container.database, prefix)
    owners = container.backend.core.col("frame_chunks").distinct("owner")
    return sorted(o for o in owners if o.startswith(prefix))
