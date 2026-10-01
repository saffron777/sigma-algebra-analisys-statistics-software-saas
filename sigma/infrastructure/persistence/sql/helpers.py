"""Small dialect-aware building blocks shared by the SQL repositories."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Table, and_, insert, update
from sqlalchemy.dialects import mysql as mysql_dialect
from sqlalchemy.dialects import postgresql as pg_dialect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from sigma.infrastructure.persistence.sql import schema


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def now_ms() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def like_pattern(text: str) -> str:
    """``%text%``, lower-cased, with LIKE metacharacters escaped (compare ``lower(column)`` using ``escape``)."""
    escaped = text.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def upsert(
    conn: Connection,
    table: Table,
    key: Mapping[str, Any],
    insert_values: Mapping[str, Any],
    update_values: Mapping[str, Any],
) -> None:
    """Insert ``key + insert_values`` or, if a row with ``key`` exists, apply ``update_values`` (SQL expressions ok).

    ``key`` must be the primary key / a unique index of ``table``. SQLite, PostgreSQL and MySQL/MariaDB do it in one
    atomic statement; SQL Server and Oracle update first and insert if nothing matched (retrying once on a race).
    """
    values = {**key, **insert_values}
    dialect = conn.dialect.name
    if dialect in ("sqlite", "postgresql"):
        module = sqlite_dialect if dialect == "sqlite" else pg_dialect
        statement = module.insert(table).values(values)
        conn.execute(
            statement.on_conflict_do_update(index_elements=[table.c[name] for name in key], set_=dict(update_values))
        )
    elif dialect in ("mysql", "mariadb"):
        conn.execute(mysql_dialect.insert(table).values(values).on_duplicate_key_update(**dict(update_values)))
    else:
        where = and_(*(table.c[name] == value for name, value in key.items()))
        if conn.execute(update(table).where(where).values(**dict(update_values))).rowcount:
            return
        try:
            conn.execute(insert(table).values(values))
        except IntegrityError:  # somebody inserted it between our UPDATE and INSERT
            conn.execute(update(table).where(where).values(**dict(update_values)))


def bump_sync(conn: Connection, user_id: int | None, scope: str) -> None:
    """Raise a user's revision of ``scope`` (datasets, results, drafts, templates): what other windows poll."""
    if user_id is None:
        return
    table = schema.sync_state
    upsert(conn, table, {"user_id": user_id, "scope": scope}, {"revision": 1}, {"revision": table.c.revision + 1})
