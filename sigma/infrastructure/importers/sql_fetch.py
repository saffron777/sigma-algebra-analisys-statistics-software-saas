"""Reads the result of one SQL statement into a DataFrame, in batches with a row cap (shared by SQL sources)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection, Engine

FETCH_SIZE = 50_000
QUERY_TIMEOUT_SECONDS = 120


def _apply_query_timeout(conn: Connection, seconds: int) -> None:
    """Ask the driver for a per-call timeout where it has one (pyodbc ``timeout``, python-oracledb ``call_timeout``)."""
    raw = getattr(conn.connection, "driver_connection", None)
    if raw is None:
        return
    if hasattr(raw, "timeout"):
        raw.timeout = seconds
    elif hasattr(raw, "call_timeout"):
        raw.call_timeout = seconds * 1000


def fetch_rows(
    engine: Engine,
    sql: str,
    max_rows: int,
    *,
    timeout_seconds: int = QUERY_TIMEOUT_SECONDS,
    before: Callable[[Connection], None] | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Up to ``max_rows`` rows of ``sql`` and whether there were more. Nothing is ever committed.

    ``before`` runs on the connection first (e.g. Oracle's ``SET TRANSACTION READ ONLY``).
    """
    with engine.connect() as conn:
        _apply_query_timeout(conn, timeout_seconds)
        try:
            if before is not None:
                before(conn)
            cursor = conn.exec_driver_sql(sql)
            columns = list(cursor.keys())
            rows: list[tuple[Any, ...]] = []
            while len(rows) <= max_rows:
                batch = cursor.fetchmany(FETCH_SIZE)
                if not batch:
                    break
                rows.extend(tuple(row) for row in batch)
        finally:
            conn.rollback()  # nothing is ever committed from an import
    truncated = len(rows) > max_rows
    return pd.DataFrame.from_records(rows[:max_rows], columns=columns, coerce_float=True), truncated
