"""SQLite connection factory and schema migrations (versioned with ``PRAGMA user_version``)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_MIGRATIONS: tuple[str, ...] = (
    # v1: datasets metadata + saved analysis results (dataset rows live in per-dataset ds_<id> tables)
    """
    CREATE TABLE datasets (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
        source_type  TEXT NOT NULL,
        source_meta  TEXT NOT NULL DEFAULT '{}',
        n_rows       INTEGER NOT NULL,
        n_cols       INTEGER NOT NULL,
        dtypes       TEXT NOT NULL,
        columns_info TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );
    CREATE TABLE saved_results (
        id            TEXT PRIMARY KEY,
        module        TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        title         TEXT NOT NULL,
        notes         TEXT NOT NULL DEFAULT '',
        dataset_id    TEXT REFERENCES datasets(id) ON DELETE SET NULL,
        dataset_name  TEXT,
        params        TEXT NOT NULL,
        result        TEXT NOT NULL,
        report        TEXT NOT NULL,
        figures       TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX idx_saved_results_module  ON saved_results(module);
    CREATE INDEX idx_saved_results_created ON saved_results(created_at DESC);
    """,
)


class Database:
    """Opens short-lived connections to the Sigma SQLite file and keeps its schema current."""

    def __init__(self, path: str | Path, busy_timeout_seconds: float = 30.0) -> None:
        self.path = Path(path)
        self._timeout = busy_timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=self._timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for target, script in enumerate(_MIGRATIONS, start=1):
                if target > version:
                    conn.executescript(script)
                    conn.execute(f"PRAGMA user_version = {target}")
