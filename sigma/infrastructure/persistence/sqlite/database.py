"""SQLite connection factory; the schema itself is versioned in ``migrations.py``."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sigma.infrastructure.persistence.sqlite.migrations import MIGRATIONS


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
        """Apply pending migrations, each one atomically and with foreign keys off (needed to rebuild tables)."""
        conn = sqlite3.connect(self.path, timeout=self._timeout)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= len(MIGRATIONS):
                return
            conn.execute("PRAGMA foreign_keys = OFF")
            for target, script in enumerate(MIGRATIONS, start=1):
                if target <= version:
                    continue
                try:
                    conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;")
                except sqlite3.Error:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(f"La migración dejó {len(violations)} referencias huérfanas.")
        finally:
            conn.close()
