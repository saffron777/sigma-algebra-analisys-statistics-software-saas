"""SQLite-backed user accounts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from sigma.domain.errors import InvalidInputError, UserNotFoundError
from sigma.domain.users import User
from sigma.infrastructure.persistence.sqlite.database import Database

_COLUMNS = (
    "id, username, display_name, role, can_use_console, can_use_ai, is_active, auth_epoch, created_at, last_login_at"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        role=row["role"],
        can_use_console=bool(row["can_use_console"]),
        can_use_ai=bool(row["can_use_ai"]),
        is_active=bool(row["is_active"]),
        auth_epoch=row["auth_epoch"],
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


class SqliteUserRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def count(self) -> int:
        with self._db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def count_active_admins(self, excluding: int | None = None) -> int:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?", (excluding or 0,)
            ).fetchone()
        return int(row[0])

    def add(
        self,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        can_use_console: bool,
        can_use_ai: bool,
    ) -> User:
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, display_name, password_hash, role, can_use_console, can_use_ai,"
                    " is_active, auth_epoch, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?)",
                    (username, display_name, password_hash, role, int(can_use_console), int(can_use_ai), _now()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise InvalidInputError(f"Ya existe un usuario llamado «{username}».") from None
        return self.get(int(user_id))

    def get(self, user_id: int) -> User:
        with self._db.connection() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise UserNotFoundError("El usuario no existe.")
        return _to_user(row)

    def find_with_hash(self, username: str) -> tuple[User, str] | None:
        """Used only by authentication: the stored hash never leaves this method otherwise."""
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS}, password_hash FROM users WHERE username = ?", ((username or "").strip(),)
            ).fetchone()
        return (_to_user(row), row["password_hash"]) if row else None

    def list(self) -> list[User]:
        with self._db.connection() as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM users ORDER BY role, username").fetchall()
        return [_to_user(row) for row in rows]

    def update(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
        can_use_console: bool | None = None,
        can_use_ai: bool | None = None,
    ) -> User:
        changes = {
            "display_name": display_name,
            "role": role,
            "is_active": None if is_active is None else int(is_active),
            "can_use_console": None if can_use_console is None else int(can_use_console),
            "can_use_ai": None if can_use_ai is None else int(can_use_ai),
        }
        changes = {column: value for column, value in changes.items() if value is not None}
        self.get(user_id)
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)  # column names come from the dict above
            bump = ", auth_epoch = auth_epoch + 1" if "is_active" in changes or "role" in changes else ""
            with self._db.connection() as conn:
                conn.execute(f"UPDATE users SET {assignments}{bump} WHERE id = ?", (*changes.values(), user_id))
        return self.get(user_id)

    def set_password(self, user_id: int, password_hash: str) -> User:
        self.get(user_id)
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, auth_epoch = auth_epoch + 1 WHERE id = ?", (password_hash, user_id)
            )
        return self.get(user_id)

    def touch_login(self, user_id: int) -> None:
        with self._db.connection() as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), user_id))

    def delete(self, user_id: int) -> None:
        self.get(user_id)
        with self._db.connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
