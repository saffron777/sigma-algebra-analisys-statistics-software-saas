"""User accounts on any SQL engine."""

from __future__ import annotations

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Row
from sqlalchemy.exc import IntegrityError

from sigma.domain.errors import InvalidInputError, UserNotFoundError
from sigma.domain.users import User
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import now_iso

_u = schema.users
_PUBLIC = [
    _u.c.id,
    _u.c.username,
    _u.c.display_name,
    _u.c.role,
    _u.c.can_use_console,
    _u.c.can_use_ai,
    _u.c.is_active,
    _u.c.auth_epoch,
    _u.c.created_at,
    _u.c.last_login_at,
]


def _to_user(row: Row) -> User:
    return User(
        id=row.id,
        username=row.username,
        display_name=row.display_name,
        role=row.role,
        can_use_console=bool(row.can_use_console),
        can_use_ai=bool(row.can_use_ai),
        is_active=bool(row.is_active),
        auth_epoch=row.auth_epoch,
        created_at=row.created_at,
        last_login_at=row.last_login_at,
    )


class SqlUserRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def count(self) -> int:
        with self._db.begin() as conn:
            return int(conn.execute(select(func.count()).select_from(_u)).scalar_one())

    def count_active_admins(self, excluding: int | None = None) -> int:
        query = select(func.count()).select_from(_u).where(_u.c.role == "admin", _u.c.is_active == 1)
        if excluding:
            query = query.where(_u.c.id != excluding)
        with self._db.begin() as conn:
            return int(conn.execute(query).scalar_one())

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
            with self._db.begin() as conn:
                result = conn.execute(
                    insert(_u).values(
                        username=username,
                        username_key=username.lower(),
                        display_name=display_name,
                        password_hash=password_hash,
                        role=role,
                        can_use_console=int(can_use_console),
                        can_use_ai=int(can_use_ai),
                        is_active=1,
                        auth_epoch=1,
                        created_at=now_iso(),
                    )
                )
                user_id = int(result.inserted_primary_key[0])
        except IntegrityError:
            raise InvalidInputError(f"Ya existe un usuario llamado «{username}».") from None
        return self.get(user_id)

    def get(self, user_id: int) -> User:
        with self._db.begin() as conn:
            row = conn.execute(select(*_PUBLIC).where(_u.c.id == user_id)).first()
        if row is None:
            raise UserNotFoundError("El usuario no existe.")
        return _to_user(row)

    def find_with_hash(self, username: str) -> tuple[User, str] | None:
        """Used only by authentication: the stored hash never leaves this method otherwise."""
        key = (username or "").strip().lower()
        with self._db.begin() as conn:
            row = conn.execute(select(*_PUBLIC, _u.c.password_hash).where(_u.c.username_key == key)).first()
        return (_to_user(row), row.password_hash) if row else None

    def list(self) -> list[User]:
        with self._db.begin() as conn:
            rows = conn.execute(select(*_PUBLIC).order_by(_u.c.role, _u.c.username)).all()
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
        changes: dict[str, object] = {
            "display_name": display_name,
            "role": role,
            "is_active": None if is_active is None else int(is_active),
            "can_use_console": None if can_use_console is None else int(can_use_console),
            "can_use_ai": None if can_use_ai is None else int(can_use_ai),
        }
        changes = {column: value for column, value in changes.items() if value is not None}
        self.get(user_id)
        if changes:
            if "is_active" in changes or "role" in changes:
                changes["auth_epoch"] = _u.c.auth_epoch + 1  # their open sessions stop being valid
            with self._db.begin() as conn:
                conn.execute(update(_u).where(_u.c.id == user_id).values(**changes))
        return self.get(user_id)

    def set_password(self, user_id: int, password_hash: str) -> User:
        self.get(user_id)
        with self._db.begin() as conn:
            conn.execute(
                update(_u).where(_u.c.id == user_id).values(password_hash=password_hash, auth_epoch=_u.c.auth_epoch + 1)
            )
        return self.get(user_id)

    def touch_login(self, user_id: int) -> None:
        with self._db.begin() as conn:
            conn.execute(update(_u).where(_u.c.id == user_id).values(last_login_at=now_iso()))

    def delete(self, user_id: int) -> None:
        self.get(user_id)
        with self._db.begin() as conn:
            conn.execute(delete(_u).where(_u.c.id == user_id))  # their rows go by ON DELETE CASCADE
