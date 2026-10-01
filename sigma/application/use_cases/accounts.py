"""Use cases for user accounts: first-run setup, sign-in and account administration."""

from __future__ import annotations

from sigma.application.ports import (
    LegacyDataAdopter,
    LoginThrottle,
    PasswordHasher,
    UserDataPurger,
    UserRepository,
)
from sigma.domain.errors import (
    AuthenticationError,
    InvalidInputError,
    PermissionDeniedError,
    UserNotFoundError,
)
from sigma.domain.users import (
    ROLE_ADMIN,
    ROLE_USER,
    User,
    validate_display_name,
    validate_password,
    validate_role,
    validate_username,
)

_BAD_CREDENTIALS = "Usuario o contraseña incorrectos."


class GetSetupState:
    def __init__(self, users: UserRepository) -> None:
        self._users = users
        self._done = False  # accounts can never drop back to zero (the last administrator is protected)

    def setup_required(self) -> bool:
        if not self._done:
            self._done = self._users.count() > 0
        return not self._done


class CreateFirstAdmin:
    """Creates the very first administrator and hands it the data that existed before accounts did."""

    def __init__(self, users: UserRepository, hasher: PasswordHasher, adopter: LegacyDataAdopter) -> None:
        self._users = users
        self._hasher = hasher
        self._adopter = adopter

    def execute(self, username: str, display_name: str, password: str) -> tuple[User, dict[str, int]]:
        if self._users.count() > 0:
            raise PermissionDeniedError("La configuración inicial ya se realizó.")
        username = validate_username(username)
        user = self._users.add(
            username,
            validate_display_name(display_name, username),
            self._hasher.hash(validate_password(password, username)),
            ROLE_ADMIN,
            can_use_console=True,
            can_use_ai=True,
        )
        adopted = self._adopter.adopt(user.id)
        self._users.touch_login(user.id)  # setup signs the administrator in
        return self._users.get(user.id), adopted


class Authenticate:
    def __init__(self, users: UserRepository, hasher: PasswordHasher, throttle: LoginThrottle) -> None:
        self._users = users
        self._hasher = hasher
        self._throttle = throttle

    def execute(self, username: str, password: str, client_ip: str) -> User:
        username = (username or "").strip()
        self._throttle.check(username, client_ip)
        found = self._users.find_with_hash(username)
        # always verify against *some* hash so response time does not reveal whether the user exists
        password_ok = self._hasher.verify(found[1] if found else self._hasher.dummy_hash, password or "")
        if not found or not password_ok or not found[0].is_active:
            self._throttle.record_failure(username, client_ip)
            raise AuthenticationError(_BAD_CREDENTIALS)
        self._throttle.reset(username, client_ip)
        self._users.touch_login(found[0].id)
        return self._users.get(found[0].id)


class GetSessionUser:
    """Resolves the user behind a session cookie; ``None`` if the account changed since (epoch) or is disabled."""

    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def execute(self, user_id: int, epoch: int) -> User | None:
        try:
            user = self._users.get(user_id)
        except UserNotFoundError:
            return None
        return user if user.is_active and user.auth_epoch == epoch else None


class ListUsers:
    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def execute(self) -> list[User]:
        return self._users.list()


class CreateUser:
    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users = users
        self._hasher = hasher

    def execute(
        self,
        username: str,
        display_name: str,
        password: str,
        role: str = ROLE_USER,
        can_use_console: bool = False,
        can_use_ai: bool = False,
    ) -> User:
        username = validate_username(username)
        return self._users.add(
            username,
            validate_display_name(display_name, username),
            self._hasher.hash(validate_password(password, username)),
            validate_role(role),
            can_use_console,
            can_use_ai,
        )


class UpdateUser:
    """Administrator edits of another account (role, active flag, permissions, display name)."""

    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def execute(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
        can_use_console: bool | None = None,
        can_use_ai: bool | None = None,
    ) -> User:
        target = self._users.get(user_id)
        if role is not None:
            validate_role(role)
        loses_admin = target.is_admin and target.is_active and (role == ROLE_USER or is_active is False)
        if loses_admin and self._users.count_active_admins(excluding=user_id) == 0:
            raise InvalidInputError("Debe quedar al menos un administrador activo.")
        return self._users.update(
            user_id,
            display_name=validate_display_name(display_name, target.username) if display_name is not None else None,
            role=role,
            is_active=is_active,
            can_use_console=can_use_console,
            can_use_ai=can_use_ai,
        )


class ResetPassword:
    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users = users
        self._hasher = hasher

    def execute(self, user_id: int, new_password: str) -> User:
        target = self._users.get(user_id)
        return self._users.set_password(user_id, self._hasher.hash(validate_password(new_password, target.username)))


class ChangeOwnPassword:
    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users = users
        self._hasher = hasher

    def execute(self, user: User, current_password: str, new_password: str) -> User:
        found = self._users.find_with_hash(user.username)
        if not found or not self._hasher.verify(found[1], current_password or ""):
            raise InvalidInputError("La contraseña actual no es correcta.")
        if self._hasher.verify(found[1], new_password or ""):
            raise InvalidInputError("La nueva contraseña debe ser distinta de la actual.")
        return self._users.set_password(user.id, self._hasher.hash(validate_password(new_password, user.username)))


class UpdateProfile:
    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def execute(self, user: User, display_name: str) -> User:
        return self._users.update(user.id, display_name=validate_display_name(display_name, user.username))


class DeleteUser:
    def __init__(self, users: UserRepository, purger: UserDataPurger) -> None:
        self._users = users
        self._purger = purger

    def execute(self, actor: User, user_id: int) -> None:
        if actor.id == user_id:
            raise InvalidInputError("No puede eliminar su propia cuenta.")
        target = self._users.get(user_id)
        if target.is_admin and target.is_active and self._users.count_active_admins(excluding=user_id) == 0:
            raise InvalidInputError("Debe quedar al menos un administrador activo.")
        self._purger.purge(user_id)
        self._users.delete(user_id)
