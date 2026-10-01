"""User accounts: entity and the rules for usernames and passwords (pure, no I/O)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sigma.domain.errors import InvalidInputError

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLES = (ROLE_ADMIN, ROLE_USER)

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 200
_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{2,39}$")
_TRIVIAL_PASSWORDS = frozenset(
    {
        "1234567890", "0123456789", "12345678910", "qwertyuiop", "qwerty12345", "password12", "password123",
        "contrasena", "contrasena1", "contraseña", "admin12345", "administrador", "sigma12345", "letmein1234",
        "iloveyou123", "abcdefghij", "1q2w3e4r5t", "passw0rd123",
    }
)  # fmt: skip


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    role: str
    can_use_console: bool
    is_active: bool
    auth_epoch: int
    created_at: str
    last_login_at: str | None = None
    can_use_ai: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


def validate_username(username: str) -> str:
    cleaned = (username or "").strip()
    if not _USERNAME.match(cleaned):
        raise InvalidInputError(
            "El usuario debe tener entre 3 y 40 caracteres (letras, números, punto, guion, guion bajo o @) "
            "y empezar por una letra o un número."
        )
    return cleaned


def validate_display_name(name: str, fallback: str) -> str:
    cleaned = " ".join((name or "").split())[:80]
    return cleaned or fallback


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise InvalidInputError("El rol debe ser «admin» o «user».")
    return role


def validate_password(password: str, username: str = "") -> str:
    """Return the password if it is acceptable, otherwise raise with a message the user can act on."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidInputError(f"La contraseña debe tener al menos {MIN_PASSWORD_LENGTH} caracteres.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise InvalidInputError(f"La contraseña no puede superar {MAX_PASSWORD_LENGTH} caracteres.")
    lowered = password.lower()
    if lowered in _TRIVIAL_PASSWORDS or len(set(password)) < 4:
        raise InvalidInputError("La contraseña es demasiado común o repetitiva. Elija otra.")
    name = (username or "").strip().lower()
    if name and (name in lowered or lowered in name):
        raise InvalidInputError("La contraseña no puede contener el nombre de usuario.")
    return password
