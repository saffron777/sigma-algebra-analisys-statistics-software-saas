"""Password hashing with werkzeug (scrypt, salted, constant-time verification)."""

from __future__ import annotations

from werkzeug.security import check_password_hash, generate_password_hash


class WerkzeugPasswordHasher:
    def __init__(self) -> None:
        # a real hash of a throw-away value: verifying against it takes as long as a real check, so
        # unknown usernames cannot be told apart from wrong passwords by response time
        self.dummy_hash = generate_password_hash("sigma-timing-equalizer-not-a-password", method="scrypt")

    def hash(self, password: str) -> str:
        return generate_password_hash(password, method="scrypt")

    def verify(self, password_hash: str, password: str) -> bool:
        try:
            return check_password_hash(password_hash, password)
        except (ValueError, TypeError):
            return False
