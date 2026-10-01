"""Helpers shared by tests that build repositories directly (without the web app)."""

from __future__ import annotations

from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.user_repository import SqliteUserRepository


def add_user(database: Database, username: str = "tester", role: str = "user") -> int:
    """Insert an account (the password hash is irrelevant here) and return its id."""
    return SqliteUserRepository(database).add(username, username.title(), "x", role, True, True).id
