"""Chooses the persistence adapters for a database setting."""

from __future__ import annotations

from sigma.application.ports import Backend
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.backend import SqlBackend
from sigma.infrastructure.persistence.sql.engine import Database


def create_backend(settings: DatabaseSettings) -> Backend:
    """Connect (creating or upgrading the schema when needed) and return the backend for ``settings``."""
    settings.validate()
    return SqlBackend(Database(settings))
