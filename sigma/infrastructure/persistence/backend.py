"""Chooses the persistence adapters for a database setting."""

from __future__ import annotations

from sigma.application.ports import Backend
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.backend import SqlBackend
from sigma.infrastructure.persistence.sql.engine import Database


def create_backend(settings: DatabaseSettings) -> Backend:
    """Connect (creating or upgrading the schema when needed) and return the backend for ``settings``."""
    settings.validate()
    if settings.engine == "mongodb":
        from sigma.infrastructure.persistence.mongo.backend import MongoBackend
        from sigma.infrastructure.persistence.mongo.core import connect

        return MongoBackend(connect(settings))
    return SqlBackend(Database(settings))
