"""Use cases of the database administration page and of the first-run wizard's «Base de datos» step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sigma.application.ports import DatabaseAdministration


class GetDatabaseStatus:
    def __init__(self, administration: DatabaseAdministration) -> None:
        self._administration = administration

    def execute(self) -> dict[str, Any]:
        return self._administration.status()


class TestDatabase:
    __test__ = False  # not a pytest class

    def __init__(self, administration: DatabaseAdministration) -> None:
        self._administration = administration

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._administration.test(payload)


class ApplyDatabase:
    def __init__(self, administration: DatabaseAdministration) -> None:
        self._administration = administration

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._administration.apply(payload)


class GetMigrationStatus:
    def __init__(self, administration: DatabaseAdministration) -> None:
        self._administration = administration

    def execute(self) -> dict[str, Any]:
        return self._administration.migration_status()
