"""Copying everything Sigma stores from one database to another (SQLite ⇄ SQL Server ⇄ MySQL ⇄ … ⇄ MongoDB)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from sigma.application.ports import DataTransfer
from sigma.domain.database import MIGRATION_ENTITIES
from sigma.domain.errors import InvalidInputError, MigrationError, SigmaError

# progress(entity key, label, copied so far, total expected)
Progress = Callable[[str, str, int, int], None]


@dataclass(frozen=True)
class MigrationReport:
    copied: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)  # records that could not be read (e.g. a broken .ipynb)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"copied": self.copied, "skipped": self.skipped, "seconds": round(self.seconds, 1)}


class MigrateData:
    """Copies every entity, ids included, into an empty target and verifies the counts; the source is only read.

    If anything fails the target is emptied again (it was empty when we began), so the migration can be retried.
    """

    def execute(self, source: DataTransfer, target: DataTransfer, progress: Progress | None = None) -> MigrationReport:
        if not target.is_empty():
            raise InvalidInputError(
                "La base de datos de destino ya contiene datos de Sigma. Para copiar, elija una base vacía."
            )
        started = time.monotonic()
        expected = source.counts()
        copied: dict[str, int] = {}
        try:
            for key, label in MIGRATION_ENTITIES:
                total = expected.get(key, 0)
                if progress:
                    progress(key, label, 0, total)
                copied[key] = target.write(key, self._counted(source.read(key), key, label, total, progress))
                if progress:
                    progress(key, label, copied[key], total)
            target.finish()
            self._verify(target, copied, expected)
        except BaseException as error:
            target.clear()
            if isinstance(error, SigmaError):
                raise
            raise MigrationError(f"La migración falló y el destino quedó vacío: {error}") from error
        skipped = {k: expected.get(k, 0) - n for k, n in copied.items() if expected.get(k, 0) > n}
        return MigrationReport(copied, skipped, time.monotonic() - started)

    @staticmethod
    def _counted(
        records: Iterator[dict[str, Any]], key: str, label: str, total: int, progress: Progress | None
    ) -> Iterator[dict[str, Any]]:
        for done, record in enumerate(records, start=1):
            yield record
            if progress and (done == total or done % 20 == 0):
                progress(key, label, done, total)

    @staticmethod
    def _verify(target: DataTransfer, copied: dict[str, int], expected: dict[str, int]) -> None:
        stored = target.counts()
        for key, label in MIGRATION_ENTITIES:
            if stored.get(key, 0) != copied[key]:
                raise MigrationError(f"{label}: se copiaron {copied[key]} pero el destino tiene {stored.get(key, 0)}.")
            if key != "notebooks" and copied[key] != expected.get(key, 0):
                raise MigrationError(f"{label}: el origen tiene {expected.get(key, 0)} y se copiaron {copied[key]}.")
