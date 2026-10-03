"""What the administration pages do with the database setting: show it, test a candidate, switch to it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from sigma.application.ports import Backend
from sigma.domain.database import STATE_EMPTY
from sigma.domain.errors import InvalidInputError, PermissionDeniedError
from sigma.infrastructure.persistence.connection_tester import ConnectionTester
from sigma.infrastructure.persistence.database_settings import (
    ALLOWED_OPTIONS,
    DatabaseSettings,
    engines_for_ui,
    settings_source,
)

# a callable that rebuilds the running application on the given database and returns whether /setup is needed
Switcher = Callable[[DatabaseSettings], bool]


class Migrator(Protocol):
    """Copies the live data to an empty database in the background, then switches to it (see ``MigrationRunner``)."""

    def start(self, settings: DatabaseSettings) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...


_SOURCE_NOTES = {
    "env": "Definida con la variable de entorno SIGMA_DATABASE_URL: cámbiela allí y reinicie Sigma.",
    "file": "Elegida desde la interfaz (instance/database.json).",
    "default": "Valor por defecto: archivo SQLite dentro de instance/.",
}


class DatabaseAdmin:
    def __init__(
        self,
        instance_dir: Path,
        default_sqlite_path: Path,
        current: DatabaseSettings,
        backend: Backend,
        tester: ConnectionTester | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._instance_dir = Path(instance_dir)
        self._default_sqlite = Path(default_sqlite_path)
        self._current = current
        self._backend = backend
        self._tester = tester or ConnectionTester()
        self._environ = environ
        self.switcher: Switcher | None = None  # set by the web application: it owns the live container
        self.migrator: Migrator | None = None

    # ------------------------------------------------------------------ queries
    def status(self) -> dict[str, Any]:
        source = settings_source(self._instance_dir, self._environ)
        try:
            live = self._backend.describe()
        except Exception:  # noqa: BLE001 - the page must still open when the server is momentarily unreachable
            live = None
        return {
            "current": self._current.describe(),
            "source": source,
            "source_note": _SOURCE_NOTES[source],
            "editable": source != "env",
            "live": live,
            "engines": engines_for_ui(),
            "options": {engine: sorted(keys) for engine, keys in ALLOWED_OPTIONS.items()},
            "default_sqlite_path": str(self._default_sqlite),
        }

    def test(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._tester.test(self._parse(payload)).to_dict()

    # ------------------------------------------------------------------ commands
    def apply(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Point Sigma at another database: it is used as it is (empty → new installation, Sigma's → its data)."""
        if settings_source(self._instance_dir, self._environ) == "env":
            raise PermissionDeniedError(_SOURCE_NOTES["env"])
        if self.switcher is None:
            raise InvalidInputError("Este servidor no permite cambiar de base de datos en caliente. Reinícielo.")
        settings = self._parse(payload)
        if settings.same_target(self._current):
            raise InvalidInputError("Sigma ya está usando esa base de datos.")
        report = self._tester.test(settings)
        if not report.ok:
            raise InvalidInputError(report.message)
        if payload.get("migrate"):
            if self.migrator is None:
                raise InvalidInputError("Este servidor no permite migrar los datos desde la interfaz.")
            if report.state != STATE_EMPTY:
                raise InvalidInputError("Para copiar los datos actuales, la base de destino debe estar vacía.")
            return {"switched": False, "migrating": True, "job": self.migrator.start(settings)}
        setup_required = self.switcher(settings)
        return {"switched": True, "setup_required": setup_required, "report": report.to_dict()}

    def migration_status(self) -> dict[str, Any]:
        return self.migrator.status() if self.migrator else {"state": "idle"}

    # ------------------------------------------------------------------ internals
    def _parse(self, payload: Mapping[str, Any]) -> DatabaseSettings:
        return DatabaseSettings.from_request(payload, self._default_sqlite, self._current)
