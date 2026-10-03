"""Running a database migration from the web application: one background job, a maintenance gate, live progress."""

from __future__ import annotations

import threading
import time
from typing import Any

from flask import Flask, request

from sigma.application.use_cases.migration import MigrateData
from sigma.domain.database import MIGRATION_ENTITIES
from sigma.domain.errors import InvalidInputError, MaintenanceError, SigmaError
from sigma.infrastructure.persistence.backend import create_backend
from sigma.infrastructure.persistence.connection_tester import explain_error
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings
from sigma.presentation.web.switching import ContainerSwitcher

MAINTENANCE_KEY = "SIGMA_MAINTENANCE"
# what keeps working while the data is being copied: assets, the health check and the progress of the migration itself
_OPEN_PATHS = (
    "/static/",
    "/favicon.ico",
    "/health",
    "/api/admin/database/migration",
    "/api/auth/me",
    "/api/auth/status",
)
_GRACE_SECONDS = 1.0  # lets requests that started before the gate closed finish


def init_maintenance(app: Flask) -> None:
    """Rejects every request (except the few above) while a migration holds the data."""

    @app.before_request
    def maintenance_gate() -> None:
        if app.config.get(MAINTENANCE_KEY) and not request.path.startswith(_OPEN_PATHS):
            raise MaintenanceError("Sigma está migrando la base de datos. Vuelva a intentarlo en unos minutos.")


def _idle() -> dict[str, Any]:
    return {"state": "idle"}


class MigrationRunner:
    """Copies the live database into an empty one and then switches to it; at most one migration at a time."""

    def __init__(self, app: Flask, switcher: ContainerSwitcher) -> None:
        self._app = app
        self._switcher = switcher
        self._lock = threading.Lock()
        self._job: dict[str, Any] = _idle()

    # ------------------------------------------------------------------ queries
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._job, "steps": [dict(step) for step in self._job.get("steps", [])]}

    # ------------------------------------------------------------------ commands
    def start(self, settings: DatabaseSettings) -> dict[str, Any]:
        with self._lock:
            if self._job.get("state") == "running":
                raise InvalidInputError("Ya hay una migración en curso.")
            self._job = {
                "state": "running",
                "target": ENGINES[settings.engine]["label"],
                "started_at": time.time(),
                "steps": [
                    {"key": key, "label": label, "done": 0, "total": 0, "state": "waiting"}
                    for key, label in MIGRATION_ENTITIES
                ],
            }
        self._app.config[MAINTENANCE_KEY] = True
        threading.Thread(target=self._run, args=(settings,), name="sigma-migration", daemon=True).start()
        return self.status()

    # ------------------------------------------------------------------ the job
    def _run(self, settings: DatabaseSettings) -> None:
        target = None
        try:
            time.sleep(_GRACE_SECONDS)
            source = self._app.extensions["sigma"]
            source.kernels.shutdown_all()  # consoles write to the database on their own
            target = create_backend(settings)  # creates the tables of the new database
            report = MigrateData().execute(
                source.backend.transfer(source.config.notebooks_dir),
                target.transfer(source.config.notebooks_dir),
                self._progress,
            )
            target.close()
            target = None
            setup_required = self._switcher(settings)
            self._finish("done", report=report.to_dict(), setup_required=setup_required)
        except SigmaError as error:
            self._finish("failed", message=error.message)
        except Exception as error:  # noqa: BLE001 - drivers raise their own hierarchies
            self._finish("failed", message=explain_error(error, settings))
        finally:
            if target is not None:
                target.close()
            self._app.config[MAINTENANCE_KEY] = False

    def _progress(self, key: str, label: str, done: int, total: int) -> None:
        with self._lock:
            for step in self._job["steps"]:
                if step["key"] == key:
                    step.update(done=done, total=total, state="done" if done >= total else "running")

    def _finish(self, state: str, **fields: Any) -> None:
        with self._lock:
            self._job.update(state=state, finished_at=time.time(), **fields)
