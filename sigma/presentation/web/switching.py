"""Hot switch of the database: build a new container on the chosen database and swap it in for the running one."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import Any

from flask import Flask

from sigma.bootstrap import Container, build_container
from sigma.domain.errors import InvalidInputError, SigmaError
from sigma.infrastructure.persistence.connection_tester import explain_error
from sigma.infrastructure.persistence.database_settings import DatabaseSettings, save_settings


class ContainerSwitcher:
    """Callable handed to ``DatabaseAdmin``: returns whether the new database needs the first-run setup.

    The old container keeps serving requests that already started; it is closed shortly after the swap. If the new
    database cannot be opened the application stays exactly as it was and the reason is reported.
    """

    def __init__(
        self,
        app: Flask,
        close_delay: float = 2.0,
        build: Callable[[object], Container] = build_container,
    ) -> None:
        self._app = app
        self._close_delay = close_delay
        self._build = build
        self._lock = threading.Lock()
        self.migrator: Any = None  # the app's MigrationRunner, handed to every container that takes over

    def __call__(self, settings: DatabaseSettings) -> bool:
        with self._lock:
            old: Container = self._app.extensions["sigma"]
            config = dataclasses.replace(old.config, database=settings)
            try:
                new = self._build(config)
            except SigmaError:
                raise
            except Exception as error:  # noqa: BLE001 - drivers raise their own hierarchies while creating the schema
                raise InvalidInputError(explain_error(error, settings)) from error
            try:
                save_settings(config.instance_dir, settings)
            except OSError as error:
                new.close()
                raise InvalidInputError(f"No se pudo guardar la configuración: {error.strerror or error}") from error
            new.database_admin.switcher = self
            new.database_admin.migrator = self.migrator
            self._app.extensions["sigma"] = new
            self._retire(old)
            return new.get_setup_state.setup_required()

    def _retire(self, old: Container) -> None:
        if self._close_delay <= 0:
            old.close()
            return
        timer = threading.Timer(self._close_delay, old.close)
        timer.daemon = True
        timer.start()
