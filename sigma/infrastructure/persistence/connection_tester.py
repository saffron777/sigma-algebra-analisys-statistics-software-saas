"""Tries a database setting before Sigma is pointed at it, and explains any failure in plain Spanish."""

from __future__ import annotations

import dataclasses
import uuid

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from sigma.domain.database import (
    STATE_EMPTY,
    STATE_FOREIGN,
    STATE_SIGMA,
    CheckStep,
    ConnectionReport,
)
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.database_errors import explain_driver_error
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available
from sigma.infrastructure.persistence.sql.engine import build_engine

_VERSION_SQL = {
    "sqlite": "SELECT sqlite_version()",
    "mssql": "SELECT @@VERSION",
    "mysql": "SELECT VERSION()",
    "postgresql": "SELECT version()",
    "oracle": "SELECT banner FROM v$version FETCH FIRST 1 ROWS ONLY",
}
_PING_SQL = {"oracle": "SELECT 1 FROM DUAL"}


def explain_error(error: BaseException, settings: DatabaseSettings) -> str:
    """A Spanish sentence for a driver failure; the password never appears in it."""
    return explain_driver_error(error, settings.engine, settings.password)


class ConnectionTester:
    """Connects, reads the server version, inspects what is already there and checks that tables can be created."""

    def __init__(self, timeout_seconds: int = 8) -> None:
        self._timeout = timeout_seconds

    def test(self, settings: DatabaseSettings) -> ConnectionReport:
        settings.validate()
        meta = ENGINES[settings.engine]
        if not settings.is_relational:
            return self._test_document_database(settings)
        if not driver_available(settings.engine):
            message = f"Falta el controlador de {meta['label']}. Instálelo con: pip install {meta['pip']}"
            return ConnectionReport(False, settings.engine, message, steps=(CheckStep("Controlador", False, message),))
        steps: list[CheckStep] = [CheckStep("Controlador", True, str(meta["package"] or "incluido en Python"))]
        engine = build_engine(dataclasses.replace(settings, schema=""), connect_timeout=self._timeout)
        try:
            with engine.connect() as conn:
                conn.execute(text(_PING_SQL.get(settings.engine, "SELECT 1")))
                version = self._version(conn, settings)
                steps.append(CheckStep("Conexión", True, version))
                tables = self._tables(conn, settings)
                state = self._classify(tables)
                steps.append(CheckStep("Contenido", state != STATE_FOREIGN, self._describe_state(state, tables)))
                writable = self._probe_write(conn, settings, steps)
        except InvalidInputError:
            raise
        except Exception as error:  # noqa: BLE001 - every driver has its own exception hierarchy
            explanation = explain_error(error, settings)
            steps.append(CheckStep("Conexión", False, explanation))
            return ConnectionReport(False, settings.engine, explanation, steps=tuple(steps))
        finally:
            engine.dispose()
        counts = self._counts(settings) if state == STATE_SIGMA else None
        ok = writable and state != STATE_FOREIGN
        return ConnectionReport(
            ok, settings.engine, self._headline(ok, state, writable), version, state, counts, tuple(steps)
        )

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _version(conn: Connection, settings: DatabaseSettings) -> str:
        sql = _VERSION_SQL.get(settings.engine)
        value = conn.execute(text(sql)).scalar() if sql else ""
        return str(value).strip().splitlines()[0][:120] if value else ""

    @staticmethod
    def _tables(conn: Connection, settings: DatabaseSettings) -> list[str]:
        return sorted(inspect(conn).get_table_names(schema=settings.schema or None))

    @staticmethod
    def _classify(tables: list[str]) -> str:
        if not tables:
            return STATE_EMPTY
        return (
            STATE_SIGMA if "sigma_schema" in tables or {"datasets", "saved_results"} <= set(tables) else STATE_FOREIGN
        )

    @staticmethod
    def _describe_state(state: str, tables: list[str]) -> str:
        if state == STATE_EMPTY:
            return "Base vacía: Sigma creará sus tablas."
        if state == STATE_SIGMA:
            return "Ya contiene datos de Sigma: se usarán tal como están."
        return f"Contiene {len(tables)} tablas que no son de Sigma."

    def _probe_write(self, conn: Connection, settings: DatabaseSettings, steps: list[CheckStep]) -> bool:
        """Creates, fills and drops a throw-away table: what Sigma needs to be able to do."""
        if settings.schema and not inspect(conn).has_schema(settings.schema):
            steps.append(CheckStep("Permisos", True, "El esquema se creará al aplicar."))
            return True
        name = f"sigma_probe_{uuid.uuid4().hex[:10]}"
        qualified = f"{settings.schema}.{name}" if settings.schema else name
        try:
            conn.execute(text(f"CREATE TABLE {qualified} (id INTEGER)"))
            conn.execute(text(f"INSERT INTO {qualified} (id) VALUES (1)"))
            conn.execute(text(f"DROP TABLE {qualified}"))
            conn.commit()
        except Exception as error:  # noqa: BLE001
            conn.rollback()
            detail = f"El usuario no puede crear tablas: {explain_error(error, settings)}"
            steps.append(CheckStep("Permisos", False, detail))
            return False
        steps.append(CheckStep("Permisos", True, "Puede crear, escribir y borrar tablas."))
        return True

    @staticmethod
    def _counts(settings: DatabaseSettings) -> dict[str, int] | None:
        from sigma.infrastructure.persistence.sql.backend import SqlBackend
        from sigma.infrastructure.persistence.sql.engine import Database

        try:
            database = Database(settings, ensure=False)
            try:
                return SqlBackend(database).describe()["counts"]
            finally:
                database.dispose()
        except Exception:  # noqa: BLE001 - counts are a courtesy; an older layout simply shows none
            return None

    @staticmethod
    def _headline(ok: bool, state: str, writable: bool) -> str:
        if state == STATE_FOREIGN:
            return (
                "La base de datos ya tiene tablas que no son de Sigma. Use una base vacía o indique un esquema propio."
            )
        if not writable:
            return "Se pudo conectar, pero el usuario no puede crear tablas."
        if state == STATE_SIGMA:
            return "Conexión correcta. La base ya contiene datos de Sigma."
        return "Conexión correcta. La base está vacía y lista para usarse."

    def _test_document_database(self, settings: DatabaseSettings) -> ConnectionReport:
        from sigma.infrastructure.persistence.mongo.tester import test_mongo

        return test_mongo(settings, self._timeout)
