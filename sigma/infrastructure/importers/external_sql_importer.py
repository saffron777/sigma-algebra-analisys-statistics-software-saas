"""Read-only import from MySQL/MariaDB, PostgreSQL and Oracle: a SELECT or a whole table, plus a catalog explorer.

Two layers keep it read-only: the SQL guard (``sql_guard``, one statement, SELECT/WITH only, per-engine forbidden words)
and the session itself, opened read-only on the server. Nothing the user types is ever interpolated into the
connection: the form fields go through ``DatabaseSettings.from_request`` (strict characters, allow-listed options).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError, SigmaError
from sigma.infrastructure.database_errors import explain_driver_error
from sigma.infrastructure.importers.common import as_bool, as_int, check_host_allowed
from sigma.infrastructure.importers.sql_fetch import QUERY_TIMEOUT_SECONDS, fetch_rows
from sigma.infrastructure.importers.sql_guard import build_table_query, validate_readonly_query
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available
from sigma.infrastructure.persistence.sql.engine import CONNECT_TIMEOUT_ARG

DEFAULT_MAX_ROWS = 1_000_000
LOGIN_TIMEOUT_SECONDS = 10
CATALOG_LIMIT = 500  # tables listed by the explorer
SUPPORTED = ("mysql", "postgresql", "oracle")

EngineFactory = Callable[[DatabaseSettings], Engine]


# ------------------------------------------------------------------ connection
def _postgres_options() -> str:
    return (
        "-c default_transaction_read_only=on "
        f"-c statement_timeout={QUERY_TIMEOUT_SECONDS * 1000} "
        "-c standard_conforming_strings=on "  # the guard reads quotes assuming this setting
        "-c idle_in_transaction_session_timeout=300000"
    )


def _mysql_session(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
        cursor.execute("SELECT @@SESSION.sql_mode")
        modes = [
            m for m in str(cursor.fetchone()[0]).split(",") if m and m not in ("NO_BACKSLASH_ESCAPES", "ANSI_QUOTES")
        ]
        cursor.execute("SET SESSION sql_mode = %s", (",".join(modes),))  # the guard assumes backslash escapes
        for statement in (  # MySQL / MariaDB each know only one of these; the timeout is best effort
            f"SET SESSION max_execution_time = {QUERY_TIMEOUT_SECONDS * 1000}",
            f"SET SESSION max_statement_time = {QUERY_TIMEOUT_SECONDS}",
        ):
            with contextlib.suppress(Exception):  # unknown variable on the other flavour
                cursor.execute(statement)
    finally:
        cursor.close()


def _oracle_session(dbapi_connection: Any, _record: Any) -> None:
    import oracledb

    wanted = {
        oracledb.DB_TYPE_CLOB: oracledb.DB_TYPE_LONG,
        oracledb.DB_TYPE_NCLOB: oracledb.DB_TYPE_LONG_NVARCHAR,
        oracledb.DB_TYPE_BLOB: oracledb.DB_TYPE_LONG_RAW,
    }
    previous = dbapi_connection.outputtypehandler  # SQLAlchemy's own (number handling): keep it for everything else

    def as_text(cursor: Any, *args: Any) -> Any:
        """Large objects arrive as plain text / bytes instead of LOB handles. Called as (cursor, metadata) by the
        driver and as (cursor, name, default_type, …) by SQLAlchemy's cursor wrapper."""
        by_driver = len(args) == 1  # (cursor, metadata): our raw queries; otherwise SQLAlchemy's own typed queries
        type_code = args[0].type_code if by_driver else args[1]
        if type_code in wanted:
            return cursor.var(wanted[type_code], arraysize=cursor.arraysize)
        return None if by_driver or previous is None else previous(cursor, *args)

    dbapi_connection.outputtypehandler = as_text


def external_engine(settings: DatabaseSettings) -> Engine:
    """A throw-away engine (no pool) whose every session is read-only and time-limited."""
    meta = ENGINES[settings.engine]
    if not driver_available(settings.engine):
        raise InvalidInputError(f"Falta el controlador de {meta['label']}. Instálelo con: pip install {meta['pip']}")
    connect_args: dict[str, Any] = {CONNECT_TIMEOUT_ARG[settings.engine]: LOGIN_TIMEOUT_SECONDS}
    if settings.engine == "postgresql":
        connect_args["options"] = _postgres_options()
    engine = create_engine(settings.sqlalchemy_url(), connect_args=connect_args, poolclass=NullPool)
    if settings.engine == "mysql":
        event.listen(engine, "connect", _mysql_session)
    elif settings.engine == "oracle":
        event.listen(engine, "connect", _oracle_session)
    return engine


def _oracle_read_only(conn: Connection) -> None:
    conn.exec_driver_sql("SET TRANSACTION READ ONLY")  # must be the first statement of the transaction


# ------------------------------------------------------------------ importer
class ExternalSqlImporter:
    """``source_type`` is the engine itself (``mysql``, ``postgresql``, ``oracle``): one pending import per engine."""

    def __init__(
        self,
        engine: str,
        engine_factory: EngineFactory = external_engine,
        allowed_hosts: tuple[str, ...] = (),
    ) -> None:
        if engine not in SUPPORTED:
            raise ValueError(f"unsupported engine {engine!r}")
        self.source_type = engine
        self._label = str(ENGINES[engine]["label"])
        self._engine_factory = engine_factory
        self._allowed_hosts = allowed_hosts

    # ------------------------------------------------------------------ reading
    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        options = request.options
        max_rows = as_int(options.get("max_rows"), "Máximo de filas", DEFAULT_MAX_ROWS, minimum=1)
        query = str(options.get("query") or "").strip()
        table = str(options.get("table") or "").strip()
        if query:
            sql: str | None = validate_readonly_query(query, self.source_type)
        elif table:
            sql = None  # built after the name is resolved against the catalog
        else:
            raise InvalidInputError("Escriba una consulta SQL o indique el nombre de una tabla.")
        settings = self._settings(options)
        engine = self._engine_factory(settings)
        schema = str(options.get("schema") or "").strip() or None
        try:
            if sql is None:
                schema, table = self._resolve(engine, schema, table)
                sql = build_table_query(*self._as_stored(engine, schema, table), max_rows, self.source_type)
            before = _oracle_read_only if self.source_type == "oracle" else None
            frame, truncated = fetch_rows(engine, sql, max_rows, before=before)
        except SigmaError:
            raise
        except Exception as error:  # noqa: BLE001 - each driver has its own exception hierarchy
            raise ImportFailedError(self._describe(error, settings)) from error
        finally:
            engine.dispose()
        detected = {
            "engine": self._label,
            "host": settings.host,
            "database": settings.database,
            "rows_truncated": truncated,
            **({"query": query[:500]} if query else {"table": table, "schema": schema}),
        }
        return frame, detected

    # ------------------------------------------------------------------ explorer
    def inspect(self, request: ImportRequest) -> dict[str, Any]:
        """Tables/views of a schema (and the columns of ``options['table']``) for the «Ver tablas» button."""
        options = request.options
        settings = self._settings(options)
        engine = self._engine_factory(settings)
        try:
            inspector = sa_inspect(engine)
            schema = self._schema(inspector, str(options.get("schema") or "").strip() or None)
            names = [(n, "tabla") for n in inspector.get_table_names(schema=schema)]
            names += [(n, "vista") for n in self._views(inspector, schema)]
            names.sort(key=lambda item: item[0].lower())
            result: dict[str, Any] = {
                "schema": schema,
                "tables": [{"name": n, "kind": k} for n, k in names[:CATALOG_LIMIT]],
                "truncated": len(names) > CATALOG_LIMIT,
            }
            if self.source_type == "postgresql":
                result["schemas"] = [
                    s for s in inspector.get_schema_names() if not s.startswith("pg_") and s != "information_schema"
                ]
            requested = str(options.get("table") or "").strip()
            if requested:
                _, exact = self._resolve(engine, schema, requested, inspector)
                result["columns"] = [
                    {"name": c["name"], "type": str(c["type"])} for c in inspector.get_columns(exact, schema=schema)
                ]
            return result
        except SigmaError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ImportFailedError(self._describe(error, settings)) from error
        finally:
            engine.dispose()

    # ------------------------------------------------------------------ internals
    def _settings(self, options: dict[str, Any]) -> DatabaseSettings:
        extra: dict[str, str] = {}
        if self.source_type == "postgresql" and options.get("sslmode"):
            extra["sslmode"] = str(options["sslmode"])
        if self.source_type == "mysql" and as_bool(options.get("ssl_disabled"), False):
            extra["ssl_disabled"] = "true"
        settings = DatabaseSettings.from_request(
            {
                "engine": self.source_type,
                "host": options.get("host"),
                "port": options.get("port"),
                "database": options.get("database"),
                "username": options.get("username"),
                "password": options.get("password"),
                "options": extra,
            },
            sqlite_path="",
        )
        check_host_allowed(settings.host, self._allowed_hosts)
        return settings

    def _schema(self, inspector: Any, requested: str | None) -> str | None:
        """The schema to read: MySQL has none (the database is it); elsewhere the typed one, matched ignoring case."""
        if self.source_type == "mysql":
            return None
        if not requested:
            return inspector.default_schema_name
        known = inspector.get_schema_names()
        if requested in known:
            return requested
        matches = [name for name in known if name.lower() == requested.lower()]
        if len(matches) == 1:
            return matches[0]
        raise InvalidInputError(f"No existe el esquema «{requested}».")

    @staticmethod
    def _views(inspector: Any, schema: str | None) -> list[str]:
        try:
            return list(inspector.get_view_names(schema=schema))
        except (NotImplementedError, SQLAlchemyError):
            return []

    def _resolve(
        self, engine: Engine, requested_schema: str | None, requested: str, inspector: Any | None = None
    ) -> tuple[str | None, str]:
        """Exact schema and table names for what the user typed (Oracle stores unquoted names in capitals)."""
        inspector = inspector or sa_inspect(engine)
        schema = self._schema(inspector, requested_schema)
        known = [*inspector.get_table_names(schema=schema), *self._views(inspector, schema)]
        if requested in known:
            return schema, requested
        matches = [name for name in known if name.lower() == requested.lower()]
        if len(matches) == 1:
            return schema, matches[0]
        where = f" del esquema «{schema}»" if schema else ""
        raise InvalidInputError(
            f"No existe la tabla o vista «{requested}»{where}. Use «Ver tablas» para elegirla de la lista."
        )

    def _as_stored(self, engine: Engine, schema: str | None, table: str) -> tuple[str, str | None]:
        """Names as the server stores them: SQLAlchemy reports Oracle's capitals in lower case, so undo that."""
        if self.source_type != "oracle":
            return table, schema
        dialect = engine.dialect
        return dialect.denormalize_name(table), dialect.denormalize_name(schema) if schema else None

    def _describe(self, error: Exception, settings: DatabaseSettings) -> str:
        return explain_driver_error(
            error, self.source_type, settings.password, prefix=f"No se pudo consultar {self._label} ({settings.host})"
        )
