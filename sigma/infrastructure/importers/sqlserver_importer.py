"""SQL Server importer (read-only) built on SQLAlchemy + pyodbc."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.common import as_bool, as_int, check_host_allowed
from sigma.infrastructure.importers.sql_fetch import fetch_rows
from sigma.infrastructure.importers.sql_guard import build_table_query, validate_readonly_query

DEFAULT_MAX_ROWS = 1_000_000
LOGIN_TIMEOUT_SECONDS = 10
QUERY_TIMEOUT_SECONDS = 120
PREFERRED_DRIVERS = (
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "SQL Server Native Client 11.0",
    "SQL Server",
)

EngineFactory = Callable[[dict[str, Any]], Engine]


def _odbc_value(value: str) -> str:
    """Brace-quote an ODBC connection-string value so ``;`` / ``}`` cannot inject attributes."""
    return "{" + value.replace("}", "}}") + "}"


def installed_driver(requested: str | None) -> str:
    import pyodbc

    available = pyodbc.drivers()
    if requested:
        if requested not in available:
            raise InvalidInputError(f"El controlador ODBC «{requested}» no está instalado.")
        return requested
    for name in PREFERRED_DRIVERS:
        if name in available:
            return name
    raise ImportFailedError(
        "No hay un controlador ODBC de SQL Server instalado (se recomienda «ODBC Driver 18 for SQL Server»)."
    )


def sqlserver_engine(options: dict[str, Any]) -> Engine:
    server = str(options.get("server") or "").strip()
    database = str(options.get("database") or "").strip()
    if not server:
        raise InvalidInputError("Indique el servidor de SQL Server.")
    if not database:
        raise InvalidInputError("Indique la base de datos.")
    driver = installed_driver(options.get("driver") or None)
    parts = [
        f"DRIVER={_odbc_value(driver)}",
        f"SERVER={_odbc_value(server)}",
        f"DATABASE={_odbc_value(database)}",
        f"Connection Timeout={LOGIN_TIMEOUT_SECONDS}",
        "ApplicationIntent=ReadOnly",
    ]
    if str(options.get("auth") or "windows") == "sql":
        username, password = str(options.get("username") or ""), str(options.get("password") or "")
        if not username:
            raise InvalidInputError("Indique el usuario de SQL Server.")
        parts += [f"UID={_odbc_value(username)}", f"PWD={_odbc_value(password)}"]
    else:
        parts.append("Trusted_Connection=yes")
    if "18" in driver:
        parts.append("Encrypt=yes")
    if as_bool(options.get("trust_server_certificate"), False):
        parts.append("TrustServerCertificate=yes")
    return create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": ";".join(parts)}))


class SqlServerImporter:
    """Runs a read-only SELECT (or reads a table) and returns the rows as a DataFrame.

    ``engine_factory`` is injectable so the same code path can be exercised against SQLite in tests.
    """

    source_type = "sqlserver"

    def __init__(self, engine_factory: EngineFactory = sqlserver_engine, allowed_hosts: tuple[str, ...] = ()) -> None:
        self._engine_factory = engine_factory
        self._allowed_hosts = allowed_hosts

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        options = request.options
        max_rows = as_int(options.get("max_rows"), "Máximo de filas", DEFAULT_MAX_ROWS, minimum=1)
        query = str(options.get("query") or "").strip()
        table = str(options.get("table") or "").strip()
        if query:
            sql = validate_readonly_query(query)
        elif table:
            sql = build_table_query(table, str(options.get("schema") or "").strip() or None, max_rows)
        else:
            raise InvalidInputError("Escriba una consulta SQL o indique el nombre de una tabla.")

        check_host_allowed(str(options.get("server") or ""), self._allowed_hosts)
        engine = self._engine_factory(options)
        try:
            frame, truncated = fetch_rows(engine, sql, max_rows, timeout_seconds=QUERY_TIMEOUT_SECONDS)
        except SQLAlchemyError as error:
            raise ImportFailedError(self._describe(error, options)) from error
        finally:
            engine.dispose()
        detected = {
            "server": options.get("server"),
            "database": options.get("database"),
            "auth": options.get("auth") or "windows",
            "rows_truncated": truncated,
            **({"query": query[:500]} if query else {"table": table, "schema": options.get("schema") or None}),
        }
        return frame, detected

    @staticmethod
    def _describe(error: SQLAlchemyError, options: dict[str, Any]) -> str:
        detail = str(getattr(error, "orig", error)).replace("\n", " ")[:400]
        hint = ""
        if "certificate" in detail.lower() or "ssl" in detail.lower():
            hint = " Si el servidor usa un certificado autofirmado, marque «Confiar en el certificado del servidor»."
        return f"No se pudo consultar SQL Server ({options.get('server')}): {detail}.{hint}"
