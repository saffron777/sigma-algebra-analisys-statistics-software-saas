"""SQLAlchemy engines, the ``Database`` handle used by every SQL repository, and schema management."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Column, Table, create_engine, event, insert, inspect, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateColumn

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.legacy_sqlite import upgrade_legacy_sqlite


# future schema changes: version -> function that upgrades from version-1 (portable DDL through SQLAlchemy)
def _add_ai_provider(engine: Engine) -> None:
    """v8: the usage log also records which service (anthropic, openai, ollama…) answered."""
    _add_column(engine, schema.ai_usage, schema.ai_usage.c.provider)


def _add_projects(engine: Engine) -> None:
    """v9: the survey projects table (a brand-new table, created by ``metadata.create_all``)."""
    return None


def _fix_projects(engine: Engine) -> None:
    """v10: the projects table changed shape while still unreleased (responses moved to a referenced dataset)."""
    schema.projects.drop(engine, checkfirst=True)


def _widen_app_settings(engine: Engine) -> None:
    """v11: ``app_settings.value`` now holds the whole AI configuration (JSON), so it must accept long text."""
    dialect = engine.dialect.name
    table = engine.dialect.identifier_preparer.quote(schema.app_settings.name)
    if dialect == "sqlite":
        return  # TEXT has no length limit in SQLite
    if dialect in ("mysql", "mariadb"):
        statement = f"ALTER TABLE {table} MODIFY value LONGTEXT"
    elif dialect == "postgresql":
        statement = f"ALTER TABLE {table} ALTER COLUMN value TYPE TEXT"
    elif dialect == "oracle":
        statement = f"ALTER TABLE {table} MODIFY (value CLOB)"
    else:  # mssql
        statement = f"ALTER TABLE {table} ALTER COLUMN value NVARCHAR(MAX)"
    with engine.begin() as conn:
        conn.execute(text(statement))


def _add_column(engine: Engine, table: Table, column: Column) -> None:
    """Add a missing column to an existing table, portably and **idempotently**.

    SQL Server and Oracle say ``ADD`` where SQLite, MySQL and PostgreSQL say ``ADD COLUMN``; the identifier is
    quoted so the lower-case names survive on engines that upper-case unquoted ones. The default is written inline
    (SQL Server and Oracle cannot add a column that is both ``NOT NULL`` and defaultless), and a step that already
    ran is skipped: the version stamp may have been interrupted, so the check must not raise.
    """
    columns = inspect(engine).get_columns(table.name, schema=table.schema)
    if any(item["name"].lower() == column.name.lower() for item in columns):
        return
    ddl = CreateColumn(column).compile(dialect=engine.dialect)
    if engine.dialect.name not in ("mssql", "oracle") and column.default is not None and "DEFAULT" not in str(ddl):
        evaluated = column.default.arg
        if isinstance(evaluated, str):  # SQLAlchemy omits a str default from the DDL: it is a client-side one
            ddl = f"{ddl} DEFAULT '{evaluated}'"
    keyword = "ADD" if engine.dialect.name in ("mssql", "oracle") else "ADD COLUMN"
    statement = f"ALTER TABLE {engine.dialect.identifier_preparer.quote(table.name)} {keyword} {ddl}"
    with engine.begin() as conn:
        conn.execute(text(statement))


_STEPS: dict[int, Callable[[Engine], None]] = {8: _add_ai_provider, 9: _add_projects, 10: _fix_projects, 11: _widen_app_settings}


CONNECT_TIMEOUT_ARG = {
    "mssql": "timeout",
    "mysql": "connect_timeout",
    "postgresql": "connect_timeout",
    "oracle": "tcp_connect_timeout",
}


def build_engine(settings: DatabaseSettings, connect_timeout: int = 15) -> Engine:
    """An engine for a relational setting; a missing driver or an unsupported engine is explained in Spanish."""
    meta = ENGINES[settings.engine]
    if not settings.is_relational:
        raise InvalidInputError(f"{meta['label']} no es una base relacional.")
    if not driver_available(settings.engine):
        raise InvalidInputError(f"Falta el controlador de {meta['label']}. Instálelo con: pip install {meta['pip']}")
    url = settings.sqlalchemy_url()
    if settings.is_sqlite:
        Path(settings.path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"timeout": 30})

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.close()

    else:
        options: dict = {
            "pool_pre_ping": True,
            "pool_recycle": 1800,
            "connect_args": {CONNECT_TIMEOUT_ARG[settings.engine]: connect_timeout},  # an unreachable host fails fast
        }
        if settings.engine == "mssql":
            options["fast_executemany"] = True  # bulk inserts of dataset rows are ~100× faster
        engine = create_engine(url, **options)
    if settings.schema and settings.engine in ("mssql", "postgresql", "oracle"):
        engine = engine.execution_options(schema_translate_map={None: settings.schema})
    return engine


def ensure_database(settings: DatabaseSettings, connect_timeout: int = 15) -> None:
    """Create the target database when it does not exist yet.

    SQLite creates its file implicitly and MongoDB its collections, so only the
    engines with an explicit ``CREATE DATABASE`` step need this. Oracle is left
    out on purpose: its ``database`` is a service/PDB that a regular connection
    cannot create.
    """
    if settings.engine == "mssql":
        _ensure_mssql_database(settings, connect_timeout)
    elif settings.engine == "mysql":
        _ensure_mysql_database(settings, connect_timeout)
    elif settings.engine == "postgresql":
        _ensure_postgresql_database(settings, connect_timeout)


def _safe_identifier(name: str) -> str:
    """Remove the quoting characters every engine could otherwise interpret, so the name stays one identifier."""
    return name.replace("[", "").replace("]", "").replace('"', "").replace("'", "").replace("`", "")


def _ensure_mssql_database(settings: DatabaseSettings, connect_timeout: int) -> None:
    """Create the SQL Server database when it is missing, connecting to ``master`` first."""
    safe = _safe_identifier(settings.database)
    if not safe:
        return
    admin = dataclasses.replace(settings, database="master", schema="")
    engine = build_engine(admin, connect_timeout=connect_timeout)
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"IF DB_ID(N'{safe}') IS NULL CREATE DATABASE [{safe}]"))
    finally:
        engine.dispose()


def _ensure_mysql_database(settings: DatabaseSettings, connect_timeout: int) -> None:
    """Create the MySQL/MariaDB database when it is missing, connecting without a default schema."""
    safe = _safe_identifier(settings.database)
    if not safe:
        return
    admin = dataclasses.replace(settings, database=None, schema="")
    engine = build_engine(admin, connect_timeout=connect_timeout)
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{safe}` CHARACTER SET utf8mb4"))
    finally:
        engine.dispose()


def _ensure_postgresql_database(settings: DatabaseSettings, connect_timeout: int) -> None:
    """Create the PostgreSQL database when it is missing, connecting to ``postgres`` first."""
    safe = _safe_identifier(settings.database)
    if not safe:
        return
    admin = dataclasses.replace(settings, database="postgres", schema="")
    engine = build_engine(admin, connect_timeout=connect_timeout)
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": safe}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{safe}"'))
    finally:
        engine.dispose()


class Database:
    """A connection source plus the schema it carries. ``Database(path)`` is a SQLite file (handy in tests)."""

    def __init__(self, target: DatabaseSettings | str | Path, *, ensure: bool = True) -> None:
        self.settings = target if isinstance(target, DatabaseSettings) else DatabaseSettings.sqlite(target)
        if ensure:
            ensure_database(self.settings)
        self.engine = build_engine(self.settings)
        self.dialect: str = self.engine.dialect.name
        if ensure:
            ensure_schema(self.engine, self.settings)

    @contextmanager
    def begin(self) -> Iterator[Connection]:
        """One transaction: committed when the block ends, rolled back if it raises."""
        with self.engine.begin() as conn:
            yield conn

    def dispose(self) -> None:
        self.engine.dispose()

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == "sqlite"


# ------------------------------------------------------------------ schema management
def _read_version(engine: Engine) -> int:
    with engine.connect() as conn:
        row = conn.execute(select(schema.schema_version.c.version).where(schema.schema_version.c.id == 1)).first()
    return int(row[0]) if row else 0


def _stamp(engine: Engine, version: int) -> None:
    with engine.begin() as conn:
        if (
            conn.execute(
                update(schema.schema_version).where(schema.schema_version.c.id == 1).values(version=version)
            ).rowcount
            == 0
        ):
            conn.execute(insert(schema.schema_version).values(id=1, version=version))


def _create_schema_namespace(engine: Engine, settings: DatabaseSettings) -> None:
    """``CREATE SCHEMA`` for the optional dedicated schema (SQL Server / PostgreSQL), when it does not exist yet."""
    name = settings.schema
    if not name or settings.engine not in ("mssql", "postgresql"):
        return
    safe = name.replace('"', "").replace("]", "").replace("[", "")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if settings.engine == "postgresql":
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{safe}"'))
            else:
                conn.execute(text(f"IF SCHEMA_ID(N'{safe}') IS NULL EXEC('CREATE SCHEMA [{safe}]')"))
    except SQLAlchemyError as error:
        raise InvalidInputError(f"No se pudo crear el esquema «{name}»: {_short(error)}") from None


def _short(error: Exception) -> str:
    return str(getattr(error, "orig", error)).splitlines()[0][:300]


def ensure_schema(engine: Engine, settings: DatabaseSettings | None = None) -> None:
    """Make the database current: create it (empty), upgrade it (older Sigma Algebra), or refuse one that is not Sigma Algebra's."""
    if settings is not None:
        _create_schema_namespace(engine, settings)
    schema_name = settings.schema if settings and settings.engine in ("mssql", "postgresql", "oracle") else None
    tables = set(inspect(engine).get_table_names(schema=schema_name or None))
    if not tables:
        schema.metadata.create_all(engine)
        _stamp(engine, schema.SCHEMA_VERSION)
        return
    if "sigma_schema" in tables:
        version = _read_version(engine)
        if version > schema.SCHEMA_VERSION:
            raise InvalidInputError("La base de datos fue creada por una versión más reciente de Sigma Algebra.")
        for target in range(version + 1, schema.SCHEMA_VERSION + 1):
            _STEPS[target](engine)
            _stamp(engine, target)
        schema.metadata.create_all(engine)  # tables added since (checkfirst: only the missing ones)
        return
    if engine.dialect.name == "sqlite" and {"datasets", "saved_results"} <= tables:
        _upgrade_sqlite_file(engine)
        return
    raise InvalidInputError(
        "La base de datos elegida ya contiene tablas que no son de Sigma Algebra. "
        "Use una base vacía o indique un esquema propio."
    )


def _upgrade_sqlite_file(engine: Engine) -> None:
    raw = engine.raw_connection()
    try:
        connection = raw.driver_connection
        assert isinstance(connection, sqlite3.Connection)
        upgrade_legacy_sqlite(connection)
    finally:
        raw.close()
    schema.metadata.create_all(engine)  # sigma_schema, notebooks… (the ones an old file does not have)
    _stamp(engine, schema.SCHEMA_VERSION)
