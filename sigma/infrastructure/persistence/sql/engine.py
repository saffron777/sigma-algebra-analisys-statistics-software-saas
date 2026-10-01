"""SQLAlchemy engines, the ``Database`` handle used by every SQL repository, and schema management."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, insert, inspect, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.legacy_sqlite import upgrade_legacy_sqlite

# future schema changes: version -> function that upgrades from version-1 (portable DDL through SQLAlchemy)
_STEPS: dict[int, Callable[[Engine], None]] = {}


def build_engine(settings: DatabaseSettings) -> Engine:
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
        options: dict = {"pool_pre_ping": True, "pool_recycle": 1800}
        if settings.engine == "mssql":
            options["fast_executemany"] = True  # bulk inserts of dataset rows are ~100× faster
        engine = create_engine(url, **options)
    if settings.schema and settings.engine in ("mssql", "postgresql", "oracle"):
        engine = engine.execution_options(schema_translate_map={None: settings.schema})
    return engine


class Database:
    """A connection source plus the schema it carries. ``Database(path)`` is a SQLite file (handy in tests)."""

    def __init__(self, target: DatabaseSettings | str | Path, *, ensure: bool = True) -> None:
        self.settings = target if isinstance(target, DatabaseSettings) else DatabaseSettings.sqlite(target)
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
    """Make the database current: create it (empty), upgrade it (older Sigma), or refuse one that is not Sigma's."""
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
            raise InvalidInputError("La base de datos fue creada por una versión más reciente de Sigma.")
        for target in range(version + 1, schema.SCHEMA_VERSION + 1):
            _STEPS[target](engine)
            _stamp(engine, target)
        schema.metadata.create_all(engine)  # tables added since (checkfirst: only the missing ones)
        return
    if engine.dialect.name == "sqlite" and {"datasets", "saved_results"} <= tables:
        _upgrade_sqlite_file(engine)
        return
    raise InvalidInputError(
        "La base de datos elegida ya contiene tablas que no son de Sigma. "
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
