"""Backend conformance: the same tests run on SQLite (always) and on every engine whose test URL is defined.

    SIGMA_TEST_MSSQL_URL      mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server
    SIGMA_TEST_MYSQL_URL      mysql+pymysql://user:pass@host/db           (MariaDB works too)
    SIGMA_TEST_POSTGRES_URL   postgresql+psycopg://user:pass@host/db
    SIGMA_TEST_ORACLE_URL     oracle+oracledb://user:pass@host:1521/?service_name=FREEPDB1
    SIGMA_TEST_MONGO_URL      mongodb://host:27017/sigma_test

The database behind each URL must be empty and disposable: every test starts by dropping all of its tables.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import MetaData

from sigma.infrastructure.persistence.backend import create_backend
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.engine import build_engine

ENV = {
    "mssql": "SIGMA_TEST_MSSQL_URL",
    "mysql": "SIGMA_TEST_MYSQL_URL",
    "postgresql": "SIGMA_TEST_POSTGRES_URL",
    "oracle": "SIGMA_TEST_ORACLE_URL",
    "mongodb": "SIGMA_TEST_MONGO_URL",
}


def _params():
    params = [pytest.param("sqlite", id="sqlite")]
    try:
        import mongomock  # noqa: F401
    except ImportError:
        params.append(
            pytest.param("mongomock", id="mongomock", marks=pytest.mark.skip(reason="mongomock no instalado"))
        )
    else:
        params.append(pytest.param("mongomock", id="mongomock"))  # MongoDB's API in memory: always available
    for engine, variable in ENV.items():
        skip = pytest.mark.skipif(not os.environ.get(variable), reason=f"{variable} no está definido")
        params.append(pytest.param(engine, id=engine, marks=skip))
    return params


def wipe(settings: DatabaseSettings) -> None:
    """Empty the target database (tests only)."""
    if settings.engine == "mongodb":
        from pymongo import MongoClient

        client = MongoClient(settings.to_url(), serverSelectionTimeoutMS=5000)
        client.drop_database(client.get_default_database().name)
        client.close()
        return
    engine = build_engine(settings)
    try:
        meta = MetaData()
        meta.reflect(engine, schema=settings.schema or None)
        meta.drop_all(engine)
    finally:
        engine.dispose()


@pytest.fixture(params=_params())
def backend(request, tmp_path):
    engine = request.param
    if engine == "mongomock":
        import mongomock

        from sigma.infrastructure.persistence.mongo.backend import MongoBackend
        from sigma.infrastructure.persistence.mongo.core import MongoCore

        settings = DatabaseSettings(engine="mongodb", host="mock", database="sigma_test")
        instance = MongoBackend(MongoCore(mongomock.MongoClient(), "sigma_test", settings))
        yield instance
        instance.close()
        return
    if engine == "sqlite":
        settings = DatabaseSettings.sqlite(tmp_path / "conformance.db")
    else:
        settings = DatabaseSettings.from_url(os.environ[ENV[engine]])
        wipe(settings)
    instance = create_backend(settings)
    yield instance
    instance.close()
    if engine != "sqlite":
        wipe(settings)


@pytest.fixture
def real_server(backend):
    """For tests that connect to the database by its settings: not possible for the in-memory MongoDB."""
    if backend.settings.host == "mock":
        pytest.skip("MongoDB en memoria: no hay servidor al que conectar")


@pytest.fixture
def uid(backend):
    return backend.users().add("tester", "Tester", "x", "user", True, True).id


@pytest.fixture
def two_users(backend):
    users = backend.users()
    return users.add("ana", "Ana", "x", "user", False, False).id, users.add(
        "luis", "Luis", "x", "user", False, False
    ).id
