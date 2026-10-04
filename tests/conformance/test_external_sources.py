"""Reading from real servers: MySQL/MariaDB, PostgreSQL, Oracle and MongoDB importers (skipped without a URL)."""

import datetime as dt
import os
import time
from decimal import Decimal

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, Numeric, Table, Unicode, UnicodeText, insert

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.external_sql_importer import (
    ExternalSqlImporter,
    _oracle_read_only,
    external_engine,
)
from sigma.infrastructure.importers.mongo_importer import MongoImporter
from sigma.infrastructure.importers.sql_fetch import fetch_rows
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.engine import build_engine
from tests.conformance.conftest import ENV, wipe

SQL_ENGINES = ("mysql", "postgresql", "oracle")
LONG_TEXT = "ñandú " * 1200  # longer than a VARCHAR2 and any single network packet row buffer


def _urls():
    return [
        pytest.param(engine, id=engine, marks=pytest.mark.skipif(not os.environ.get(ENV[engine]), reason=ENV[engine]))
        for engine in SQL_ENGINES
    ]


def connection_options(settings: DatabaseSettings, **extra) -> dict:
    return {
        "host": settings.host,
        "port": settings.port,
        "database": settings.database,
        "username": settings.username,
        "password": settings.password,
        **extra,
    }


@pytest.fixture(params=_urls())
def server(request):
    """A real server with a ``ventas_ext`` table in an otherwise empty database."""
    settings = DatabaseSettings.from_url(os.environ[ENV[request.param]])
    wipe(settings)
    engine = build_engine(settings)
    meta = MetaData()
    table = Table(
        "ventas_ext",
        meta,
        Column("id", Integer, primary_key=True),
        Column("nombre", Unicode(60)),
        Column("monto", Numeric(10, 2)),
        Column("fecha", DateTime),
        Column("nota", UnicodeText),
    )
    meta.create_all(engine)
    rows = [
        {
            "id": i,
            "nombre": "Ñandú café" if i % 2 else "Sur",
            "monto": Decimal(i) / 4 if i % 7 else None,
            "fecha": dt.datetime(2024, 1, 1) + dt.timedelta(days=i),
            "nota": LONG_TEXT if i == 3 else None,
        }
        for i in range(1, 31)
    ]
    with engine.begin() as conn:
        conn.execute(insert(table), rows)
    engine.dispose()
    time.sleep(3)  # Oracle refuses a read-only snapshot of a table created a moment ago (ORA-01466)
    yield request.param, settings
    wipe(settings)


def test_a_table_arrives_with_its_types(server):
    engine_key, settings = server
    frame, detected = ExternalSqlImporter(engine_key).read(
        ImportRequest(engine_key, connection_options(settings, table="ventas_ext"))
    )
    assert len(frame) == 30 and set(frame.columns) >= {"id", "nombre", "monto", "fecha", "nota"}
    frame = frame.sort_values("id").reset_index(drop=True)
    assert frame["id"].tolist() == list(range(1, 31))
    assert frame["monto"].dtype.kind == "f" and frame["monto"].iloc[0] == 0.25 and frame["monto"].isna().sum() == 4
    assert frame["nombre"].iloc[0] == "Ñandú café"
    assert frame["fecha"].iloc[0] == dt.datetime(2024, 1, 2) or str(frame["fecha"].iloc[0]).startswith("2024-01-02")
    assert frame["nota"].iloc[2] == LONG_TEXT  # CLOB / LONGTEXT / TEXT come back as plain text
    assert detected["table"] and detected["rows_truncated"] is False and settings.password not in str(detected)


def test_a_query_and_the_row_cap(server):
    engine_key, settings = server
    reader = ExternalSqlImporter(engine_key)
    frame, detected = reader.read(
        ImportRequest(engine_key, connection_options(settings, query="SELECT id, monto FROM ventas_ext WHERE id <= 12"))
    )
    assert sorted(frame["id"]) == list(range(1, 13))
    capped, detected = reader.read(
        ImportRequest(engine_key, connection_options(settings, table="ventas_ext", max_rows=7))
    )
    assert len(capped) == 7 and detected["rows_truncated"] is True


def test_table_names_are_resolved_whatever_their_case(server):
    engine_key, settings = server
    frame, _ = ExternalSqlImporter(engine_key).read(
        ImportRequest(engine_key, connection_options(settings, table="VENTAS_EXT"))
    )
    assert len(frame) == 30


def test_the_explorer_lists_the_table_and_its_columns(server):
    engine_key, settings = server
    listing = ExternalSqlImporter(engine_key).inspect(ImportRequest(engine_key, connection_options(settings)))
    assert "ventas_ext" in {t["name"].lower() for t in listing["tables"]}
    detail = ExternalSqlImporter(engine_key).inspect(
        ImportRequest(engine_key, connection_options(settings, table="ventas_ext"))
    )
    assert [c["name"].lower() for c in detail["columns"]] == ["id", "nombre", "monto", "fecha", "nota"]


def test_the_session_itself_is_read_only_even_without_the_guard(server):
    """Bypass the guard on purpose: the server must still refuse to write."""
    engine_key, settings = server
    engine = external_engine(settings)
    before = _oracle_read_only if engine_key == "oracle" else None
    try:
        with pytest.raises(Exception) as refused:  # noqa: PT011 - each driver has its own exception
            fetch_rows(engine, "INSERT INTO ventas_ext (id, nombre) VALUES (999, 'x')", 10, before=before)
        assert "read" in str(refused.value).lower() or "solo lectura" in str(refused.value).lower()
        with pytest.raises(Exception):  # noqa: B017, PT011
            fetch_rows(engine, "DELETE FROM ventas_ext", 10, before=before)
    finally:
        engine.dispose()
    rows, _ = fetch_rows(external_engine(settings), "SELECT COUNT(*) AS n FROM ventas_ext", 10, before=before)
    assert int(rows["n"].iloc[0]) == 30


def test_a_wrong_password_is_explained_without_revealing_it(server):
    engine_key, settings = server
    with pytest.raises(ImportFailedError) as refused:
        ExternalSqlImporter(engine_key).read(
            ImportRequest(engine_key, connection_options(settings, password="clave-mala-123", table="ventas_ext"))
        )
    assert "contraseña" in str(refused.value) and "clave-mala-123" not in str(refused.value)


def test_a_missing_table_is_explained(server):
    engine_key, settings = server
    with pytest.raises(InvalidInputError, match="No existe la tabla"):
        ExternalSqlImporter(engine_key).read(
            ImportRequest(engine_key, connection_options(settings, table="no_existe_xyz"))
        )


# ------------------------------------------------------------------ MongoDB
@pytest.mark.skipif(not os.environ.get(ENV["mongodb"]), reason=ENV["mongodb"])
def test_mongodb_collections_are_read_filtered_and_explored():
    from pymongo import MongoClient

    settings = DatabaseSettings.from_url(os.environ[ENV["mongodb"]])
    wipe(settings)
    client = MongoClient(settings.to_url(), serverSelectionTimeoutMS=5000)
    try:
        client[settings.database]["clientes"].insert_many(
            [
                {"nombre": "Ana", "edad": 31, "dir": {"ciudad": "Lima"}, "alta": dt.datetime(2024, 3, 1)},
                {"nombre": "Luis", "edad": 25, "dir": {"ciudad": "Quito"}},
                {"nombre": "Marta", "edad": 47},
            ]
        )
        options = {
            "host": settings.host,
            "port": settings.port,
            "database": settings.database,
            "collection": "clientes",
        }
        reader = MongoImporter()
        frame, detected = reader.read(ImportRequest("mongodb", options))
        assert len(frame) == 3 and "dir.ciudad" in frame.columns and detected["rows_truncated"] is False
        older = reader.read(ImportRequest("mongodb", {**options, "filter": '{"edad": {"$gt": 30}}'}))[0]
        assert sorted(older["nombre"]) == ["Ana", "Marta"]
        with pytest.raises(InvalidInputError):
            reader.read(ImportRequest("mongodb", {**options, "filter": '{"$where": "1==1"}'}))
        listing = reader.inspect(ImportRequest("mongodb", options))
        assert [t["name"] for t in listing["tables"]] == ["clientes"]
        assert "nombre" in {c["name"] for c in listing["columns"]}
    finally:
        client.close()
        wipe(settings)
