"""Reading from MySQL / PostgreSQL / Oracle / MongoDB: the same code paths run here against SQLite and mongomock."""

import datetime as dt

import pytest
from sqlalchemy import create_engine, text

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.external_sql_importer import ExternalSqlImporter
from sigma.infrastructure.importers.mongo_importer import MongoImporter, parse_filter
from sigma.infrastructure.importers.registry import DefaultImporterRegistry

PASSWORD = "S3creta-NO-FILTRAR"


def request(source_type, **options):
    return ImportRequest(source_type, {"host": "db.interna", "database": "datos", "password": PASSWORD, **options})


@pytest.fixture
def sqlite_url(tmp_path):
    path = tmp_path / "externa.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE Ventas (id INTEGER, region TEXT, monto REAL, fecha TEXT)"))
        conn.execute(text("CREATE VIEW vista_norte AS SELECT * FROM Ventas WHERE region = 'Norte'"))
        for i in range(1, 26):
            conn.execute(
                text("INSERT INTO Ventas VALUES (:i, :r, :m, :f)"),
                {"i": i, "r": "Norte" if i % 2 else "Sur", "m": i * 1.5 if i % 5 else None, "f": f"2024-01-{i:02d}"},
            )
    engine.dispose()
    return f"sqlite:///{path}"


def importer(engine_key, sqlite_url, seen=None, **kwargs):
    def factory(settings):
        if seen is not None:
            seen.append(settings)
        return create_engine(sqlite_url)

    return ExternalSqlImporter(engine_key, engine_factory=factory, **kwargs)


# ------------------------------------------------------------------ SQL sources (SQLite stands in for the server)
@pytest.mark.parametrize("engine_key", ["mysql", "postgresql"])
def test_a_query_returns_typed_columns_and_never_the_password(engine_key, sqlite_url):
    frame, detected = importer(engine_key, sqlite_url).read(
        request(engine_key, query="SELECT id, region, monto FROM Ventas WHERE id <= 5 ORDER BY id")
    )
    assert frame["id"].tolist() == [1, 2, 3, 4, 5] and frame["monto"].isna().sum() == 1
    assert frame["monto"].dtype.kind == "f"
    assert detected["host"] == "db.interna" and detected["rows_truncated"] is False and "query" in detected
    assert PASSWORD not in str(detected)


@pytest.mark.parametrize("engine_key", ["mysql", "postgresql"])
def test_a_whole_table_is_read_with_the_row_cap(engine_key, sqlite_url):
    frame, detected = importer(engine_key, sqlite_url).read(request(engine_key, table="Ventas", max_rows=10))
    assert len(frame) == 10 and detected["rows_truncated"] is True and detected["table"] == "Ventas"
    whole, detected = importer(engine_key, sqlite_url).read(request(engine_key, table="Ventas", max_rows=100))
    assert len(whole) == 25 and detected["rows_truncated"] is False


def test_table_names_are_matched_ignoring_case_and_views_work(sqlite_url):
    reader = importer("postgresql", sqlite_url)
    assert len(reader.read(request("postgresql", table="ventas"))[0]) == 25
    assert len(reader.read(request("postgresql", table="VISTA_NORTE"))[0]) == 13
    with pytest.raises(InvalidInputError, match="Ver tablas"):
        reader.read(request("postgresql", table="no_existe"))


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Ventas",
        "SELECT 1; DROP TABLE Ventas",
        "SELECT * INTO copia FROM Ventas",
        "UPDATE Ventas SET id = 0",
    ],
)
def test_anything_but_a_select_is_refused_before_connecting(sql, sqlite_url):
    seen = []
    with pytest.raises(InvalidInputError):
        importer("mysql", sqlite_url, seen).read(request("mysql", query=sql))
    assert seen == []  # the guard runs before any connection is made


def test_nothing_to_read_is_a_clear_error(sqlite_url):
    with pytest.raises(InvalidInputError, match="consulta SQL o indique"):
        importer("mysql", sqlite_url).read(request("mysql"))


def test_driver_failures_become_a_clean_message_without_the_password(sqlite_url):
    with pytest.raises(ImportFailedError) as failure:
        importer("postgresql", sqlite_url).read(request("postgresql", query="SELECT * FROM tabla_inexistente"))
    assert "PostgreSQL" in str(failure.value) and PASSWORD not in str(failure.value)


def test_the_form_goes_through_strict_validation(sqlite_url):
    seen = []
    importer("postgresql", sqlite_url, seen).read(
        request("postgresql", port="5433", query="SELECT 1 AS uno", sslmode="require", username="lector")
    )
    settings = seen[0]
    assert (settings.host, settings.port, settings.username) == ("db.interna", 5433, "lector")
    assert dict(settings.options) == {"sslmode": "require"}
    for bad in ({"host": "a b"}, {"host": ""}, {"database": "x;y"}, {"port": "abc"}, {"sslmode": "nope"}):
        with pytest.raises(InvalidInputError):
            importer("postgresql", sqlite_url).read(request("postgresql", query="SELECT 1", **bad))


def test_the_administrators_server_list_is_enforced(sqlite_url):
    reader = importer("mysql", sqlite_url, allowed_hosts=("db.interna", "otro.local"))
    assert len(reader.read(request("mysql", query="SELECT 1 AS uno"))[0]) == 1
    with pytest.raises(InvalidInputError, match="lista de servidores permitidos"):
        reader.read(request("mysql", host="atacante.com", query="SELECT 1"))


def test_the_explorer_lists_tables_views_and_columns(sqlite_url):
    listing = importer("mysql", sqlite_url).inspect(request("mysql"))
    assert {(t["name"], t["kind"]) for t in listing["tables"]} == {("Ventas", "tabla"), ("vista_norte", "vista")}
    detail = importer("mysql", sqlite_url).inspect(request("mysql", table="ventas"))
    assert [c["name"] for c in detail["columns"]] == ["id", "region", "monto", "fecha"]
    assert PASSWORD not in str(detail)


# ------------------------------------------------------------------ MongoDB (mongomock stands in for the server)
@pytest.fixture
def mongo():
    import mongomock
    from bson import Decimal128, ObjectId

    client = mongomock.MongoClient()
    client["datos"]["clientes"].insert_many(
        [
            {"_id": ObjectId(), "nombre": "Ana", "edad": 31, "gasto": Decimal128("12.50"), "dir": {"ciudad": "Lima"}},
            {"_id": ObjectId(), "nombre": "Luis", "edad": 25, "dir": {"ciudad": "Quito"}, "etiquetas": ["a", "b"]},
            {"_id": ObjectId(), "nombre": "Marta", "edad": 47, "alta": dt.datetime(2024, 3, 1)},
        ]
    )
    return client


def mongo_importer(client, **kwargs):
    return MongoImporter(client_factory=lambda settings: client, **kwargs)


def mongo_request(**options):
    return ImportRequest("mongodb", {"host": "mongo.interna", "database": "datos", "collection": "clientes", **options})


def test_documents_are_flattened_into_typed_columns(mongo):
    frame, detected = mongo_importer(mongo).read(mongo_request())
    assert len(frame) == 3 and {"_id", "nombre", "edad", "gasto", "dir.ciudad", "etiquetas", "alta"} <= set(
        frame.columns
    )
    assert frame["_id"].map(type).eq(str).all() and frame["gasto"].dropna().iloc[0] == 12.5
    assert frame["dir.ciudad"].tolist()[:2] == ["Lima", "Quito"]
    assert detected["collection"] == "clientes" and detected["rows_truncated"] is False


def test_filters_fields_and_the_document_cap(mongo):
    reader = mongo_importer(mongo)
    older = reader.read(mongo_request(filter='{"edad": {"$gt": 30}}'))[0]
    assert sorted(older["nombre"]) == ["Ana", "Marta"]
    only = reader.read(mongo_request(fields="nombre, edad"))[0]
    assert set(only.columns) == {"_id", "nombre", "edad"}
    frame, detected = reader.read(mongo_request(max_rows=2))
    assert len(frame) == 2 and detected["rows_truncated"] is True


@pytest.mark.parametrize(
    "bad_filter",
    [
        '{"$where": "sleep(5000)"}',
        '{"a": {"$function": {"body": "x", "args": [], "lang": "js"}}}',
        '{"$expr": {"$eq": [1, 1]}}',
    ],
)
def test_filters_that_run_code_on_the_server_are_refused(mongo, bad_filter):
    with pytest.raises(InvalidInputError):
        mongo_importer(mongo).read(mongo_request(filter=bad_filter))


@pytest.mark.parametrize(
    "bad", [{"filter": "no es json"}, {"filter": "[1, 2]"}, {"collection": "system.users"}, {"collection": ""}]
)
def test_malformed_requests_are_rejected(mongo, bad):
    with pytest.raises(InvalidInputError):
        mongo_importer(mongo).read(mongo_request(**bad))


def test_an_empty_result_is_explained(mongo):
    with pytest.raises(ImportFailedError, match="no tiene documentos"):
        mongo_importer(mongo).read(mongo_request(filter='{"edad": {"$gt": 100}}'))


def test_the_mongo_explorer_lists_collections_and_fields(mongo):
    listing = mongo_importer(mongo).inspect(ImportRequest("mongodb", {"host": "m", "database": "datos"}))
    assert [t["name"] for t in listing["tables"]] == ["clientes"]
    detail = mongo_importer(mongo).inspect(mongo_request())
    assert {"nombre", "dir.ciudad"} <= {c["name"] for c in detail["columns"]}


def test_the_server_list_also_applies_to_mongodb(mongo):
    with pytest.raises(InvalidInputError, match="lista de servidores permitidos"):
        mongo_importer(mongo, allowed_hosts=("solo.este",)).read(mongo_request())


def test_parse_filter_reads_extended_json_dates():
    parsed = parse_filter('{"alta": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}}')
    assert isinstance(parsed["alta"]["$gte"], dt.datetime)


# ------------------------------------------------------------------ registry
def test_the_registry_offers_every_engine_with_its_driver_status():
    registry = DefaultImporterRegistry(allowed_hosts=("a.local",))
    engines = {item["source_type"]: item for item in registry.external_databases()}
    assert set(engines) == {"sqlserver", "mysql", "postgresql", "oracle", "mongodb"}
    assert engines["postgresql"]["default_port"] == 5432 and engines["mongodb"]["install_hint"] == "pymongo"
    assert all(registry.get(key).source_type == key for key in engines)
