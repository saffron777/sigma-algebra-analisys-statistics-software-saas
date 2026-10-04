"""The «Base de datos» source through the API: engines on offer, explorer, preview/commit, secrets and refusals."""

import pytest
from sqlalchemy import create_engine, text

from sigma.app_factory import create_app
from sigma.config import Config
from sigma.infrastructure.importers.external_sql_importer import ExternalSqlImporter
from sigma.infrastructure.importers.mongo_importer import MongoImporter
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from tests.integration.conftest import PASSWORD, ApiClient
from tests.support import dump

SECRET = "NO-GUARDAR-123"


@pytest.fixture
def sources(app, tmp_path):
    """Replaces the real database importers by ones that read a local SQLite file / an in-memory MongoDB."""
    import mongomock

    url = f"sqlite:///{tmp_path / 'externa.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE ventas (id INTEGER, region TEXT, monto REAL)"))
        for i in range(1, 41):
            conn.execute(
                text("INSERT INTO ventas VALUES (:i, :r, :m)"), {"i": i, "r": "Norte" if i % 2 else "Sur", "m": i * 2.5}
            )
    engine.dispose()
    client = mongomock.MongoClient()
    client["datos"]["clientes"].insert_many(
        [{"nombre": f"C{i}", "edad": 20 + i, "dir": {"ciudad": "Lima"}} for i in range(12)]
    )
    app.extensions["sigma"].importers = DefaultImporterRegistry(
        [
            ExternalSqlImporter("mysql", engine_factory=lambda settings: create_engine(url)),
            ExternalSqlImporter("postgresql", engine_factory=lambda settings: create_engine(url)),
            MongoImporter(client_factory=lambda settings: client),
        ]
    )


def connection(**extra):
    return {"host": "db.interna", "database": "datos", "username": "lector", "password": SECRET, **extra}


def test_the_engines_on_offer_are_listed_with_their_driver_status(api):
    databases = api.get("/api/datasets/import/databases").parsed["databases"]
    assert [d["source_type"] for d in databases] == ["sqlserver", "mysql", "postgresql", "oracle", "mongodb"]
    assert all({"label", "default_port", "driver_available", "install_hint"} <= set(d) for d in databases)


def test_a_table_is_previewed_and_saved_without_leaving_secrets(app, api, sources):
    preview = api.post(
        "/api/datasets/import/preview",
        {"source_type": "postgresql", "options": connection(table="ventas", schema="main", max_rows=1000)},
    )
    assert preview.status_code == 200, preview.get_data(as_text=True)
    body = preview.parsed
    assert body["n_rows"] == 40 and body["suggested_name"] == "ventas"
    assert body["detected"]["host"] == "db.interna" and SECRET not in str(body)
    drafts = api.get("/api/datasets/import/drafts").parsed["drafts"]
    assert drafts[0]["source_type"] == "postgresql" and "password" not in drafts[0]["options"]
    saved = api.post("/api/datasets/import/commit", {"token": body["token"], "name": "ventas_externas"})
    assert saved.status_code == 201 and saved.parsed["dataset"]["source_type"] == "postgresql"
    assert SECRET not in dump(app.extensions["sigma"], ("import_drafts", "workspace", "datasets"))


def test_each_engine_keeps_its_own_pending_import(api, sources):
    api.post(
        "/api/datasets/import/preview", {"source_type": "mysql", "options": connection(query="SELECT id FROM ventas")}
    )
    api.post("/api/datasets/import/preview", {"source_type": "postgresql", "options": connection(table="ventas")})
    api.post(
        "/api/datasets/import/preview",
        {"source_type": "mongodb", "options": {"host": "m.interna", "database": "datos", "collection": "clientes"}},
    )
    drafts = {d["source_type"]: d for d in api.get("/api/datasets/import/drafts").parsed["drafts"]}
    assert set(drafts) == {"mysql", "postgresql", "mongodb"} and drafts["mysql"]["n_cols"] == 1


def test_mongodb_documents_become_columns(api, sources):
    preview = api.post(
        "/api/datasets/import/preview",
        {
            "source_type": "mongodb",
            "options": {
                "host": "m.interna",
                "database": "datos",
                "collection": "clientes",
                "filter": '{"edad": {"$gte": 28}}',
            },
        },
    ).parsed
    assert preview["n_rows"] == 4 and "dir.ciudad" in [c["name"] for c in preview["columns"]]


def test_the_explorer_lists_tables_and_columns(api, sources):
    listing = api.post("/api/datasets/import/inspect", {"source_type": "postgresql", "options": connection()}).parsed
    assert [t["name"] for t in listing["tables"]] == ["ventas"] and SECRET not in str(listing)
    detail = api.post(
        "/api/datasets/import/inspect", {"source_type": "mysql", "options": connection(table="ventas")}
    ).parsed
    assert [c["name"] for c in detail["columns"]] == ["id", "region", "monto"]
    collections = api.post(
        "/api/datasets/import/inspect", {"source_type": "mongodb", "options": {"host": "m", "database": "datos"}}
    ).parsed
    assert [t["name"] for t in collections["tables"]] == ["clientes"]


@pytest.mark.parametrize("source", ["mysql", "postgresql", "oracle"])
@pytest.mark.parametrize("query", ["DROP TABLE users", "SELECT 1; DELETE FROM users", "SELECT * INTO copia FROM t"])
def test_a_write_is_refused_before_any_connection(api, source, query):
    refused = api.post("/api/datasets/import/preview", {"source_type": source, "options": connection(query=query)})
    assert refused.status_code == 422


def test_mongodb_operators_that_run_code_are_refused(api):
    refused = api.post(
        "/api/datasets/import/preview",
        {
            "source_type": "mongodb",
            "options": {"host": "m", "database": "d", "collection": "c", "filter": '{"$where": "1==1"}'},
        },
    )
    assert refused.status_code == 422 and "$where" in refused.parsed["error"]["message"]


@pytest.mark.parametrize("source", ["mysql", "postgresql", "oracle", "mongodb"])
def test_malformed_connection_details_are_rejected(api, source):
    options = connection(host="bad host!", query="SELECT 1", collection="c")
    assert api.post("/api/datasets/import/inspect", {"source_type": source, "options": options}).status_code == 422


def test_the_administrators_server_list_applies(tmp_path):
    config = Config.from_env(
        testing=True,
        instance_dir=tmp_path,
        db_path=tmp_path / "sigma.db",
        notebooks_dir=tmp_path / "nb",
        sample_dir=tmp_path / "s",
        external_db_hosts=("permitido.local",),
    )
    app = create_app(config)
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    client = ApiClient(app.test_client())
    assert client.login("admin").status_code == 200
    for source in ("mysql", "postgresql", "oracle", "mongodb"):
        refused = client.post(
            "/api/datasets/import/inspect",
            {"source_type": source, "options": {"host": "otro.servidor", "database": "d", "collection": "c"}},
        )
        assert refused.status_code == 422 and "lista de servidores permitidos" in refused.parsed["error"]["message"]
    sqlserver = client.post(
        "/api/datasets/import/preview",
        {
            "source_type": "sqlserver",
            "options": {"server": "otro\\INSTANCIA,1433", "database": "d", "query": "SELECT 1"},
        },
    )
    assert sqlserver.status_code == 422 and "lista de servidores permitidos" in sqlserver.parsed["error"]["message"]
