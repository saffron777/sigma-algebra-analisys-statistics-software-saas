import io
import json
import threading

import numpy as np
import pandas as pd
import pytest
from flask import Flask, jsonify
from flask import request as flask_request
from sqlalchemy import create_engine, text
from werkzeug.serving import make_server

from sigma.application.ports import ImportRequest
from sigma.application.use_cases.datasets import CommitImport, InspectSource, PreviewImport
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.file_importers import CsvImporter, ExcelImporter, TextImporter
from sigma.infrastructure.importers.json_api_importer import JsonApiImporter, parse_pairs
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from sigma.infrastructure.importers.sql_guard import build_table_query, validate_readonly_query
from sigma.infrastructure.importers.sqlserver_importer import SqlServerImporter, _odbc_value, sqlserver_engine
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.import_draft_store import SqlImportDraftStore
from tests.support import add_user


def req(source, content=None, name=None, **options):
    return ImportRequest(source, options, name, content)


def make_repository(path):
    database = Database(path)
    return SqlDatasetRepository(database, add_user(database))


def make_workspace(path):
    """A dataset repository and the draft store of the same account."""
    database = Database(path)
    user = add_user(database)
    return SqlDatasetRepository(database, user), SqlImportDraftStore(database, user)


# ------------------------------------------------------------------------- CSV / text
def test_csv_comma_and_dot_decimal():
    frame, detected = CsvImporter().read(req("csv", b"a,b,c\n1,2.5,x\n3,4.5,y\n"))
    assert frame["b"].tolist() == [2.5, 4.5] and detected["delimiter"] == "comma"


def test_csv_semicolon_with_decimal_comma_and_cp1252():
    content = "ciudad;precio\nMedellín;1,5\nBogotá;2,25\n".encode("cp1252")
    frame, detected = CsvImporter().read(req("csv", content))
    assert detected["delimiter"] == "semicolon" and detected["decimal"] == "," and detected["encoding"] == "cp1252"
    assert frame["precio"].tolist() == [1.5, 2.25] and frame["ciudad"][0] == "Medellín"


def test_csv_utf8_bom_tab_and_no_header():
    content = "﻿x\ty\n1\t2\n".encode()
    frame, detected = CsvImporter().read(req("csv", content))
    assert list(frame.columns) == ["x", "y"] and detected["encoding"] == "utf-8-sig"
    no_header, _ = CsvImporter().read(req("csv", b"1,2\n3,4\n", header=False))
    assert list(no_header.columns) == ["columna_1", "columna_2"] and len(no_header) == 2


def test_csv_errors():
    with pytest.raises(InvalidInputError):
        CsvImporter().read(req("csv"))
    with pytest.raises(ImportFailedError):
        CsvImporter().read(req("csv", b"   \n"))
    with pytest.raises(ImportFailedError):
        CsvImporter().read(req("csv", "ñ".encode(), encoding="ascii"))


def test_text_whitespace_and_fixed_width():
    frame, _ = TextImporter().read(req("text", b"a b c\n1 2 3\n4 5 6\n", delimiter="whitespace"))
    assert frame.shape == (2, 3) and frame["c"].tolist() == [3, 6]
    fixed = b"ID  NOMBRE  EDAD\n1   Ana     30\n2   Luis    41\n"
    frame, detected = TextImporter().read(req("text", fixed, layout="fixed"))
    assert frame["NOMBRE"].tolist() == ["Ana", "Luis"] and detected["layout"] == "fixed"
    frame, _ = TextImporter().read(req("text", fixed, layout="fixed", widths="4,8,4"))
    assert frame.shape == (2, 3)


# ------------------------------------------------------------------------- Excel
@pytest.fixture
def workbook() -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(writer, sheet_name="Ventas", index=False)
        pd.DataFrame({"z": [9.5]}).to_excel(writer, sheet_name="Otra", index=False, startrow=2)
    return buffer.getvalue()


def test_excel_default_sheet_and_listing(workbook):
    importer = ExcelImporter()
    assert importer.inspect(req("excel", workbook))["sheets"] == ["Ventas", "Otra"]
    frame, detected = importer.read(req("excel", workbook))
    assert frame["a"].tolist() == [1, 2] and detected["sheet"] == "Ventas"


def test_excel_sheet_by_name_index_and_header_row(workbook):
    importer = ExcelImporter()
    assert importer.read(req("excel", workbook, sheet="2", header_row=3))[0]["z"].tolist() == [9.5]
    assert importer.read(req("excel", workbook, sheet="Otra", header_row=3))[0].shape == (1, 1)
    with pytest.raises(InvalidInputError):
        importer.read(req("excel", workbook, sheet="Nope"))
    with pytest.raises(ImportFailedError):
        importer.read(req("excel", b"not an excel file"))


# ------------------------------------------------------------------------- SQL guard
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "  select a, b from t where x = 'DROP TABLE'",
        "WITH c AS (SELECT 1 a) SELECT * FROM c;",
        'SELECT [update], "delete" FROM t -- INSERT INTO x',
        "SELECT /* DELETE */ 1",
    ],
)
def test_guard_accepts_read_queries(sql):
    assert validate_readonly_query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DELETE FROM t",
        "SELECT 1; DROP TABLE t",
        "UPDATE t SET a = 1",
        "EXEC sp_who",
        "SELECT * INTO nueva FROM t",
        "WITH c AS (SELECT 1) DELETE FROM t",
        "SELECT * FROM OPENROWSET('x','y','z')",
        "INSERT INTO t VALUES (1)",
        "SELECT 1 /* ok */; SELECT 2",
        "TRUNCATE TABLE t",
        "SELECT xp_cmdshell('dir')",
        "MERGE t USING s ON 1=1",
    ],
)
def test_guard_rejects_anything_that_could_write(sql):
    with pytest.raises(InvalidInputError):
        validate_readonly_query(sql)


def test_table_query_quotes_identifiers():
    assert build_table_query("Ventas", "dbo", 10) == "SELECT TOP (11) * FROM [dbo].[Ventas]"
    assert build_table_query("Ventas 2024", None, 5) == "SELECT TOP (6) * FROM [Ventas 2024]"
    for malicious in ("a]b", "t; DROP TABLE x", "x' OR '1'='1", "t]; DROP TABLE x; --"):
        with pytest.raises(InvalidInputError):
            build_table_query(malicious, None, 5)
    with pytest.raises(InvalidInputError):
        build_table_query("t", "dbo]; --", 5)


def test_odbc_values_cannot_inject_attributes():
    assert _odbc_value("pa;ss}word") == "{pa;ss}}word}"


def test_sqlserver_engine_builds_url_without_connecting(monkeypatch):
    monkeypatch.setattr(
        "sigma.infrastructure.importers.sqlserver_importer.installed_driver",
        lambda _requested: "ODBC Driver 18 for SQL Server",
    )
    engine = sqlserver_engine(
        {
            "server": "srv,1433",
            "database": "db",
            "auth": "sql",
            "username": "u",
            "password": "p;Trusted_Connection=yes",
            "trust_server_certificate": True,
        }
    )
    connect_string = engine.url.query["odbc_connect"]
    assert "SERVER={srv,1433}" in connect_string and "PWD={p;Trusted_Connection=yes}" in connect_string
    assert "TrustServerCertificate=yes" in connect_string and "ApplicationIntent=ReadOnly" in connect_string
    with pytest.raises(InvalidInputError):
        sqlserver_engine({"database": "x"})
    with pytest.raises(InvalidInputError):
        sqlserver_engine({"server": "x", "database": "d", "auth": "sql"})


# ------------------------------------------------------------------------- SQL importer (SQLite engine, same code path)
@pytest.fixture
def sql_importer(tmp_path):
    path = tmp_path / "source.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE ventas (id INTEGER, monto REAL, region TEXT)"))
        conn.execute(text("INSERT INTO ventas VALUES (1, 10.5, 'N'), (2, 20.25, 'S'), (3, NULL, 'N')"))
    engine.dispose()
    return SqlServerImporter(engine_factory=lambda options: create_engine(f"sqlite:///{path}"))


def test_sql_query_returns_typed_frame(sql_importer):
    frame, detected = sql_importer.read(req("sqlserver", query="SELECT * FROM ventas WHERE region = 'N'", server="s"))
    assert frame.shape == (2, 3) and frame["monto"].isna().sum() == 1
    assert detected["rows_truncated"] is False and "password" not in detected


def test_sql_row_cap_flags_truncation(sql_importer):
    frame, detected = sql_importer.read(req("sqlserver", query="SELECT * FROM ventas", max_rows=2))
    assert len(frame) == 2 and detected["rows_truncated"] is True


def test_sql_importer_refuses_writes_and_never_commits(sql_importer):
    with pytest.raises(InvalidInputError):
        sql_importer.read(req("sqlserver", query="DELETE FROM ventas"))
    assert len(sql_importer.read(req("sqlserver", query="SELECT * FROM ventas"))[0]) == 3
    with pytest.raises(InvalidInputError):
        sql_importer.read(req("sqlserver"))


def test_sql_errors_are_reported_cleanly(sql_importer):
    with pytest.raises(ImportFailedError) as error:
        sql_importer.read(req("sqlserver", query="SELECT * FROM no_existe", server="srv"))
    assert "no_existe" in str(error.value)


# ------------------------------------------------------------------------- JSON API (real local HTTP server)
@pytest.fixture(scope="module")
def api_server():
    app = Flask("fake_api")

    @app.get("/items")
    def items():
        return jsonify(
            {
                "meta": {"page": 1},
                "data": {"items": [{"id": 1, "p": {"x": 1.5}, "tags": ["a"]}, {"id": 2, "p": {"x": 2.5}, "tags": []}]},
            }
        )

    @app.get("/list")
    def plain_list():
        return jsonify([{"a": 1}, {"a": 2}, {"a": 3}])

    @app.get("/auth")
    def auth():
        if flask_request.headers.get("Authorization") != "Bearer secreto" or flask_request.args.get("year") != "2024":
            return jsonify({"error": "no"}), 401
        return jsonify({"rows": [{"ok": True}]})

    @app.post("/search")
    def search():
        return jsonify([{"echo": flask_request.get_json()["q"]}])

    @app.get("/text")
    def not_json():
        return "hola", 200

    @app.get("/big")
    def big():
        return jsonify([{"v": "x" * 100}] * 2000)

    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_api_nested_records_with_path_and_autodetect(api_server):
    importer = JsonApiImporter()
    frame, detected = importer.read(req("json_api", url=f"{api_server}/items?token=SECRET", records_path="data.items"))
    assert (
        frame["p.x"].tolist() == [1.5, 2.5] and "token" not in detected["url"] and "SECRET" not in json.dumps(detected)
    )
    auto, _ = importer.read(req("json_api", url=f"{api_server}/items"))
    assert len(auto) == 2
    assert len(importer.read(req("json_api", url=f"{api_server}/list"))[0]) == 3


def test_api_headers_params_and_post_body(api_server):
    importer = JsonApiImporter()
    frame, _ = importer.read(
        req("json_api", url=f"{api_server}/auth", headers="Authorization: Bearer secreto", params="year=2024")
    )
    assert frame["ok"].tolist() == [True]
    frame, _ = importer.read(req("json_api", url=f"{api_server}/search", method="POST", body='{"q": "hola"}'))
    assert frame["echo"].tolist() == ["hola"]


def test_api_error_paths(api_server):
    importer = JsonApiImporter()
    with pytest.raises(ImportFailedError, match="401"):
        importer.read(req("json_api", url=f"{api_server}/auth"))
    with pytest.raises(ImportFailedError, match="JSON"):
        importer.read(req("json_api", url=f"{api_server}/text"))
    with pytest.raises(ImportFailedError, match="tamaño"):
        importer.read(req("json_api", url=f"{api_server}/big", max_bytes=1000))
    with pytest.raises(InvalidInputError):
        importer.read(req("json_api", url="file:///etc/passwd"))
    with pytest.raises(InvalidInputError):
        importer.read(req("json_api", url=f"{api_server}/items", records_path="data.nope"))
    with pytest.raises(InvalidInputError):
        importer.read(req("json_api", url=f"{api_server}/items", method="DELETE"))
    with pytest.raises(ImportFailedError):
        importer.read(req("json_api", url="http://127.0.0.1:9/nothing", timeout=1))


def test_parse_pairs():
    assert parse_pairs("A: 1\nB: dos: tres", ":", "x") == {"A": "1", "B": "dos: tres"}
    assert parse_pairs("a=1&b=2", "=", "x") == {"a": "1", "b": "2"}
    with pytest.raises(InvalidInputError):
        parse_pairs("sin separador", ":", "x")


# ------------------------------------------------------------------------- use cases: preview → commit
def test_preview_then_commit_flow(tmp_path):
    registry = DefaultImporterRegistry()
    repository, drafts = make_workspace(tmp_path / "s.db")
    content = b"Nombre,Edad,Nombre\nAna,30,x\nLuis,41,y\n"
    preview = PreviewImport(registry, drafts).execute(req("csv", content, "clientes.csv"))
    assert preview.suggested_name == "clientes" and preview.n_rows == 2
    assert [c.name for c in preview.columns] == ["Nombre", "Edad", "Nombre.1"]  # duplicate made unique
    info = CommitImport(drafts, repository).execute(preview.token, None, {"Edad": "text"})
    assert info.name == "clientes" and info.source_type == "csv"
    assert repository.get(info.id).data["Edad"].tolist() == ["30", "41"]
    with pytest.raises(InvalidInputError):  # a saved draft is gone
        CommitImport(drafts, repository).execute(preview.token, "otra", None)
    assert drafts.list() == []


def test_a_failed_commit_keeps_the_draft_so_it_can_be_retried(tmp_path):
    registry = DefaultImporterRegistry()
    repository, drafts = make_workspace(tmp_path / "r.db")
    repository.add("clientes", pd.DataFrame({"x": [1]}), "csv", {})
    token = PreviewImport(registry, drafts).execute(req("csv", b"a\n1\n2\n", "clientes.csv")).token
    with pytest.raises(InvalidInputError, match="Ya existe"):
        CommitImport(drafts, repository).execute(token, None, None)
    assert CommitImport(drafts, repository).execute(token, "clientes 2", None).n_rows == 2


def test_inspect_source_and_unknown_type(workbook, tmp_path):
    registry = DefaultImporterRegistry()
    _, drafts = make_workspace(tmp_path / "i.db")
    assert InspectSource(registry, drafts).execute(req("excel", workbook))["sheets"] == ["Ventas", "Otra"]
    assert InspectSource(registry, drafts).execute(req("csv", b"a\n1")) == {}
    with pytest.raises(InvalidInputError):
        registry.get("ftp")


def test_complex_json_cells_become_text(api_server, tmp_path):
    registry = DefaultImporterRegistry()
    repository, drafts = make_workspace(tmp_path / "j.db")
    request = req("json_api", url=f"{api_server}/items", records_path="data.items")
    preview = PreviewImport(registry, drafts).execute(request)
    assert preview.rows["tags"].map(lambda v: isinstance(v, str)).all()
    assert CommitImport(drafts, repository).execute(preview.token, "api").n_rows == 2


def test_numpy_scalars_in_object_columns_are_storable(tmp_path):
    repository = make_repository(tmp_path / "n.db")
    frame = pd.DataFrame({"mix": pd.Series([np.int64(1), "dos", np.float64(3.5), None], dtype=object)})
    info = repository.add("mix", frame, "csv", {})
    assert repository.get(info.id).data["mix"].tolist()[1] == "dos"
