"""Pending imports: the uploaded file and its preview outlive page changes, restarts and other windows."""

import io

import pytest
from openpyxl import Workbook

from sigma.app_factory import create_app
from sigma.config import Config
from tests.integration.conftest import ApiClient, import_frame

CSV = b"nombre;edad\nAna;30\nLuis;41\nMarta;28\n"


def upload(client, content=CSV, name="gente.csv", source="csv"):
    return client.post(
        "/api/datasets/import/draft",
        data={"source_type": source, "file": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


def preview(client, draft_id, **options):
    return client.post("/api/datasets/import/preview", {"source_type": "csv", "draft_id": draft_id, "options": options})


def test_upload_keeps_the_file_and_lists_it(api):
    created = upload(api)
    assert created.status_code == 201
    draft = created.parsed["draft"]
    assert draft["file_name"] == "gente.csv" and draft["file_size"] == len(CSV) and not draft["has_preview"]
    assert api.get("/api/datasets/import/drafts").parsed["drafts"] == [draft]


def test_the_stored_file_is_reanalysed_with_new_options_without_uploading_again(api):
    draft = upload(api).parsed["draft"]
    wrong = preview(api, draft["id"])  # auto-detection
    assert wrong.status_code == 200 and wrong.parsed["token"] == draft["id"]
    forced = preview(api, draft["id"], delimiter="semicolon")
    assert forced.status_code == 200
    assert [c["name"] for c in forced.parsed["columns"]] == ["nombre", "edad"] and forced.parsed["n_rows"] == 3
    assert api.get("/api/datasets/import/drafts").parsed["drafts"][0]["options"] == {"delimiter": "semicolon"}


def test_the_preview_can_be_rebuilt_later_and_committed(api):
    draft = upload(api).parsed["draft"]
    shown = preview(api, draft["id"], delimiter="semicolon").parsed
    again = api.get(f"/api/datasets/import/drafts/{draft['id']}")
    assert again.status_code == 200
    assert again.parsed == shown  # exactly what the user saw before changing page
    committed = api.post(
        "/api/datasets/import/commit", {"token": draft["id"], "name": "gente", "column_kinds": {"edad": "text"}}
    )
    assert committed.status_code == 201 and committed.parsed["dataset"]["n_rows"] == 3
    assert api.get("/api/datasets/import/drafts").parsed["drafts"] == []
    assert api.get(f"/api/datasets/import/drafts/{draft['id']}").status_code == 422  # consumed


def test_a_failed_commit_keeps_the_pending_import(api, sales_frame):
    import_frame(api, sales_frame.head(3), "gente")
    draft = upload(api).parsed["draft"]
    preview(api, draft["id"], delimiter="semicolon")
    clash = api.post("/api/datasets/import/commit", {"token": draft["id"], "name": "gente"})
    assert clash.status_code == 422 and "Ya existe" in clash.parsed["error"]["message"]
    assert api.post("/api/datasets/import/commit", {"token": draft["id"], "name": "gente 2"}).status_code == 201


def test_uploading_a_new_file_replaces_the_pending_one(api):
    first = upload(api).parsed["draft"]
    preview(api, first["id"], delimiter="semicolon")
    second = upload(api, b"x\n1\n", "otro.csv").parsed["draft"]
    assert second["id"] == first["id"] and second["file_name"] == "otro.csv" and not second["has_preview"]
    assert api.get(f"/api/datasets/import/drafts/{first['id']}").status_code == 422  # its old preview is gone


def test_discarding_only_the_preview_keeps_the_file(api):
    draft = upload(api).parsed["draft"]
    preview(api, draft["id"], delimiter="semicolon")
    assert api.delete(f"/api/datasets/import/drafts/{draft['id']}?preview_only=1").status_code == 200
    kept = api.get("/api/datasets/import/drafts").parsed["drafts"][0]
    assert kept["file_name"] == "gente.csv" and not kept["has_preview"]
    assert preview(api, draft["id"], delimiter="semicolon").status_code == 200  # still analysable
    assert api.delete(f"/api/datasets/import/drafts/{draft['id']}").status_code == 200
    assert api.get("/api/datasets/import/drafts").parsed["drafts"] == []


def test_one_step_upload_and_preview_also_keeps_the_file(api, sales_frame):
    """The original multipart flow still works, and now leaves the file as the pending import."""
    dataset = import_frame(api, sales_frame.head(4), "una_vez")  # preview + commit in one go
    assert dataset["n_rows"] == 4 and api.get("/api/datasets/import/drafts").parsed["drafts"] == []
    response = api.post(
        "/api/datasets/import/preview",
        data={"source_type": "csv", "options": "{}", "file": (io.BytesIO(CSV), "gente.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    draft = api.get("/api/datasets/import/drafts").parsed["drafts"][0]
    assert draft["id"] == response.parsed["token"] and draft["has_preview"] and draft["file_name"] == "gente.csv"


def test_excel_sheets_come_from_the_stored_workbook(api):
    workbook = Workbook()
    workbook.active.title = "Ventas"
    workbook.active.append(["a", "b"])
    workbook.active.append([1, 2])
    workbook.create_sheet("Otra").append(["c"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    draft = upload(api, buffer.getvalue(), "libro.xlsx", "excel").parsed["draft"]
    sheets = api.post("/api/datasets/import/inspect", {"source_type": "excel", "draft_id": draft["id"], "options": {}})
    assert sheets.parsed["sheets"] == ["Ventas", "Otra"]
    chosen = api.post(
        "/api/datasets/import/preview",
        {"source_type": "excel", "draft_id": draft["id"], "options": {"sheet": "Otra"}},
    )
    assert [c["name"] for c in chosen.parsed["columns"]] == ["c"]


def test_validation(api):
    assert upload(api, b"").status_code == 422
    assert upload(api, source="ftp").status_code == 422
    assert api.post("/api/datasets/import/draft", {"source_type": "csv"}).status_code == 422  # no file
    draft = upload(api).parsed["draft"]
    wrong_type = api.post(
        "/api/datasets/import/preview", {"source_type": "excel", "draft_id": draft["id"], "options": {}}
    )
    assert wrong_type.status_code == 422 and "otro tipo" in wrong_type.parsed["error"]["message"]
    assert preview(api, "0" * 32).status_code == 422  # unknown draft
    assert api.get("/api/datasets/import/drafts/not-an-id").status_code == 422


def test_drafts_are_private_per_account(api, make_client):
    draft = upload(api).parsed["draft"]
    preview(api, draft["id"], delimiter="semicolon")
    other = make_client("luis")
    assert other.get("/api/datasets/import/drafts").parsed["drafts"] == []
    for response in (
        other.get(f"/api/datasets/import/drafts/{draft['id']}"),
        other.delete(f"/api/datasets/import/drafts/{draft['id']}"),
        preview(other, draft["id"]),
        other.post("/api/datasets/import/commit", {"token": draft["id"], "name": "robado"}),
    ):
        assert response.status_code == 422 and "ya no existe" in response.parsed["error"]["message"]
    assert api.get(f"/api/datasets/import/drafts/{draft['id']}").status_code == 200


def test_pending_imports_survive_a_restart_and_show_up_in_sync(app, api, tmp_path):
    before = api.get("/api/sync").parsed["drafts"]
    draft = upload(api).parsed["draft"]
    preview(api, draft["id"], delimiter="semicolon")
    assert api.get("/api/sync").parsed["drafts"] > before  # other windows learn about it
    # a brand-new application on the same database = the server was restarted
    restarted = create_app(
        Config.from_env(
            testing=True,
            instance_dir=tmp_path,
            db_path=tmp_path / "sigma.db",
            notebooks_dir=tmp_path / "notebooks",
            sample_dir=tmp_path / "samples",
        )
    )
    client = ApiClient(restarted.test_client())
    assert client.login("admin").status_code == 200
    assert client.get(f"/api/datasets/import/drafts/{draft['id']}").parsed["n_rows"] == 3
    assert (
        client.post("/api/datasets/import/commit", {"token": draft["id"], "name": "tras_reinicio"}).status_code == 201
    )


def test_deleting_an_account_removes_its_pending_imports(app, api, make_client):
    other = make_client("luis")
    draft = upload(other).parsed["draft"]
    preview(other, draft["id"], delimiter="semicolon")
    luis = other.get("/api/auth/me").parsed["user"]["id"]
    assert api.delete(f"/api/users/{luis}").status_code == 200
    with app.extensions["sigma"].database.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM import_drafts").fetchone()[0] == 0
        assert [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'stg_%'")] == []


@pytest.mark.parametrize("source", ["sqlserver", "json_api"])
def test_connection_secrets_never_reach_the_database(app, api, source):
    """Even when the query fails, nothing typed in the connection form is stored (only successful previews are)."""
    options = {
        "server": "x",
        "password": "NO-GUARDAR",
        "headers": "Authorization: Bearer NO-GUARDAR",
        "url": "http://127.0.0.1:9/x?t=NO-GUARDAR",
    }
    api.post("/api/datasets/import/preview", {"source_type": source, "options": options})
    with app.extensions["sigma"].database.connection() as conn:
        dump = " ".join(
            str(v)
            for table in ("import_drafts", "workspace")
            for row in conn.execute(f"SELECT * FROM {table}")
            for v in tuple(row)
        )
    assert "NO-GUARDAR" not in dump


def test_clearing_the_datos_workspace_also_discards_pending_imports(api):
    draft = upload(api).parsed["draft"]
    preview(api, draft["id"], delimiter="semicolon")
    api.delete("/api/workspace/anova")  # other modules do not touch them
    assert len(api.get("/api/datasets/import/drafts").parsed["drafts"]) == 1
    assert api.delete("/api/workspace/datos").status_code == 200
    assert api.get("/api/datasets/import/drafts").parsed["drafts"] == []
    upload(api)
    assert api.delete("/api/workspace").status_code == 200  # «Limpiar todos los módulos»
    assert api.get("/api/datasets/import/drafts").parsed["drafts"] == []
