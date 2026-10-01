"""Every account is a sealed workspace: what user A owns must look *non-existent* (404) to user B."""

import pytest

from tests.integration.conftest import ApiClient, import_frame

ANALYSES = {
    "descriptive": ("/api/descriptive/run", {"numeric_columns": ["ventas"]}),
    "regression": ("/api/regression/run", {"response": "ventas", "predictors": ["precio"]}),
    "anova": ("/api/anova/run", {"response": "ventas", "factors": ["region"]}),
    "hypothesis": (
        "/api/hypothesis/run",
        {"test_id": "t_one_sample", "roles": {"x": "precio"}, "params": {"mu0": 50}},
    ),
    "draw": ("/api/markets/draw", {"design": "srs", "sample_size": 10}),
    "estimate": ("/api/markets/estimate", {"variable": "canal"}),
    "crosstab": ("/api/markets/crosstab", {"row_variable": "region", "col_variable": "canal"}),
    "likert": ("/api/markets/likert", {"items": ["comprador", "comprador"]}),
}


@pytest.fixture
def alice(api, sales_frame):
    """Account A (the administrator) with a dataset, a saved result and a notebook."""
    dataset = import_frame(api, sales_frame, "ventas")
    saved = api.post(
        "/api/results",
        {
            "module": "descriptive",
            "analysis_type": "summary",
            "title": "Privado de A",
            "dataset_id": dataset["id"],
            "report": {"blocks": [{"type": "text", "text": "secreto"}]},
        },
    ).parsed["result"]
    notebook = api.post("/api/notebooks", {"name": "cuaderno de A"}).parsed["notebook"]
    return {"client": api, "dataset": dataset, "result": saved, "notebook": notebook}


@pytest.fixture
def bob(make_client) -> ApiClient:
    return make_client("bob", console=True)


# ------------------------------------------------------------------ datasets
def test_other_accounts_do_not_see_my_datasets(alice, bob):
    assert bob.get("/api/datasets").parsed["datasets"] == []
    assert [d["id"] for d in alice["client"].get("/api/datasets").parsed["datasets"]] == [alice["dataset"]["id"]]


def test_dataset_endpoints_answer_404_for_a_foreign_id(alice, bob):
    ds = alice["dataset"]["id"]
    for response in (
        bob.get(f"/api/datasets/{ds}"),
        bob.get(f"/api/datasets/{ds}/preview"),
        bob.patch(f"/api/datasets/{ds}", {"name": "robado"}),
        bob.post(f"/api/datasets/{ds}/column-types", {"column_kinds": {"region": "categorical"}}),
        bob.delete(f"/api/datasets/{ds}"),
    ):
        assert response.status_code == 404, response.request.path
        assert response.parsed["error"]["code"] == "dataset_not_found"
    mine = alice["client"].get(f"/api/datasets/{ds}")  # nothing happened to the real owner's copy
    assert mine.status_code == 200 and mine.parsed["dataset"]["name"] == "ventas"


@pytest.mark.parametrize("analysis", sorted(ANALYSES))
def test_analyses_refuse_a_foreign_dataset(alice, bob, analysis):
    path, payload = ANALYSES[analysis]
    response = bob.post(path, {"dataset_id": alice["dataset"]["id"], **payload})
    assert response.status_code == 404, response.get_data(as_text=True)
    assert alice["client"].post(path, {"dataset_id": alice["dataset"]["id"], **payload}).status_code == 200


def test_the_same_dataset_name_can_exist_for_two_accounts(alice, bob, sales_frame):
    other = import_frame(bob, sales_frame.head(20), "ventas")
    assert other["id"] != alice["dataset"]["id"] and other["n_rows"] == 20
    assert alice["client"].get(f"/api/datasets/{alice['dataset']['id']}").parsed["dataset"]["n_rows"] == 200


# ------------------------------------------------------------------ saved results
def test_other_accounts_do_not_see_my_results(alice, bob):
    rid = alice["result"]["id"]
    assert bob.get("/api/results").parsed["total"] == 0
    for response in (
        bob.get(f"/api/results/{rid}"),
        bob.patch(f"/api/results/{rid}", {"title": "hackeado"}),
        bob.get(f"/api/results/{rid}/export"),
        bob.delete(f"/api/results/{rid}"),
    ):
        assert response.status_code == 404
    still_mine = alice["client"].get(f"/api/results/{rid}").parsed["result"]
    assert still_mine["title"] == "Privado de A"


def test_saving_a_result_never_links_to_a_foreign_dataset(alice, bob):
    saved = bob.post(
        "/api/results",
        {
            "module": "anova",
            "analysis_type": "x",
            "title": "intento",
            "dataset_id": alice["dataset"]["id"],
            "report": {"a": 1},
        },
    )
    assert saved.status_code == 201
    summary = saved.parsed["result"]
    assert summary["dataset_id"] is None and summary["dataset_name"] is None  # no name leaks back to the caller


# ------------------------------------------------------------------ notebooks and kernels
def test_other_accounts_do_not_see_my_notebooks(alice, bob):
    nid = alice["notebook"]["id"]
    assert bob.get("/api/notebooks").parsed["notebooks"] == []
    for response in (
        bob.get(f"/api/notebooks/{nid}"),
        bob.put(f"/api/notebooks/{nid}", {"cells": []}),
        bob.patch(f"/api/notebooks/{nid}", {"name": "mio"}),
        bob.get(f"/api/notebooks/{nid}/export"),
        bob.post(f"/api/notebooks/{nid}/execute", {"source": "1 + 1"}),
        bob.get(f"/api/notebooks/{nid}/kernel"),
        bob.post(f"/api/notebooks/{nid}/kernel/start"),
        bob.post(f"/api/notebooks/{nid}/kernel/restart"),
        bob.delete(f"/api/notebooks/{nid}"),
    ):
        assert response.status_code == 404, response.request.path
    assert alice["client"].get(f"/api/notebooks/{nid}").parsed["notebook"]["name"] == "cuaderno de A"


def test_a_notebook_id_in_a_path_cannot_climb_out_of_the_users_folder(bob):
    for evil in ("..%2F1%2Fx", "..", "%2e%2e%2f%2e%2e%2fsigma"):
        assert bob.get(f"/api/notebooks/{evil}").status_code in (400, 404)


def test_kernels_only_reach_the_data_of_their_own_account(alice, bob, sales_frame):
    own = import_frame(bob, sales_frame.head(5), "solo_de_bob")
    notebook = bob.post("/api/notebooks", {"name": "espía"}).parsed["notebook"]
    run = bob.post(
        f"/api/notebooks/{notebook['id']}/execute",
        {"source": "print(sorted(sigma.datasets()['nombre']))\nprint(len(sigma.load('solo_de_bob')))"},
    )
    assert run.status_code == 200 and run.parsed["status"] == "ok", run.parsed
    assert run.parsed["outputs"][0]["text"] == "['solo_de_bob']\n5\n"  # A's «ventas» is invisible
    peek = bob.post(f"/api/notebooks/{notebook['id']}/execute", {"source": "sigma.load('ventas')"})
    assert peek.parsed["status"] == "error" and "ventas" in str(peek.parsed["outputs"])
    assert own["name"] == "solo_de_bob"
    # what a kernel saves lands in its owner's account only
    assert (
        bob.post(
            f"/api/notebooks/{notebook['id']}/execute", {"source": "sigma.save(sigma.load('solo_de_bob'), 'copia')"}
        )
    ).parsed["status"] == "ok"
    assert {d["name"] for d in bob.get("/api/datasets").parsed["datasets"]} == {"solo_de_bob", "copia"}
    assert [d["name"] for d in alice["client"].get("/api/datasets").parsed["datasets"]] == ["ventas"]


def test_the_kernel_environment_carries_no_secrets(bob, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-no-debe-verse")
    notebook = bob.post("/api/notebooks", {"name": "env"}).parsed["notebook"]
    run = bob.post(
        f"/api/notebooks/{notebook['id']}/execute",
        {"source": "import os\nprint(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('SIGMA_USER_ID'))"},
    )
    assert run.status_code == 200 and run.parsed["outputs"][0]["text"].startswith("None ")


# ------------------------------------------------------------------ adoption of single-user data
def test_the_first_administrator_inherits_the_pre_accounts_data(tmp_path):
    """A v1 database (no accounts) keeps working: /setup assigns its datasets, results and notebooks."""
    import sqlite3

    from sigma.app_factory import create_app
    from sigma.config import Config
    from sigma.infrastructure.persistence.sqlite.migrations import V1_BASE
    from tests.integration.conftest import PASSWORD

    db = tmp_path / "sigma.db"
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / ("c" * 32 + ".ipynb")).write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}', encoding="utf-8"
    )
    conn = sqlite3.connect(db)
    conn.executescript(V1_BASE + "PRAGMA user_version = 1;")
    conn.execute('CREATE TABLE "ds_' + "d" * 32 + '" (x INTEGER)')
    conn.execute('INSERT INTO "ds_' + "d" * 32 + '" VALUES (7)')
    conn.execute(
        "INSERT INTO datasets VALUES (?, 'heredado', 'csv', '{}', 1, 1, '{\"x\": \"int64\"}',"
        ' \'[{"name":"x","kind":"numeric","pandas_dtype":"int64","n_missing":0,"n_unique":1}]\', \'ayer\')',
        ("d" * 32,),
    )
    conn.commit()
    conn.close()

    app = create_app(Config.from_env(testing=True, instance_dir=tmp_path, db_path=db, notebooks_dir=notebooks))
    admin = ApiClient(app.test_client())
    created = admin.post("/api/auth/setup", {"username": "ana", "password": PASSWORD})
    assert created.status_code == 201 and created.parsed["adopted"] == {"datasets": 1, "results": 0, "notebooks": 1}
    assert [d["name"] for d in admin.get("/api/datasets").parsed["datasets"]] == ["heredado"]
    assert len(admin.get("/api/notebooks").parsed["notebooks"]) == 1
    app.extensions["sigma"].create_user.execute("bob", "Bob", PASSWORD, "user", True, False)
    bob = ApiClient(app.test_client())
    assert bob.login("bob").status_code == 200
    assert bob.get("/api/datasets").parsed["datasets"] == []  # only the first administrator inherits
    assert bob.get("/api/notebooks").parsed["notebooks"] == []
