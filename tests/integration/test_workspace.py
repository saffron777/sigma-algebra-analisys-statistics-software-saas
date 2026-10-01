"""Per-module workspace and the revision probe that keeps several windows of one user in step."""

import json

import pytest

from sigma.domain.workspace import WORKSPACE_MODULES
from tests.integration.conftest import ApiClient, import_frame
from tests.support import scalar


def put(client, module, **body):
    response = client.put(f"/api/workspace/{module}", body)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.parsed["revision"]


def test_an_untouched_module_has_an_empty_workspace(api):
    entry = api.get("/api/workspace/regresion").parsed["workspace"]
    assert entry == {"module": "regresion", "state": None, "payload": None, "revision": 0, "updated_at": None}


def test_state_and_payload_round_trip_and_are_saved_independently(api):
    form = {"form": {"response": "ventas", "predictors": ["precio"]}, "dataset_id": "abc"}
    result = {"meta": {"title": "Regresión"}, "report": {"blocks": []}, "figures": [{"data": [1.5, None]}]}
    first = put(api, "regresion", state=form)
    second = put(api, "regresion", payload=result)  # a partial save keeps the form
    entry = api.get("/api/workspace/regresion").parsed["workspace"]
    assert entry["state"] == form and entry["payload"] == result
    assert 0 < first < second == entry["revision"] and entry["updated_at"]
    put(api, "regresion", state={"form": {}})  # …and the other way round keeps the result
    entry = api.get("/api/workspace/regresion").parsed["workspace"]
    assert entry["state"] == {"form": {}} and entry["payload"] == result


def test_null_clears_a_single_field(api):
    put(api, "anova", state={"x": 1}, payload={"y": 2})
    put(api, "anova", payload=None)
    entry = api.get("/api/workspace/anova").parsed["workspace"]
    assert entry["state"] == {"x": 1} and entry["payload"] is None


def test_clearing_a_module_keeps_a_growing_revision(api):
    before = put(api, "mercados", state={"tab": "draw"}, payload={"a": 1})
    cleared = api.delete("/api/workspace/mercados")
    assert cleared.status_code == 200 and cleared.parsed["revision"] > before
    entry = api.get("/api/workspace/mercados").parsed["workspace"]
    assert entry["state"] is None and entry["payload"] is None and entry["revision"] == cleared.parsed["revision"]
    assert put(api, "mercados", state={"tab": "size"}) > cleared.parsed["revision"]  # never goes backwards


def test_clearing_everything_only_touches_modules_that_had_content(api):
    put(api, "datos", state={"a": 1})
    put(api, "anova", state={"b": 2})
    revisions = api.delete("/api/workspace").parsed["revisions"]
    assert set(revisions) == {"datos", "anova"}
    assert all(api.get(f"/api/workspace/{m}").parsed["workspace"]["state"] is None for m in WORKSPACE_MODULES)


def test_workspace_survives_a_new_session(app, api):
    put(api, "hipotesis", state={"testId": "z_mean"})
    fresh = ApiClient(app.test_client())
    assert fresh.login("admin").status_code == 200  # another browser of the same account
    assert fresh.get("/api/workspace/hipotesis").parsed["workspace"]["state"] == {"testId": "z_mean"}


def test_validation(api):
    assert api.get("/api/workspace/nope").status_code == 422
    assert api.put("/api/workspace/anova", {"state": [1, 2]}).status_code == 422  # must be an object
    assert api.put("/api/workspace/anova", {"payload": "texto"}).status_code == 422
    assert api.delete("/api/workspace/nope").status_code == 422


def test_oversized_workspaces_are_rejected(api):
    big = {"blob": "x" * (17 * 1024 * 1024)}
    response = api.put("/api/workspace/descriptiva", {"payload": big})
    assert response.status_code == 422 and "16 MB" in response.parsed["error"]["message"]
    assert api.get("/api/workspace/descriptiva").parsed["workspace"]["payload"] is None


def test_workspaces_are_private_per_account(api, make_client):
    put(api, "regresion", state={"secreto": True})
    other = make_client("luis")
    assert other.get("/api/workspace/regresion").parsed["workspace"]["state"] is None
    put(other, "regresion", state={"mio": True})
    assert api.get("/api/workspace/regresion").parsed["workspace"]["state"] == {"secreto": True}
    other.delete("/api/workspace")
    assert api.get("/api/workspace/regresion").parsed["workspace"]["state"] == {"secreto": True}


def test_workspace_requires_a_session_and_csrf(app, api):
    guest = ApiClient(app.test_client())  # ``api`` is signed in; this one is not
    assert guest.get("/api/workspace/anova").status_code == 401
    assert guest.get("/api/sync").status_code == 401
    assert api.raw.put("/api/workspace/anova", json={"state": {}}).status_code == 403  # no CSRF header


def test_deleting_an_account_removes_its_workspace(app, api, make_client):
    other = make_client("luis")
    put(other, "anova", state={"x": 1})
    luis = other.get("/api/auth/me").parsed["user"]["id"]
    assert api.delete(f"/api/users/{luis}").status_code == 200
    assert scalar(app.extensions["sigma"].database, "SELECT COUNT(*) FROM workspace WHERE user_id = :u", u=luis) == 0


# ------------------------------------------------------------------ /api/sync
def test_sync_reports_revisions_of_every_scope(api, sales_frame):
    start = api.get("/api/sync").parsed
    assert set(start) == {"datasets", "results", "drafts", "templates", "notebooks", "workspace"}
    assert start["datasets"] == start["results"] == 0 and start["workspace"] == {}

    import_frame(api, sales_frame, "ventas")
    after_import = api.get("/api/sync").parsed
    assert after_import["datasets"] > start["datasets"] and after_import["results"] == start["results"]

    saved = api.post("/api/results", {"module": "anova", "analysis_type": "x", "title": "t", "report": {"a": 1}})
    after_save = api.get("/api/sync").parsed
    assert after_save["results"] > after_import["results"]
    api.delete(f"/api/results/{saved.parsed['result']['id']}")
    assert api.get("/api/sync").parsed["results"] > after_save["results"]  # deletes count as changes too

    revision = put(api, "descriptiva", state={"a": 1})
    assert api.get("/api/sync").parsed["workspace"] == {"descriptiva": revision}


def test_sync_sees_notebook_changes_and_is_private(api, make_client):
    other = make_client("luis", console=True)
    before = api.get("/api/sync").parsed["notebooks"]
    notebook = api.post("/api/notebooks", {"name": "uno"}).parsed["notebook"]
    created = api.get("/api/sync").parsed["notebooks"]
    assert created != before
    api.patch(f"/api/notebooks/{notebook['id']}", {"name": "dos"})
    renamed = api.get("/api/sync").parsed["notebooks"]
    assert renamed != created
    assert other.get("/api/sync").parsed["notebooks"] == before  # another account's notebooks do not move it
    api.delete(f"/api/notebooks/{notebook['id']}")
    assert api.get("/api/sync").parsed["notebooks"] == before


def test_sigma_save_from_a_kernel_bumps_the_dataset_revision(api, sales_frame):
    """The triggers also count writes made from the console, not only from the HTTP API."""
    notebook = api.post("/api/notebooks", {"name": "k"}).parsed["notebook"]
    import_frame(api, sales_frame.head(5), "base")
    before = api.get("/api/sync").parsed["datasets"]
    run = api.post(f"/api/notebooks/{notebook['id']}/execute", {"source": "sigma.save(sigma.load('base'), 'copia')"})
    assert run.parsed["status"] == "ok"
    assert api.get("/api/sync").parsed["datasets"] > before


# ------------------------------------------------------------------ notebook revisions (409)
def test_notebook_save_detects_a_concurrent_edit(api):
    created = api.post("/api/notebooks", {"name": "Demo"}).parsed["notebook"]
    nid, revision = created["id"], created["revision"]
    assert revision
    cells = created["cells"]
    cells[0]["source"] = "x = 1"
    first = api.put(f"/api/notebooks/{nid}", {"cells": cells, "base_revision": revision})
    assert first.status_code == 200 and first.parsed["notebook"]["revision"] != revision
    cells[0]["source"] = "x = 2"
    stale = api.put(f"/api/notebooks/{nid}", {"cells": cells, "base_revision": revision})  # the other window's copy
    assert stale.status_code == 409 and stale.parsed["error"]["code"] == "conflict"
    assert stale.parsed["error"]["details"]["revision"] == first.parsed["notebook"]["revision"]
    assert api.get(f"/api/notebooks/{nid}").parsed["notebook"]["cells"][0]["source"] == "x = 1"
    # sending the revision just received (or none at all, i.e. «overwrite») works
    current = first.parsed["notebook"]["revision"]
    assert api.put(f"/api/notebooks/{nid}", {"cells": cells, "base_revision": current}).status_code == 200
    assert api.put(f"/api/notebooks/{nid}", {"cells": cells}).status_code == 200


def test_notebook_listing_and_get_carry_revisions(api):
    created = api.post("/api/notebooks", {"name": "Demo"}).parsed["notebook"]
    listed = api.get("/api/notebooks").parsed["notebooks"][0]
    fetched = api.get(f"/api/notebooks/{created['id']}").parsed["notebook"]
    assert listed["revision"] == created["revision"] == fetched["revision"]


@pytest.mark.parametrize("module", WORKSPACE_MODULES)
def test_every_workspace_module_is_accepted(api, module):
    assert put(api, module, state={"ok": True}) == 1
    stored = json.loads(api.get(f"/api/workspace/{module}").get_data(as_text=True))["workspace"]["state"]
    assert stored == {"ok": True}
