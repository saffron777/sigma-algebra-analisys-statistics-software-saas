"""The analysis catalogue through the API: listing, running, datasets left behind, saved results and the code view."""

import pytest

import tests.analysis_fixtures  # noqa: F401 - the internal test analyses (hidden from the public catalogue)
from sigma.domain.analysis.modules import CATALOG_MODULES
from tests.analysis_data import standard_frame
from tests.integration.conftest import import_frame

POWER = {
    "analysis_id": "t_power",
    "roles": {},
    "params": {"design": "two_sample", "solve_for": "n", "effect_size": 0.5, "power": 0.8},
}


@pytest.fixture
def series_dataset(api):
    return import_frame(api, standard_frame(120), "serie")


def check_payload(data):
    assert set(data) >= {"meta", "result", "report", "figures"} and data["report"]["blocks"]
    for block in data["report"]["blocks"]:
        if block["type"] == "figure":
            assert 0 <= block["index"] < len(data["figures"])
    for figure in data["figures"]:
        assert figure["data"]


@pytest.mark.parametrize("module", [m.key for m in CATALOG_MODULES])
def test_every_module_lists_its_catalogue(api, module):
    response = api.get(f"/api/analysis/{module}/catalog")
    assert response.status_code == 200
    body = response.parsed
    assert body["module"]["key"] == module and body["module"]["label"]
    for analysis in body["analyses"]:
        assert not analysis["id"].startswith("_")  # internal test analyses stay hidden
        assert {"roles", "params", "outputs", "needs_dataset", "group"} <= set(analysis)


def test_an_unknown_module_is_refused(api):
    assert api.get("/api/analysis/inventado/catalog").status_code == 422


def test_an_analysis_without_data_runs_and_reports(api):
    data = api.post("/api/analysis/potencia/run", POWER).parsed
    check_payload(data)
    assert data["meta"]["module"] == "potencia" and data["meta"]["analysis_type"] == "t_power"
    assert data["result"]["n"] == 64 and data["meta"]["dataset_id"] is None
    assert data["datasets"] == []


def test_an_analysis_on_a_dataset_checks_the_columns(api, series_dataset):
    body = {
        "analysis_id": "acf_pacf",
        "dataset_id": series_dataset["id"],
        "roles": {"series": "s", "time": "t"},
        "params": {"lags": 10},
    }
    data = api.post("/api/analysis/series/run", body).parsed
    check_payload(data)
    assert data["meta"]["dataset_id"] == series_dataset["id"] and "serie" in data["report"]["subtitle"]
    assert len(data["figures"]) == 3
    for bad_roles in ({"series": "g"}, {"series": "nada"}, {}):
        refused = api.post("/api/analysis/series/run", {**body, "roles": bad_roles})
        assert refused.status_code == 422
    assert api.post("/api/analysis/series/run", {**body, "dataset_id": None}).status_code == 422
    assert api.post("/api/analysis/series/run", {**body, "params": {"lags": 0}}).status_code == 422


def test_parameters_are_validated_with_spanish_messages(api):
    refused = api.post("/api/analysis/potencia/run", {**POWER, "params": {**POWER["params"], "alpha": 7}})
    assert refused.status_code == 422 and "α" in refused.parsed["error"]["message"]
    missing = api.post("/api/analysis/potencia/run", {"analysis_id": "inexistente", "params": {}})
    assert missing.status_code == 422


def test_the_result_can_be_saved_and_reopened_with_its_form(api):
    data = api.post("/api/analysis/potencia/run", POWER).parsed
    saved = api.post(
        "/api/results",
        {
            "module": data["meta"]["module"],
            "analysis_type": data["meta"]["analysis_type"],
            "title": data["meta"]["title"],
            "params": data["meta"]["params"],
            "result": data["result"],
            "report": data["report"],
            "figures": data["figures"],
        },
    )
    assert saved.status_code == 201
    listing = api.get("/api/results").parsed
    assert (
        listing["modules"]["potencia"] == "Potencia y tamaño de muestra" and listing["pages"]["potencia"] == "/potencia"
    )
    reopened = api.get(f"/api/results/{saved.parsed['result']['id']}").parsed["result"]
    assert reopened["params"]["analysis_id"] == "t_power" and reopened["params"]["params"]["effect_size"] == 0.5


def test_ai_interpretation_knows_the_new_modules(api):
    from sigma.domain.interpretation import MODULES, build_prompt

    assert all(m.key in MODULES for m in CATALOG_MODULES)
    prompt = build_prompt("series", "texto del informe", "Título")
    assert "estacionariedad" in prompt.user


def test_the_python_that_reproduces_an_analysis_is_offered(api, series_dataset):
    body = {
        "analysis_id": "acf_pacf",
        "dataset_id": series_dataset["id"],
        "roles": {"series": "s", "time": "t"},
        "params": {"lags": 10},
    }
    code = api.post("/api/analysis/series/code", body).parsed["code"]
    assert 'sigma.load("serie")' in code and "acorr_ljungbox" in code and "nlags=10" in code
    compile(code.replace('sigma.load("serie")', "None"), "<codigo>", "exec")  # it is valid Python
    assert "TTestIndPower" in api.post("/api/analysis/potencia/code", POWER).parsed["code"]


def test_datasets_left_behind_by_an_analysis_are_saved(api, series_dataset):
    body = {
        "analysis_id": "_copy",
        "dataset_id": series_dataset["id"],
        "roles": {"col": "x1"},
        "params": {"factor": 3},
        "save_as": {"resultado": "triple"},
    }
    data = api.post("/api/analysis/transformar/run", body).parsed
    assert [d["name"] for d in data["datasets"]] == ["triple"] and data["datasets"][0]["n_rows"] == 120
    names = [d["name"] for d in api.get("/api/datasets").parsed["datasets"]]
    assert "triple" in names
    # a required output needs a name; unknown outputs and replacing without permission are refused
    assert api.post("/api/analysis/transformar/run", {**body, "save_as": {}}).status_code == 422
    assert api.post("/api/analysis/transformar/run", {**body, "save_as": {"otro": "x"}}).status_code == 422
    assert api.post("/api/analysis/potencia/run", {**POWER, "replace": True}).status_code == 422


def test_a_transformation_can_replace_the_original(api, series_dataset):
    body = {
        "analysis_id": "_copy",
        "dataset_id": series_dataset["id"],
        "roles": {"col": "x1"},
        "params": {"factor": 2},
        "replace": True,
    }
    data = api.post("/api/analysis/transformar/run", body).parsed
    assert data["datasets"][0]["id"] == series_dataset["id"] and data["datasets"][0]["n_cols"] == 2


def test_other_accounts_cannot_use_my_datasets(api, make_client, series_dataset):
    other = make_client("luis")
    refused = other.post(
        "/api/analysis/series/run",
        {"analysis_id": "acf_pacf", "dataset_id": series_dataset["id"], "roles": {"series": "s"}, "params": {}},
    )
    assert refused.status_code == 404


def test_the_new_pages_are_in_the_menu_and_the_home(api):
    html = api.raw.get("/").get_data(as_text=True)
    for module in CATALOG_MODULES:
        assert f'href="{module.path}"' in html
    assert api.raw.get("/series").status_code == 200 and "catalogoPage('series')" in api.raw.get("/series").get_data(
        as_text=True
    )
