"""Data transformations through the API: the datasets they leave behind, heavy ones in a worker, hostile input."""

import pytest

from sigma.domain.analysis import registry
from tests.analysis_data import standard_frame
from tests.integration.conftest import import_frame

RUN = "/api/analysis/transformar/run"


@pytest.fixture
def dataset(api):
    return import_frame(api, standard_frame(120), "base")


def run(api, dataset, analysis_id, roles=None, params=None, **extra):
    body = {
        "analysis_id": analysis_id,
        "dataset_id": dataset["id"],
        "roles": roles or {},
        "params": params or {},
        **extra,
    }
    return api.post(RUN, body)


def column_values(api, dataset_id, column, limit=200):
    data = api.get(f"/api/datasets/{dataset_id}/preview?limit={limit}").parsed
    index = data["columns"].index(column)
    return [row[index] for row in data["rows"]]


def test_the_catalogue_lists_every_transformation_with_its_help(api):
    analyses = {a["id"]: a for a in api.get("/api/analysis/transformar/catalog").parsed["analyses"]}
    assert len(analyses) == len(registry.catalog("transformar")) >= 20
    assert all(a["has_code"] for a in analyses.values())
    assert {"term", "text"} <= set(analyses["compute"]["reference"][0])
    assert analyses["compute"]["params"][1]["kind"] == "code"
    assert all(a["outputs"] for a in analyses.values())
    groups = {a["group"] for a in analyses.values()}
    assert {"Crear y modificar variables", "Filas", "Estructura", "Calidad de los datos"} <= groups


def test_a_new_column_is_saved_in_a_new_dataset(api, dataset):
    response = run(
        api, dataset, "compute", params={"name": "doble", "expression": "x1 * 2"}, save_as={"resultado": "con_doble"}
    )
    assert response.status_code == 200, response.parsed
    created = response.parsed["datasets"][0]
    assert created["name"] == "con_doble" and created["n_cols"] == dataset["n_cols"] + 1 and created["n_rows"] == 120
    originals, doubles = column_values(api, dataset["id"], "x1"), column_values(api, created["id"], "doble")
    assert doubles == pytest.approx([2 * v for v in originals])
    assert api.get(f"/api/datasets/{dataset['id']}").parsed["dataset"]["n_cols"] == dataset["n_cols"]  # untouched


def test_a_transformation_can_replace_the_original_dataset(api, dataset):
    response = run(api, dataset, "standardize", {"cols": ["x1"]}, {"method": "zscore", "mode": "replace"}, replace=True)
    assert response.status_code == 200
    assert response.parsed["datasets"][0]["id"] == dataset["id"]
    values = column_values(api, dataset["id"], "x1")
    assert sum(values) == pytest.approx(0, abs=1e-9)


def test_a_name_is_required_unless_replacing(api, dataset):
    assert run(api, dataset, "compute", params={"name": "n", "expression": "x1"}).status_code == 422
    assert run(api, dataset, "filter", params={"condition": "x1 > 1"}, save_as={"resultado": " "}).status_code == 422


def test_a_partition_leaves_every_set_as_a_dataset(api, dataset):
    names = {"train": "entrenamiento", "test": "prueba", "validation": "validacion"}
    response = run(
        api, dataset, "partition", params={"test_fraction": 0.25, "validation_fraction": 0.15}, save_as=names
    )
    assert response.status_code == 200, response.parsed
    sizes = {d["name"]: d["n_rows"] for d in response.parsed["datasets"]}
    assert sizes == {"entrenamiento": 72, "prueba": 30, "validacion": 18}
    assert run(api, dataset, "partition", save_as={"train": "solo"}).status_code == 422  # both main sets need a name
    refused = run(api, dataset, "partition", save_as={"train": "a", "test": "b", "validation": "c"})
    assert refused.status_code == 422  # no validation fraction → nothing to save under that name


def test_a_heavy_imputation_runs_in_its_own_process(api, dataset):
    response = run(
        api,
        dataset,
        "impute_model",
        {"cols": ["m"], "predictors": ["x1", "x3"]},
        {"method": "mice"},
        save_as={"resultado": "imputado"},
    )
    assert response.status_code == 200, response.parsed
    created = response.parsed["datasets"][0]
    assert all(v is not None for v in column_values(api, created["id"], "m"))
    assert any(v is None for v in column_values(api, dataset["id"], "m"))


def test_hostile_formulas_come_back_as_clear_errors(api, dataset):
    for expression in ("__import__('os').getcwd()", "x1.__class__", "[1 for _ in x1]", "open('/etc/passwd')", "x1 +"):
        response = run(
            api, dataset, "compute", params={"name": "malo", "expression": expression}, save_as={"resultado": "x"}
        )
        assert response.status_code == 422, expression
        assert response.parsed["error"]["message"]
    assert [d["name"] for d in api.get("/api/datasets").parsed["datasets"]] == ["base"]


def test_results_of_a_transformation_can_be_saved_and_reopened(api, dataset):
    data = run(
        api,
        dataset,
        "bin",
        {"col": "x1"},
        {"name": "nivel", "method": "quantile", "bins": 3},
        save_as={"resultado": "b"},
    ).parsed
    saved = api.post(
        "/api/results",
        {
            "module": "transformar",
            "analysis_type": data["meta"]["analysis_type"],
            "title": data["meta"]["title"],
            "params": data["meta"]["params"],
            "result": data["result"],
            "report": data["report"],
            "figures": data["figures"],
        },
    )
    assert saved.status_code == 201
    reopened = api.get(f"/api/results/{saved.parsed['result']['id']}").parsed["result"]
    assert reopened["params"]["analysis_id"] == "bin" and reopened["params"]["params"]["bins"] == 3


def test_text_and_boolean_columns_survive_storage(api, dataset):
    created = run(
        api,
        dataset,
        "compute",
        params={"name": "etiqueta", "expression": 'concat(g, "_", h)'},
        save_as={"resultado": "texto"},
    ).parsed["datasets"][0]
    assert column_values(api, created["id"], "etiqueta")[0].count("_") == 1
    flags = run(
        api,
        {"id": created["id"]},
        "compute",
        params={"name": "alto", "expression": "x1 > 10"},
        save_as={"resultado": "banderas"},
    ).parsed["datasets"][0]
    assert set(column_values(api, flags["id"], "alto")) <= {True, False}
    info = api.get(f"/api/datasets/{flags['id']}").parsed["dataset"]
    assert {c["name"]: c["kind"] for c in info["columns"]}["alto"] == "boolean"


def test_other_accounts_cannot_transform_my_data(api, make_client, dataset):
    other = make_client("luis")
    refused = other.post(
        RUN,
        {"analysis_id": "sort", "dataset_id": dataset["id"], "roles": {"cols": ["x1"]}, "params": {}, "replace": True},
    )
    assert refused.status_code == 404


def test_a_second_dataset_must_belong_to_the_same_account(api, make_client, dataset):
    from tests.analysis_data import lookup_frame

    mine = import_frame(api, lookup_frame(), "referencia")
    body = {
        "analysis_id": "merge",
        "dataset_id": dataset["id"],
        "roles": {"keys": ["g"], "other_keys": ["g"]},
        "params": {"other": mine["id"], "how": "left"},
        "save_as": {"resultado": "unido"},
    }
    ok = api.post(RUN, body)
    assert ok.status_code == 200, ok.parsed
    assert ok.parsed["datasets"][0]["n_rows"] == 120 and "peso" in column_names(api, ok.parsed["datasets"][0]["id"])
    other = make_client("luis")
    theirs = import_frame(other, standard_frame(30), "ajeno")
    stolen = api.post(RUN, {**body, "params": {"other": theirs["id"]}, "save_as": {"resultado": "robado"}})
    assert stolen.status_code == 404
    assert "robado" not in [d["name"] for d in api.get("/api/datasets").parsed["datasets"]]
    assert api.post(RUN, {**body, "params": {"how": "left"}}).status_code == 422  # the second dataset is required
    code = api.post(
        "/api/analysis/transformar/code", {k: body[k] for k in ("analysis_id", "dataset_id", "roles", "params")}
    )
    assert code.status_code == 200 and "sigma.load('referencia')" in code.parsed["code"]


def column_names(api, dataset_id):
    return [c["name"] for c in api.get(f"/api/datasets/{dataset_id}").parsed["dataset"]["columns"]]
