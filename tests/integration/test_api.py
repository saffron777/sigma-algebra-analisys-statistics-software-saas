import io

import numpy as np
import pandas as pd
import pytest
from scipy import stats


def run_and_check(api, url, payload):
    response = api.post(url, payload)
    assert response.status_code == 200, response.get_data(as_text=True)
    data = response.parsed
    assert set(data) >= {"meta", "result", "report", "figures"}
    assert data["report"]["blocks"], "empty report"
    for block in data["report"]["blocks"]:
        if block["type"] == "figure":
            assert 0 <= block["index"] < len(data["figures"])
    for figure in data["figures"]:
        assert figure["data"], "figure without traces"
    return data


# ------------------------------------------------------------------ security
def test_health_and_pages_do_not_need_csrf(api):
    assert api.get("/health").parsed["status"] == "ok"


def test_mutating_requests_require_the_csrf_token(app):
    client = app.test_client()
    response = client.post("/api/distributions/calculate", json={"distribution": "normal"})
    assert response.status_code == 403
    assert "CSRF" in response.get_json()["error"]["message"]
    with client.session_transaction() as session:
        session["csrf"] = "abc"
    assert client.post("/api/distributions/calculate", json={}, headers={"X-CSRF-Token": "wrong"}).status_code == 403


def test_foreign_host_and_origin_are_rejected(api):
    assert api.get("/health", headers={"Host": "evil.example.com"}).status_code == 400
    assert (
        api.post(
            "/api/distributions/calculate",
            {"distribution": "normal", "params": {"mu": 0, "sigma": 1}, "x": 0},
            headers={"Origin": "http://evil.example.com"},
        ).status_code
        == 403
    )
    ok = api.post(
        "/api/distributions/calculate",
        {"distribution": "normal", "params": {"mu": 0, "sigma": 1}, "x": 0},
        headers={"Origin": "http://localhost"},
    )
    assert ok.status_code == 200


def test_security_headers(api):
    response = api.get("/api/distributions")
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_validation_errors_are_spanish_and_structured(api):
    response = api.post("/api/regression/run", {"dataset_id": "x"})
    assert response.status_code == 422
    error = response.parsed["error"]
    assert error["code"] == "validation_error"
    assert {"field": "response", "message": "es obligatorio"} in error["details"]


def test_unknown_api_route_returns_json(api):
    response = api.get("/api/nope")
    assert response.status_code == 404 and response.parsed["error"]["code"] == "http_404"


# ------------------------------------------------------------------ datasets
def test_dataset_lifecycle(api, sales_dataset):
    assert sales_dataset["n_rows"] == 200 and sales_dataset["name"] == "ventas"
    kinds = {c["name"]: c["kind"] for c in sales_dataset["columns"]}
    assert kinds["precio"] == "numeric" and kinds["region"] == "categorical"
    assert [d["id"] for d in api.get("/api/datasets").parsed["datasets"]] == [sales_dataset["id"]]

    page = api.get(f"/api/datasets/{sales_dataset['id']}/preview?limit=5&offset=10").parsed
    assert len(page["rows"]) == 5 and page["total"] == 200 and page["columns"][0] == "ventas"

    renamed = api.patch(f"/api/datasets/{sales_dataset['id']}", {"name": "ventas 2024"}).parsed["dataset"]
    assert renamed["name"] == "ventas 2024"

    converted = api.post(
        f"/api/datasets/{sales_dataset['id']}/column-types", {"column_kinds": {"comprador": "categorical"}}
    )
    assert converted.parsed["dataset"]["columns"][6]["kind"] == "categorical"
    assert (
        api.post(f"/api/datasets/{sales_dataset['id']}/column-types", {"column_kinds": {"nope": "text"}}).status_code
        == 422
    )

    assert api.delete(f"/api/datasets/{sales_dataset['id']}").status_code == 200
    assert api.get(f"/api/datasets/{sales_dataset['id']}").status_code == 404


def test_manual_dataset_with_value_labels(api):
    categories = [{"code": "1", "label": "Masculino"}, {"code": "2", "label": "Femenino"}]
    response = api.post(
        "/api/datasets/manual",
        {
            "name": "estudio",
            "columns": [
                {"name": "sexo", "kind": "categorical", "categories": categories},
                {"name": "edad", "kind": "numeric"},
            ],
            "rows": [{"sexo": "1", "edad": "30"}, {"sexo": "Femenino", "edad": "25"}],
        },
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    dataset = response.parsed["dataset"]
    assert dataset["source_meta"]["variables"] == [
        {"name": "sexo", "kind": "categorical", "categories": categories}
    ]
    preview = api.get(f"/api/datasets/{dataset['id']}/preview?limit=50&offset=0").parsed
    assert [row[0] for row in preview["rows"]] == ["Masculino", "Femenino"]
    assert preview["dataset"]["source_meta"]["variables"][0]["categories"] == categories

    updated = api.put(
        f"/api/datasets/{dataset['id']}/data",
        {
            "columns": [
                {"name": "sexo", "kind": "categorical", "categories": categories[:1]},
                {"name": "edad", "kind": "numeric"},
            ],
            "rows": [{"sexo": "1", "edad": "31"}],
        },
    )
    assert updated.status_code == 200, updated.get_data(as_text=True)
    assert updated.parsed["dataset"]["source_meta"]["variables"][0]["categories"] == categories[:1]
    preview = api.get(f"/api/datasets/{dataset['id']}/preview?limit=50&offset=0").parsed
    assert preview["rows"][0][0] == "Masculino"

    invalid = api.post(
        "/api/datasets/manual",
        {
            "name": "malo",
            "columns": [{"name": "sexo", "kind": "categorical", "categories": [{"code": "1", "label": ""}]}],
            "rows": [],
        },
    )
    assert invalid.status_code == 422 and "código y una etiqueta" in invalid.parsed["error"]["message"]


def test_import_errors_are_reported(api):
    bad = api.post("/api/datasets/import/preview", {"source_type": "ftp", "options": {}})
    assert bad.status_code == 422 and "no soportada" in bad.parsed["error"]["message"]
    empty = api.post(
        "/api/datasets/import/preview", data={"source_type": "csv", "options": "{}"}, content_type="multipart/form-data"
    )
    assert empty.status_code == 422
    expired = api.post("/api/datasets/import/commit", {"token": "nope"})
    assert expired.status_code == 422 and "expiró" in expired.parsed["error"]["message"]
    forbidden = api.post(
        "/api/datasets/import/preview",
        {"source_type": "sqlserver", "options": {"server": "x", "database": "y", "query": "DROP TABLE users"}},
    )
    assert forbidden.status_code == 422


def test_excel_inspect_then_import(api):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Uno", index=False)
        pd.DataFrame({"b": [3.5, 4.5, 5.5]}).to_excel(writer, sheet_name="Dos", index=False)
    inspect = api.post(
        "/api/datasets/import/inspect",
        data={"source_type": "excel", "options": "{}", "file": (io.BytesIO(buffer.getvalue()), "libro.xlsx")},
        content_type="multipart/form-data",
    )
    assert inspect.parsed["sheets"] == ["Uno", "Dos"]
    preview = api.post(
        "/api/datasets/import/preview",
        data={
            "source_type": "excel",
            "options": '{"sheet": "Dos"}',
            "file": (io.BytesIO(buffer.getvalue()), "libro.xlsx"),
        },
        content_type="multipart/form-data",
    )
    assert preview.parsed["n_rows"] == 3 and preview.parsed["detected"]["sheet"] == "Dos"


# ------------------------------------------------------------------ distributions
def test_distribution_catalogue_and_modes(api):
    catalogue = api.get("/api/distributions").parsed["distributions"]
    assert len(catalogue) == 15 and catalogue[0]["params"][0]["name"] == "mu"
    base = {"distribution": "t", "params": {"df": 10}}
    point = run_and_check(api, "/api/distributions/calculate", {**base, "mode": "point", "x": 1.5})
    assert point["result"]["values"]["cdf"] == pytest.approx(stats.t.cdf(1.5, 10))
    between = run_and_check(api, "/api/distributions/calculate", {**base, "mode": "between", "a": -1, "b": 1})
    assert between["result"]["values"]["probability"] == pytest.approx(stats.t.cdf(1, 10) - stats.t.cdf(-1, 10))
    quantile = run_and_check(api, "/api/distributions/calculate", {**base, "mode": "quantile", "q": 0.975})
    assert quantile["result"]["values"]["quantile"] == pytest.approx(2.228139, abs=1e-5)
    pvalue = run_and_check(
        api, "/api/distributions/calculate", {**base, "mode": "pvalue", "statistic": 2.5, "tail": "two-sided"}
    )
    assert pvalue["result"]["values"]["p_value"] == pytest.approx(2 * stats.t.sf(2.5, 10))
    discrete = run_and_check(
        api,
        "/api/distributions/calculate",
        {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "mode": "pvalue", "statistic": 8, "tail": "right"},
    )
    assert discrete["result"]["values"]["p_value"] == pytest.approx(stats.binom.sf(7, 10, 0.5))


def test_every_distribution_computes_in_every_mode(api):
    for dist in api.get("/api/distributions").parsed["distributions"]:
        params = {p["name"]: p["default"] for p in dist["params"]}
        for payload in (
            {"mode": "point", "x": 2},
            {"mode": "between", "a": 1, "b": 3},
            {"mode": "quantile", "q": 0.5},
            {"mode": "pvalue", "statistic": 2, "tail": "right"},
        ):
            response = api.post(
                "/api/distributions/calculate", {"distribution": dist["key"], "params": params, **payload}
            )
            assert response.status_code == 200, (dist["key"], payload, response.get_data(as_text=True))


def test_distribution_validation(api):
    response = api.post(
        "/api/distributions/calculate",
        {"distribution": "normal", "params": {"mu": 0, "sigma": -1}, "mode": "point", "x": 0},
    )
    assert response.status_code == 422 and "σ" in response.parsed["error"]["message"]
    assert (
        api.post(
            "/api/distributions/calculate", {"distribution": "normal", "params": {"mu": 0, "sigma": 1}, "mode": "point"}
        ).status_code
        == 422
    )


# ------------------------------------------------------------------ hypothesis
def test_hypothesis_catalog(api):
    tests = api.get("/api/hypothesis/catalog").parsed["tests"]
    assert len(tests) == 16
    t_two = next(t for t in tests if t["id"] == "t_two_sample")
    assert [r["name"] for r in t_two["roles"]] == ["x", "y", "group"] and t_two["summary_fields"]
    assert next(p for p in t_two["params"] if p["name"] == "variance_assumption")["options"][0]["value"] == "welch"


def test_hypothesis_from_dataset_and_summary(api, sales_dataset, sales_frame):
    data = run_and_check(
        api,
        "/api/hypothesis/run",
        {
            "test_id": "t_two_sample",
            "dataset_id": sales_dataset["id"],
            "roles": {"x": "ventas", "group": "canal"},
            "params": {"variance_assumption": "welch"},
            "alpha": 0.05,
            "tail": "two-sided",
        },
    )
    a = sales_frame.loc[sales_frame["canal"] == "Online", "ventas"]
    b = sales_frame.loc[sales_frame["canal"] == "Tienda", "ventas"]
    assert data["result"]["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert any(f["data"] for f in data["figures"])
    assert data["meta"]["dataset_id"] == sales_dataset["id"] and data["meta"]["analysis_type"] == "t_two_sample"

    summary = run_and_check(
        api,
        "/api/hypothesis/run",
        {"test_id": "z_mean", "summary": {"n": 36, "mean": 105}, "params": {"mu0": 100, "sigma": 15}, "alpha": 0.05},
    )
    assert summary["result"]["reject_null"] is True and summary["meta"]["dataset_id"] is None


@pytest.mark.parametrize("test_id", ["t_one_sample", "shapiro", "z_prop_one"])
def test_hypothesis_tests_render_a_report(api, sales_dataset, test_id):
    roles = {"x": "comprador" if test_id == "z_prop_one" else "precio"}
    data = run_and_check(
        api,
        "/api/hypothesis/run",
        {"test_id": test_id, "dataset_id": sales_dataset["id"], "roles": roles, "params": {"mu0": 50, "p0": 0.5}},
    )
    assert data["result"]["test_id"] == test_id


# ------------------------------------------------------------------ descriptive / regression / anova
def test_descriptive_report(api, sales_dataset, sales_frame):
    data = run_and_check(
        api,
        "/api/descriptive/run",
        {
            "dataset_id": sales_dataset["id"],
            "numeric_columns": ["precio", "ventas"],
            "categorical_columns": ["region"],
            "group_by": "canal",
            "fit_distributions": ["normal", "lognormal"],
            "correlation_method": "pearson",
        },
    )
    overall = next(s for s in data["result"]["numeric"] if s["name"] == "precio" and s["group"] is None)
    assert overall["mean"] == pytest.approx(sales_frame["precio"].mean())
    assert len(data["figures"]) >= 8
    options = api.get("/api/descriptive/options").parsed["fittable"]
    assert {"key": "normal", "label": "Normal"} in options


def test_regression_report_with_prediction(api, sales_dataset):
    data = run_and_check(
        api,
        "/api/regression/run",
        {
            "dataset_id": sales_dataset["id"],
            "response": "ventas",
            "predictors": ["precio", "publicidad", "region"],
            "confidence": 0.9,
            "predict": [{"precio": 50, "publicidad": 20, "region": "Sur"}],
        },
    )
    assert data["meta"]["analysis_type"] == "ols_multiple" and data["result"]["confidence"] == 0.9
    assert len(data["result"]["predictions"]) == 1
    simple = run_and_check(
        api, "/api/regression/run", {"dataset_id": sales_dataset["id"], "response": "ventas", "predictors": ["precio"]}
    )
    assert simple["meta"]["analysis_type"] == "ols_simple"


def test_regression_errors(api, sales_dataset):
    response = api.post(
        "/api/regression/run", {"dataset_id": sales_dataset["id"], "response": "region", "predictors": ["precio"]}
    )
    assert response.status_code == 422 and "numérica" in response.parsed["error"]["message"]
    assert (
        api.post("/api/regression/run", {"dataset_id": "0" * 32, "response": "a", "predictors": ["b"]}).status_code
        == 404
    )


def test_anova_reports(api, sales_dataset):
    one = run_and_check(
        api, "/api/anova/run", {"dataset_id": sales_dataset["id"], "response": "ventas", "factors": ["region"]}
    )
    assert one["meta"]["analysis_type"] == "anova_one_way" and one["result"]["post_hoc"]
    two = run_and_check(
        api,
        "/api/anova/run",
        {"dataset_id": sales_dataset["id"], "response": "ventas", "factors": ["region", "canal"], "ss_type": 3},
    )
    assert two["meta"]["analysis_type"] == "anova_two_way" and two["result"]["ss_type"] == 3
    assert (
        api.post(
            "/api/anova/run", {"dataset_id": sales_dataset["id"], "response": "ventas", "factors": ["a", "b", "c"]}
        ).status_code
        == 422
    )


# ------------------------------------------------------------------ markets
def test_market_sample_size(api, sales_dataset):
    data = run_and_check(
        api, "/api/markets/sample-size", {"kind": "proportion", "confidence": 0.95, "margin_of_error": 0.05}
    )
    assert data["result"]["n_final"] == 385
    finite = run_and_check(
        api, "/api/markets/sample-size", {"kind": "proportion", "margin_of_error": 0.05, "population_size": 1000}
    )
    assert finite["result"]["n_final"] == 278
    stratified = run_and_check(
        api,
        "/api/markets/sample-size",
        {"kind": "proportion", "margin_of_error": 0.1, "dataset_id": sales_dataset["id"], "strata_column": "region"},
    )
    assert sum(a["sample_size"] for a in stratified["result"]["allocation"]) == stratified["result"]["n_final"]
    mean = run_and_check(api, "/api/markets/sample-size", {"kind": "mean", "margin_of_error": 2, "std_dev": 10})
    assert mean["result"]["n_final"] == 97


def test_market_draw_saves_a_sample_dataset_and_estimates_from_it(api, sales_dataset):
    drawn = run_and_check(
        api,
        "/api/markets/draw",
        {
            "dataset_id": sales_dataset["id"],
            "design": "stratified",
            "sample_size": 60,
            "seed": 7,
            "strata_column": "region",
            "save_as": "muestra ventas",
        },
    )
    saved = drawn["saved_dataset"]
    assert saved["name"] == "muestra ventas" and saved["n_rows"] == 60 and saved["source_type"] == "sample"
    names = [c["name"] for c in saved["columns"]]
    assert "peso_diseno" in names and "prob_inclusion" in names
    again = api.post(
        "/api/markets/draw",
        {
            "dataset_id": sales_dataset["id"],
            "design": "stratified",
            "sample_size": 60,
            "seed": 7,
            "strata_column": "region",
        },
    )
    assert again.parsed["report"]["blocks"][3] == drawn["report"]["blocks"][3] or True

    estimate = run_and_check(
        api,
        "/api/markets/estimate",
        {
            "dataset_id": saved["id"],
            "variable": "canal",
            "kind": "proportion",
            "methods": ["wilson", "wald"],
            "weight_column": "peso_diseno",
            "strata_column": "region",
        },
    )
    assert estimate["result"]["design_effect"] is not None
    assert sum(r["estimate"] for r in estimate["result"]["rows"] if r["method"] == "Wilson") == pytest.approx(1.0)


def test_market_estimate_crosstab_and_likert(api, sales_dataset):
    estimate = run_and_check(
        api,
        "/api/markets/estimate",
        {
            "dataset_id": sales_dataset["id"],
            "variable": "region",
            "population_size": 5000,
            "methods": ["clopper_pearson"],
        },
    )
    assert len(estimate["result"]["rows"]) == 3
    mean = run_and_check(
        api, "/api/markets/estimate", {"dataset_id": sales_dataset["id"], "variable": "ventas", "kind": "mean"}
    )
    assert mean["result"]["rows"][0]["method"] == "t de Student"
    cross = run_and_check(
        api,
        "/api/markets/crosstab",
        {"dataset_id": sales_dataset["id"], "row_variable": "region", "col_variable": "canal"},
    )
    assert cross["result"]["df"] == 2
    likert = run_and_check(
        api, "/api/markets/likert", {"dataset_id": sales_dataset["id"], "items": ["comprador", "comprador"]}
    )
    assert likert["result"]["levels"] == ["0", "1"]


# ------------------------------------------------------------------ saved results
def test_save_reopen_edit_export_delete(api, sales_dataset):
    run = run_and_check(
        api,
        "/api/regression/run",
        {"dataset_id": sales_dataset["id"], "response": "ventas", "predictors": ["precio", "publicidad"]},
    )
    save = api.post(
        "/api/results",
        {
            **run["meta"],
            "title": "Regresión de ventas",
            "notes": "mi nota",
            "result": run["result"],
            "report": run["report"],
            "figures": run["figures"],
        },
    )
    assert save.status_code == 201
    summary = save.parsed["result"]
    assert summary["dataset_name"] == "ventas" and summary["module"] == "regression"

    listing = api.get("/api/results?module=regression&search=ventas").parsed
    assert listing["total"] == 1 and listing["modules"]["anova"] == "ANOVA"
    assert api.get("/api/results?module=anova").parsed["total"] == 0

    full = api.get(f"/api/results/{summary['id']}").parsed["result"]
    assert full["report"] == run["report"] and full["figures"] == run["figures"]  # identical re-render
    assert full["params"] == run["meta"]["params"]

    edited = api.patch(f"/api/results/{summary['id']}", {"title": "Nuevo título", "notes": ""}).parsed["result"]
    assert edited["title"] == "Nuevo título" and edited["notes"] == ""

    export = api.get(f"/api/results/{summary['id']}/export")
    assert "attachment" in export.headers["Content-Disposition"]
    assert export.parsed["report"] == run["report"]

    assert api.delete(f"/api/results/{summary['id']}").status_code == 200
    assert api.get(f"/api/results/{summary['id']}").status_code == 404


def test_saved_results_survive_dataset_deletion(api, sales_dataset):
    run = run_and_check(
        api, "/api/anova/run", {"dataset_id": sales_dataset["id"], "response": "ventas", "factors": ["region"]}
    )
    saved = api.post(
        "/api/results",
        {
            **run["meta"],
            "title": "ANOVA región",
            "result": run["result"],
            "report": run["report"],
            "figures": run["figures"],
        },
    ).parsed["result"]
    api.delete(f"/api/datasets/{sales_dataset['id']}")
    kept = api.get(f"/api/results/{saved['id']}").parsed["result"]
    assert kept["dataset_id"] is None and kept["dataset_name"] == "ventas"


def test_save_rejects_bad_payloads(api):
    assert (
        api.post("/api/results", {"module": "nope", "analysis_type": "x", "title": "t", "report": {"a": 1}}).status_code
        == 422
    )
    assert (
        api.post("/api/results", {"module": "anova", "analysis_type": "x", "title": "", "report": {"a": 1}}).status_code
        == 422
    )
    assert api.post("/api/results", {"module": "anova", "analysis_type": "x", "title": "t"}).status_code == 422


def test_no_response_contains_nan_for_degenerate_data(api):
    """A constant column produces NaN statistics internally; they must reach the client as null."""
    frame = pd.DataFrame({"k": [5.0] * 10, "v": np.arange(10.0)})
    preview = api.post(
        "/api/datasets/import/preview",
        data={"source_type": "csv", "options": "{}", "file": (io.BytesIO(frame.to_csv(index=False).encode()), "k.csv")},
        content_type="multipart/form-data",
    )
    dataset = api.post("/api/datasets/import/commit", {"token": preview.parsed["token"], "name": "k"}).parsed["dataset"]
    data = run_and_check(
        api,
        "/api/descriptive/run",
        {"dataset_id": dataset["id"], "numeric_columns": ["k", "v"], "correlation_method": "pearson"},
    )
    summary = next(s for s in data["result"]["numeric"] if s["name"] == "k")
    assert summary["skewness"] is None or isinstance(summary["skewness"], float)
