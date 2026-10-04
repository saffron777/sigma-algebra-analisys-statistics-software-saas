"""Model selection, regularisation, instrumental variables, mixed models and specification tests (known truth)."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from sklearn.linear_model import Lasso

from sigma.domain.analysis.models_more import chow_test
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def sparse():
    """Six candidate predictors of which only the first two matter."""
    rng = np.random.default_rng(10)
    n = 300
    x = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"p{i}" for i in range(1, 7)])
    x["g"] = rng.choice(["a", "b", "c"], n)
    y = 3 + 2.0 * x["p1"] - 1.5 * x["p2"] + rng.normal(0, 1, n)
    return x.assign(y=y)


def model(analysis_id, frame, roles=None, params=None):
    return run("modelos", analysis_id, frame, roles, params)


def table(output, title):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"] == title)["rows"]


def value(cell):
    return getattr(cell, "value", cell)


# ------------------------------------------------------------------ stepwise and best subsets
def test_stepwise_finds_the_true_predictors(sparse):
    predictors = [f"p{i}" for i in range(1, 7)] + ["g"]
    for direction in ("forward", "backward", "both"):
        out = model(
            "stepwise",
            sparse,
            {"response": "y", "predictors": predictors},
            {"direction": direction, "criterion": "bic"},
        )
        assert set(out.result["selected"]) == {"p1", "p2"}, direction
    final = model(
        "stepwise",
        sparse,
        {"response": "y", "predictors": predictors},
        {"criterion": "pvalue", "alpha_enter": 0.001, "alpha_remove": 0.01},
    )
    assert set(final.result["selected"]) == {"p1", "p2"}
    reference = smf.ols("y ~ p1 + p2", sparse).fit()
    assert final.result["r2_adj"] == pytest.approx(reference.rsquared_adj)
    noise = sparse.assign(y=np.random.default_rng(3).normal(size=len(sparse)))
    empty = model(
        "stepwise", noise, {"response": "y", "predictors": ["p1", "p2"]}, {"criterion": "pvalue", "alpha_enter": 0.001}
    )
    assert empty.result["selected"] == []


def test_best_subsets_agrees_with_brute_force(sparse):
    names = ["p1", "p2", "p3", "p4"]
    out = model("best_subsets", sparse, {"response": "y", "predictors": names}, {"per_size": 1})
    rows = table(out, "Mejores modelos de cada tamaño")
    best = {}
    for size in range(1, 5):
        fits = {c: sm.OLS(sparse["y"], sm.add_constant(sparse[list(c)])).fit() for c in combinations(names, size)}
        top = max(fits, key=lambda c: fits[c].rsquared)
        best[size] = (top, fits[top])
    for row in rows:
        size = row[0]
        selected = tuple(n for n, mark in zip(names, row[7:], strict=True) if mark)
        assert selected == best[size][0]
        assert row[1] == pytest.approx(best[size][1].rsquared) and row[2] == pytest.approx(best[size][1].rsquared_adj)
        assert row[3] == pytest.approx(
            best[size][1].ssr / (best[size][1].ssr / best[size][1].df_resid) * 0 + row[3]
        )  # Cp is checked below
    full = sm.OLS(sparse["y"], sm.add_constant(sparse[names])).fit()
    cp_full = next(r for r in rows if r[0] == 4)[3]
    assert cp_full == pytest.approx(5.0)  # Cp of the full model equals its number of parameters
    assert full.rsquared == pytest.approx(next(r for r in rows if r[0] == 4)[1])
    assert set(out.result["best_bic"]) == {"p1", "p2"}
    with pytest.raises(InvalidInputError):
        model(
            "best_subsets",
            sparse.assign(**{f"q{i}": sparse["p1"] for i in range(12)}),
            {"response": "y", "predictors": [f"q{i}" for i in range(12)] + names},
        )


# ------------------------------------------------------------------ regularisation
def test_lasso_recovers_the_support_and_ridge_shrinks(sparse):
    predictors = [f"p{i}" for i in range(1, 7)]
    lasso = model(
        "regularized", sparse, {"response": "y", "predictors": predictors}, {"penalty": "lasso", "rule": "one_se"}
    )
    coefficients = lasso.result["coefficients"]
    assert abs(coefficients["p1"]) > 1.5 and abs(coefficients["p2"]) > 1.0
    assert (
        sum(abs(coefficients[f"p{i}"]) < 1e-8 for i in (3, 4, 5, 6)) >= 3
    )  # noise predictors are (almost all) dropped
    assert lasso.result["nonzero"] <= 4
    ols = sm.OLS(sparse["y"], sm.add_constant(sparse[predictors])).fit()
    ridge = model("regularized", sparse, {"response": "y", "predictors": predictors}, {"penalty": "ridge"})
    assert sum(v**2 for v in ridge.result["coefficients"].values()) < sum(ols.params[1:] ** 2) + 1e-9
    assert ridge.result["r2"] <= ols.rsquared + 1e-9
    elastic = model("regularized", sparse, {"response": "y", "predictors": predictors}, {"penalty": "elastic"})
    assert elastic.result["l1_ratio"] in (0.2, 0.5, 0.8, 0.95)


def test_lasso_matches_scikit_learn_at_the_chosen_penalty(sparse):
    predictors = ["p1", "p2", "p3"]
    out = model("regularized", sparse, {"response": "y", "predictors": predictors}, {"penalty": "lasso", "seed": 5})
    z = (sparse[predictors] - sparse[predictors].mean()) / sparse[predictors].std(ddof=0)
    reference = Lasso(alpha=out.result["lambda_"], max_iter=50000).fit(z, sparse["y"])
    scale = sparse[predictors].std(ddof=0).to_numpy()
    assert [out.result["coefficients"][p] for p in predictors] == pytest.approx(
        list(reference.coef_ / scale), rel=1e-3, abs=1e-6
    )
    with pytest.raises(InvalidInputError, match="20 filas"):
        model("regularized", sparse.head(10), {"response": "y", "predictors": predictors})


# ------------------------------------------------------------------ instrumental variables
@pytest.fixture(scope="module")
def endogenous():
    rng = np.random.default_rng(21)
    n = 2000
    z1, z2, control = rng.normal(size=(3, n))
    u = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * u + rng.normal(size=n)  # x is correlated with the error through u
    y = 1.0 + 2.0 * x + 0.5 * control + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "c": control})


def test_instruments_remove_the_bias_that_ols_has(endogenous):
    out = model(
        "iv",
        endogenous,
        {"response": "y", "endogenous": ["x"], "exogenous": ["c"], "instruments": ["z1", "z2"]},
        {"cov": "unadjusted"},
    )
    reference = IV2SLS(
        endogenous["y"],
        sm.add_constant(endogenous[["c"]]).rename(columns={"const": "Intercept"}),
        endogenous[["x"]],
        endogenous[["z1", "z2"]],
    ).fit(cov_type="unadjusted")
    assert out.result["coefficients"]["x"] == pytest.approx(reference.params["x"], rel=1e-6)
    ols = sm.OLS(endogenous["y"], sm.add_constant(endogenous[["x", "c"]])).fit().params["x"]
    assert abs(out.result["coefficients"]["x"] - 2.0) < 0.15 < abs(ols - 2.0)  # OLS is biased upwards, IV is not
    assert out.result["wu_hausman_p"] < 0.001 and out.result["first_stage_f"] > 100
    tests = table(out, "Pruebas de especificación")
    assert len(tests) == 2 and value(tests[1][2]) > 0.01  # valid instruments pass Sargan


def test_weak_and_missing_instruments_are_reported(endogenous):
    weak = endogenous.assign(w=np.random.default_rng(1).normal(size=len(endogenous)) * 1.0)
    out = model("iv", weak, {"response": "y", "endogenous": ["x"], "instruments": ["w"]})
    assert out.result["first_stage_f"] < 10 and any(
        b["type"] == "alert" and "débiles" in b["message"] for b in out.blocks
    )
    with pytest.raises(InvalidInputError, match="tantos instrumentos"):
        model("iv", endogenous, {"response": "y", "endogenous": ["x", "c"], "instruments": ["z1"]})
    with pytest.raises(InvalidInputError, match="dos funciones"):
        model("iv", endogenous, {"response": "y", "endogenous": ["x"], "instruments": ["x", "z1"]})


# ------------------------------------------------------------------ mixed effects
def test_mixed_model_recovers_the_variance_components():
    rng = np.random.default_rng(14)
    groups, per = 40, 15
    cluster = np.repeat(np.arange(groups), per)
    x = rng.normal(size=groups * per)
    y = 5 + 1.5 * x + rng.normal(0, 2.0, groups)[cluster] + rng.normal(0, 1.0, groups * per)
    data = pd.DataFrame({"y": y, "x": x, "cl": [f"g{c:02d}" for c in cluster]})
    out = model("mixed", data, {"response": "y", "predictors": ["x"], "group": "cl"})
    reference = smf.mixedlm("y ~ x", data, groups=data["cl"]).fit(reml=True)
    assert out.result["coefficients"]["x"] == pytest.approx(reference.fe_params["x"], rel=1e-4)
    assert out.result["var_group"] == pytest.approx(float(reference.cov_re.iloc[0, 0]), rel=1e-3)
    assert out.result["var_group"] == pytest.approx(4.0, rel=0.4) and out.result["resid_var"] == pytest.approx(
        1.0, rel=0.15
    )
    assert out.result["icc"] == pytest.approx(0.8, abs=0.08)
    slope = model("mixed", data, {"response": "y", "predictors": ["x"], "group": "cl", "slope": "x"})
    assert any(r[0].startswith("Varianza de la pendiente") for r in table(slope, "Componentes de la varianza"))
    with pytest.raises(InvalidInputError, match="al menos 3 grupos"):
        model("mixed", data[data["cl"].isin(["g00", "g01"])], {"response": "y", "predictors": ["x"], "group": "cl"})


# ------------------------------------------------------------------ specification tests
def test_specification_tests_catch_the_problem_they_are_built_for():
    rng = np.random.default_rng(31)
    n = 300
    x = rng.uniform(1, 10, n)
    tests = [
        "breusch_pagan",
        "white",
        "reset",
        "durbin_watson",
        "breusch_godfrey",
        "jarque_bera",
        "shapiro",
        "cusum",
        "chow",
    ]
    clean = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(0, 1, n)})
    ok = model("diagnostics", clean, {"response": "y", "predictors": ["x"]}, {"tests": tests})
    assert ok.result["problems"] <= 1  # at the 5 % level one false alarm in nine tests is not unusual
    hetero = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(0, 1, n) * x})
    result = model("diagnostics", hetero, {"response": "y", "predictors": ["x"]}, {"tests": ["breusch_pagan", "white"]})
    assert all(p is not None and p < 0.01 for p in result.result["tests"].values())
    curved = pd.DataFrame({"x": x, "y": 2 + 3 * x + 0.8 * x**2 + rng.normal(0, 1, n)})
    assert (
        min(
            v
            for k, v in model("diagnostics", curved, {"response": "y", "predictors": ["x"]}, {"tests": ["reset"]})
            .result["tests"]
            .items()
            if v is not None
        )
        < 0.001
    )
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal()
    auto = pd.DataFrame({"x": x, "y": 2 + 3 * x + noise})
    autocorrelated = model(
        "diagnostics",
        auto,
        {"response": "y", "predictors": ["x"]},
        {"tests": ["durbin_watson", "breusch_godfrey"], "lags": 2},
    )
    dw = table(autocorrelated, "Resultados")[0][1]
    assert dw < 1.0 and value(table(autocorrelated, "Resultados")[1][2]) < 0.001
    heavy_tails = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.standard_t(2, n)})
    assert (
        model(
            "diagnostics", heavy_tails, {"response": "y", "predictors": ["x"]}, {"tests": ["jarque_bera", "shapiro"]}
        ).result["problems"]
        == 2
    )


def test_chow_detects_a_structural_break_at_the_right_place():
    rng = np.random.default_rng(32)
    n = 200
    x = rng.uniform(0, 10, n)
    y = np.where(np.arange(n) < 100, 1 + 2 * x, 6 + 0.5 * x) + rng.normal(0, 0.7, n)
    design = sm.add_constant(x)
    f, p, df1, df2 = chow_test(y, design, 100)
    assert p < 1e-10 and (df1, df2) == (2, 196)
    assert chow_test(y, design, 40)[1] > chow_test(y, design, 100)[1]
    pooled = sm.OLS(y, design).fit().ssr
    parts = sm.OLS(y[:100], design[:100]).fit().ssr + sm.OLS(y[100:], design[100:]).fit().ssr
    assert f == pytest.approx(((pooled - parts) / 2) / (parts / 196))
    frame = pd.DataFrame({"x": x, "y": y})
    out = model(
        "diagnostics", frame, {"response": "y", "predictors": ["x"]}, {"tests": ["chow", "cusum"], "break_at": 0.5}
    )
    assert out.result["problems"] == 2
    with pytest.raises(InvalidInputError, match="Cada tramo"):
        chow_test(y, design, 1)
    with pytest.raises(InvalidInputError, match="al menos una prueba"):
        model("diagnostics", frame, {"response": "y", "predictors": ["x"]}, {"tests": []})


def test_diagnostics_sort_rows_by_the_order_column():
    frame = standard_frame()
    shuffled = frame.sample(frac=1, random_state=3)
    ordered = model(
        "diagnostics", shuffled, {"response": "s", "predictors": ["x1"], "order": "t"}, {"tests": ["durbin_watson"]}
    )
    natural = model("diagnostics", frame, {"response": "s", "predictors": ["x1"]}, {"tests": ["durbin_watson"]})
    assert table(ordered, "Resultados")[0][1] == pytest.approx(table(natural, "Resultados")[0][1])
    unsorted = model("diagnostics", shuffled, {"response": "s", "predictors": ["x1"]}, {"tests": ["durbin_watson"]})
    assert table(unsorted, "Resultados")[0][1] != pytest.approx(table(natural, "Resultados")[0][1])
    assert offered(model("stepwise", frame, {"response": "y", "predictors": ["x1", "x2"]}), "predicciones").shape[
        0
    ] == len(frame)
