"""Econometric analyses on simulated data with known truth, and against statsmodels / arch / linearmodels."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from arch import arch_model
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sigma.domain.analysis.econ_models import andrews_p_value, bai_perron, breusch_pagan_lm, kupiec, sup_f_path
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


def econ(analysis_id, frame, roles=None, params=None):
    return run("econometria", analysis_id, frame, roles, params)


def table(output, title_start):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"].startswith(title_start))["rows"]


def value(cell):
    return getattr(cell, "value", cell)


def dated(n):
    return pd.date_range("2000-01-01", periods=n, freq="MS")


# ------------------------------------------------------------------ VAR and Granger
@pytest.fixture(scope="module")
def var_data():
    rng = np.random.default_rng(1)
    n = 400
    a = np.array([[0.5, 0.0], [0.4, 0.3]])  # x1 feeds x2 but not the other way round
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.normal(size=2)
    return pd.DataFrame({"t": dated(n), "u": y[:, 0], "v": y[:, 1]})


def test_var_matches_statsmodels_and_finds_the_causal_direction(var_data):
    out = econ("var", var_data, {"vars": ["u", "v"], "time": "t"}, {"lags": 1, "horizon": 6, "forecast_steps": 4})
    reference = VAR(var_data[["u", "v"]]).fit(1)
    rows = {(r[0], r[1]): r for r in table(out, "Coeficientes por ecuación")}
    assert rows[("v", "u (rezago 1)")][2] == pytest.approx(reference.params.loc["L1.u", "v"]) and rows[
        ("v", "u (rezago 1)")
    ][2] == pytest.approx(0.4, abs=0.1)
    assert out.result["stable"] and out.result["granger_p"]["u->v"] < 1e-6 and out.result["granger_p"]["v->u"] > 0.05
    assert out.result["aic"] == pytest.approx(reference.aic)
    fevd = table(out, "Descomposición de la varianza")
    assert all(sum(row[2:]) == pytest.approx(100) for row in fevd)
    assert len(table(out, "Pronósticos")) == 4
    irf = reference.irf(6).orth_irfs
    chart = [b for b in out.blocks if b["type"] == "chart"][1]["spec"]
    assert chart["panels"][0]["series"][0]["y"] == pytest.approx(irf[:, 0, 0])  # response of u to a shock in u
    assert offered(out, "residuos").shape == (399, 3)


def test_var_selects_the_lag_and_refuses_too_little_data(var_data):
    out = econ(
        "var", var_data, {"vars": ["u", "v"], "time": "t"}, {"criterion": "bic", "max_lags": 6, "forecast_steps": 0}
    )
    assert out.result["lags"] == 1  # the true order, found by the BIC
    with pytest.raises(InvalidInputError, match="al menos"):
        econ("var", var_data.head(20), {"vars": ["u", "v"]}, {})


def test_pairwise_granger_matches_statsmodels(var_data):
    from statsmodels.tsa.stattools import grangercausalitytests

    out = econ("granger", var_data, {"x": "u", "y": "v", "time": "t"}, {"max_lag": 3, "difference": False})
    reference = grangercausalitytests(var_data[["v", "u"]], maxlag=3)
    rows = [r for r in table(out, "Pruebas por rezago") if r[0] == "u → v"]
    assert [value(r[4]) for r in rows] == pytest.approx([reference[k][0]["ssr_ftest"][1] for k in (1, 2, 3)])
    assert out.result["best"]["u->v"]["p"] < 1e-6 and out.result["best"]["v->u"]["p"] > 0.01


# ------------------------------------------------------------------ cointegration and VECM
@pytest.fixture(scope="module")
def cointegrated():
    rng = np.random.default_rng(2)
    n = 400
    x = np.cumsum(rng.normal(size=n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.6 * gap[t - 1] + rng.normal(0, 0.5)
    y = 2.0 * x + gap  # y and x share a stochastic trend: y − 2x is stationary
    walks = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    return pd.DataFrame({"t": dated(n), "x": x, "y": y, "w1": walks[:, 0], "w2": walks[:, 1]})


def test_johansen_finds_the_cointegrating_relation(cointegrated):
    out = econ("cointegration", cointegrated, {"vars": ["y", "x"], "time": "t"}, {"lags": 1})
    reference = coint_johansen(cointegrated[["y", "x"]], det_order=0, k_ar_diff=1)
    assert out.result["trace"] == pytest.approx(list(reference.lr1)) and out.result["rank"] == 1
    assert out.result["beta"][1][0] == pytest.approx(-2.0, abs=0.1)  # y − 2x = 0 → β = (1, −2), normalised on y
    assert out.result["engle_granger_p"] < 0.01
    independent = econ("cointegration", cointegrated, {"vars": ["w1", "w2"], "time": "t"}, {"lags": 1})
    assert independent.result["rank"] == 0 and independent.result["engle_granger_p"] > 0.05
    with pytest.raises(InvalidInputError, match="al menos"):
        econ("cointegration", cointegrated.head(20), {"vars": ["y", "x"]}, {})


def test_vecm_recovers_the_long_run_relation_and_the_adjustment(cointegrated):
    out = econ(
        "vecm", cointegrated, {"vars": ["x", "y"], "time": "t"}, {"lags": 1, "deterministic": "ci", "forecast_steps": 3}
    )
    assert out.result["rank"] == 1
    beta = np.array(out.result["beta"])[:, 0]
    assert -beta[0] / beta[1] == pytest.approx(2.0, abs=0.1)  # x − 0.5·y = 0: y = 2x in the long run
    alpha = np.array(out.result["alpha"])[:, 0]
    assert alpha[1] > 0 and abs(alpha[1]) > 3 * abs(
        alpha[0]
    )  # y falls when x − 0.5·y < 0: it is y that corrects the gap
    assert len(table(out, "Pronósticos")) == 3
    with pytest.raises(InvalidInputError, match="No se detecta cointegración"):
        econ("vecm", cointegrated, {"vars": ["w1", "w2"], "time": "t"}, {"lags": 1})


# ------------------------------------------------------------------ GARCH
@pytest.fixture(scope="module")
def garch_data():
    rng = np.random.default_rng(3)
    n = 2500
    r, variance = np.zeros(n), np.full(n, 1.0)
    for t in range(1, n):
        variance[t] = 0.1 + 0.1 * r[t - 1] ** 2 + 0.8 * variance[t - 1]
        r[t] = np.sqrt(variance[t]) * rng.normal()
    return pd.DataFrame({"t": dated(n), "r": r})


def test_garch_recovers_the_parameters_and_matches_arch(garch_data):
    out = econ(
        "garch",
        garch_data,
        {"series": "r", "time": "t"},
        {"variant": "garch", "distribution": "normal", "percent": False, "horizon": 3},
    )
    reference = arch_model(garch_data["r"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    assert out.result["parameters"]["alpha[1]"] == pytest.approx(reference.params["alpha[1]"], rel=1e-4)
    assert out.result["parameters"]["alpha[1]"] == pytest.approx(0.1, abs=0.04) and out.result["parameters"][
        "beta[1]"
    ] == pytest.approx(0.8, abs=0.07)
    assert out.result["persistence"] == pytest.approx(0.9, abs=0.05) and out.result["aic"] == pytest.approx(
        reference.aic
    )
    assert (
        out.result["lm_p"] < 0.01 and out.result["residual_lm_p"] > 0.01
    )  # ARCH effects in the series, none left in the residuals
    assert len(table(out, "Pronóstico de la volatilidad")) == 3
    volatility = offered(out, "volatilidad")["volatilidad"]
    assert volatility.to_numpy() == pytest.approx(np.asarray(reference.conditional_volatility), rel=1e-3)


def test_garch_variants_and_value_at_risk(garch_data):
    for variant in ("gjr", "egarch", "tarch", "arch"):
        out = econ(
            "garch",
            garch_data.head(900),
            {"series": "r", "time": "t"},
            {"variant": variant, "percent": True, "horizon": 0},
        )
        assert np.isfinite(out.result["aic"]) and out.result["parameters"]["omega"] > 0
    var = econ(
        "garch",
        garch_data,
        {"series": "r", "time": "t"},
        {"variant": "garch", "var_level": 0.05, "distribution": "normal", "percent": False},
    )
    assert var.result["var_hits"] / 2500 == pytest.approx(0.05, abs=0.02) and var.result["kupiec_p"] > 0.01
    rng = np.random.default_rng(5)
    iid = pd.DataFrame({"r": rng.normal(size=1500)})
    quiet = econ("garch", iid, {"series": "r"}, {"variant": "garch", "percent": False})
    assert quiet.result["lm_p"] > 0.01
    with pytest.raises(InvalidInputError, match="al menos"):
        econ("garch", garch_data.head(50), {"series": "r"}, {})


def test_kupiec_test_matches_its_definition():
    lr, p = kupiec(10, 250, 0.05)
    expected = -2 * (240 * np.log(0.95) + 10 * np.log(0.05) - 240 * np.log(1 - 10 / 250) - 10 * np.log(10 / 250))
    assert lr == pytest.approx(expected) and 0 < p < 1
    assert kupiec(12, 240, 0.05)[0] == pytest.approx(0, abs=1e-9)  # exactly the expected number of violations
    assert kupiec(0, 100, 0.05)[0] > 0 and kupiec(100, 100, 0.05)[1] < 1e-6


# ------------------------------------------------------------------ ARDL
def test_ardl_recovers_the_long_run_multiplier():
    rng = np.random.default_rng(6)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + rng.normal(0, 0.5)
    data = pd.DataFrame({"t": dated(n), "y": y, "x": x})
    out = econ("ardl", data, {"y": "y", "x": ["x"], "time": "t"}, {"max_lag_y": 3, "max_lag_x": 3, "case": "3"})
    assert out.result["long_run"]["x"] == pytest.approx((0.8 + 0.4) / (1 - 0.5), abs=0.25)
    assert out.result["f_bounds"] > 20 and out.result["bounds_p_upper"] < 0.001
    assert out.result["speed_of_adjustment"] == pytest.approx(-0.5, abs=0.12)
    selection = ardl_select_order(data["y"], 3, data[["x"]], 3, trend="c", ic="aic")
    reference = ARDL(data["y"], selection.model.ardl_order[0], data[["x"]], selection.model.dl_lags, trend="c").fit()
    assert list(out.result["order"]) == list(selection.model.ardl_order)
    coefficients = {r[0]: r[1] for r in table(out, "Coeficientes del ARDL")}
    assert coefficients["x.L0"] == pytest.approx(reference.params["x.L0"], rel=1e-4)
    noise = pd.DataFrame({"y": rng.normal(size=300), "x": rng.normal(size=300)})
    with pytest.raises(InvalidInputError, match="ninguna variable explicativa"):
        econ("ardl", noise, {"y": "y", "x": ["x"]}, {"max_lag_y": 2, "max_lag_x": 2})


# ------------------------------------------------------------------ panel data
@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(7)
    entities, periods = 60, 8
    effect = rng.normal(0, 2, entities)
    x = (
        rng.normal(size=(entities, periods)) + 0.8 * effect[:, None]
    )  # the effect is correlated with x: RE is inconsistent
    y = 1.5 * x + effect[:, None] + rng.normal(size=(entities, periods))
    return pd.DataFrame(
        {
            "ent": np.repeat([f"e{i:02d}" for i in range(entities)], periods),
            "per": np.tile(np.arange(periods), entities),
            "x": x.ravel(),
            "y": y.ravel(),
        }
    )


def test_fixed_effects_are_consistent_and_hausman_rejects_random_effects(panel):
    out = econ(
        "panel", panel, {"response": "y", "predictors": ["x"], "entity": "ent", "time": "per"}, {"cov": "clustered"}
    )
    assert out.result["fixed"]["x"] == pytest.approx(1.5, abs=0.08)
    assert out.result["pooled"]["x"] > 1.8  # omitted-effect bias pushes the pooled estimate up
    assert (
        out.result["hausman_p"] < 0.01 and out.result["poolability_p"] < 1e-6 and out.result["breusch_pagan_p"] < 1e-6
    )
    index = panel.set_index(["ent", "per"])
    fixed = PanelOLS(index["y"], index[["x"]], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    assert out.result["fixed"]["x"] == pytest.approx(fixed.params["x"])
    random = RandomEffects(index["y"], sm.add_constant(index[["x"]])).fit(cov_type="clustered", cluster_entity=True)
    assert out.result["random"]["x"] == pytest.approx(random.params["x"])
    within = panel["x"] - panel.groupby("ent")["x"].transform("mean")
    demeaned = panel["y"] - panel.groupby("ent")["y"].transform("mean")
    assert out.result["fixed"]["x"] == pytest.approx((within * demeaned).sum() / (within**2).sum())
    assert len(offered(out, "efectos")) == 60


def test_panel_validation_and_the_breusch_pagan_statistic(panel):
    with pytest.raises(InvalidInputError, match="repetidas"):
        econ(
            "panel",
            pd.concat([panel, panel.head(2)]),
            {"response": "y", "predictors": ["x"], "entity": "ent", "time": "per"},
        )
    with pytest.raises(InvalidInputError, match="al menos 3 individuos"):
        econ("panel", panel.head(8), {"response": "y", "predictors": ["x"], "entity": "ent", "time": "per"})
    constant = panel.assign(z=panel["ent"].str[1:].astype(int) % 3)
    with pytest.raises(Exception, match="varía en el tiempo|no se pudo"):  # noqa: B017 - the message is what matters
        econ("panel", constant, {"response": "y", "predictors": ["x", "z"], "entity": "ent", "time": "per"})
    e = np.random.default_rng(8).normal(size=(30, 6))
    balanced = 30 * 6 / (2 * 5) * ((e.sum(axis=1) ** 2).sum() / (e**2).sum() - 1) ** 2
    assert breusch_pagan_lm(e.ravel(), np.repeat(np.arange(30), 6))[0] == pytest.approx(balanced)


# ------------------------------------------------------------------ structural breaks
@pytest.fixture(scope="module")
def shifted():
    rng = np.random.default_rng(9)
    values = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100), rng.normal(-1, 1, 100)]
    return pd.DataFrame({"t": dated(300), "y": values})


def test_quandt_andrews_finds_the_break_and_its_critical_value_matches_andrews(shifted):
    one = pd.DataFrame(
        {
            "t": dated(200),
            "y": np.r_[np.random.default_rng(10).normal(0, 1, 100), np.random.default_rng(11).normal(2, 1, 100)],
        }
    )
    out = econ("qa_break", one, {"series": "y", "time": "t"}, {"trim": 0.15})
    assert abs(out.result["break_position"] - 101) <= 5 and out.result["p"] < 0.001
    calm = pd.DataFrame({"y": np.random.default_rng(12).normal(size=200)})
    assert econ("qa_break", calm, {"series": "y"}, {}).result["p"] > 0.05
    p, critical = andrews_p_value(8.68, 1, 0.15, reps=6000)
    assert critical == pytest.approx(8.68, abs=0.5) and p == pytest.approx(
        0.05, abs=0.02
    )  # Andrews (1993), k = 1, 15 % trimming
    x = np.ones((60, 1))
    y = np.random.default_rng(13).normal(size=60)
    positions, f = sup_f_path(x, y, 0.15)
    s = int(positions[10])
    rss = sm.OLS(y, x).fit().ssr
    split = sm.OLS(y[:s], x[:s]).fit().ssr + sm.OLS(y[s:], x[s:]).fit().ssr
    assert f[10] == pytest.approx(((rss - split) / 1) / (split / 58))


def test_bai_perron_counts_and_places_the_breaks(shifted):
    out = econ("bai_perron", shifted, {"series": "y", "time": "t"}, {"max_breaks": 3, "trim": 0.1})
    assert out.result["chosen"] == 2 and all(
        abs(b - e) <= 4 for b, e in zip(out.result["breaks"], (101, 201), strict=True)
    )
    none = econ(
        "bai_perron",
        pd.DataFrame({"y": np.random.default_rng(14).normal(size=150)}),
        {"series": "y"},
        {"max_breaks": 2},
    )
    assert none.result["chosen"] == 0
    rng = np.random.default_rng(15)
    y = np.r_[rng.normal(0, 1, 15), rng.normal(4, 1, 15)]
    found = bai_perron(np.ones((30, 1)), y, 1, 5)
    brute = min(((y[:s] - y[:s].mean()) ** 2).sum() + ((y[s:] - y[s:].mean()) ** 2).sum() for s in range(5, 26))
    assert found[1][0] == pytest.approx(brute)
    with pytest.raises(InvalidInputError, match="600"):
        econ("bai_perron", pd.DataFrame({"y": np.random.default_rng(1).normal(size=700)}), {"series": "y"}, {})


# ------------------------------------------------------------------ regimes and state space
def test_markov_switching_recovers_two_regimes():
    rng = np.random.default_rng(16)
    n = 600
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.uniform() < 0.95 else 1 - state[t - 1]
    y = np.where(state == 0, 0.0, 4.0) + rng.normal(0, np.where(state == 0, 0.5, 1.5))
    out = econ("markov", pd.DataFrame({"t": dated(n), "y": y}), {"series": "y", "time": "t"}, {"restarts": 8})
    assert sorted(round(m) for m in out.result["means"]) == [0, 4]
    low, high = np.argsort(out.result["means"])
    assert out.result["durations"][low] == pytest.approx(20, rel=0.6) and out.result["transition"][low][
        low
    ] == pytest.approx(0.95, abs=0.04)
    regime = offered(out, "regimenes")["regimen"].to_numpy() - 1
    agreement = max((regime == state).mean(), (regime != state).mean())
    assert agreement > 0.92


def test_unobserved_components_local_level_matches_statsmodels():
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    rng = np.random.default_rng(17)
    n = 300
    level = np.cumsum(rng.normal(0, 0.5, n))
    y = level + rng.normal(0, 1.0, n)
    out = econ(
        "ucm",
        pd.DataFrame({"t": dated(n), "y": y}),
        {"series": "y", "time": "t"},
        {"structure": "llevel", "horizon": 6},
    )
    reference = UnobservedComponents(y, level="llevel").fit(disp=False)
    assert out.result["aic"] == pytest.approx(reference.aic, rel=1e-4)
    variances = out.result["variances"]
    assert variances["sigma2.level"] == pytest.approx(0.25, rel=0.6) and variances["sigma2.irregular"] == pytest.approx(
        1.0, rel=0.4
    )
    assert len(table(out, "Pronósticos")) == 6
    components = offered(out, "componentes")
    assert np.corrcoef(components["nivel"], level)[0, 1] > 0.95
    with pytest.raises(InvalidInputError, match="periodo"):
        econ("ucm", pd.DataFrame({"y": y}), {"series": "y"}, {"seasonal": True})


def test_the_standard_frame_exposes_the_econometric_examples():
    frame = standard_frame()
    assert {"ent", "per", "yp", "ret", "rg"} <= set(frame.columns) and frame["ent"].nunique() == 20
