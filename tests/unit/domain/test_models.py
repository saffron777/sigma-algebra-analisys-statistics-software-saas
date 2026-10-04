"""Regression models against statsmodels / sklearn run directly on the same data, and against known truths."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

from sigma.domain.analysis.models import cameron_trivedi, hosmer_lemeshow
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def model(analysis_id, frame, roles=None, params=None):
    return run("modelos", analysis_id, frame, roles, params)


def coefficients(output):
    return output.result["coefficients"]


# ------------------------------------------------------------------ binary
def test_logistic_regression_matches_statsmodels(frame):
    out = model("binary", frame, {"response": "b", "predictors": ["x1", "x2"]})
    reference = sm.Logit(frame["b"], sm.add_constant(frame[["x1", "x2"]])).fit(disp=False)
    got = coefficients(out)
    assert got["Intercept"] == pytest.approx(reference.params["const"], rel=1e-5)
    assert got["x1"] == pytest.approx(reference.params["x1"], rel=1e-5) and got["x2"] == pytest.approx(
        reference.params["x2"], rel=1e-5
    )
    assert out.result["mcfadden"] == pytest.approx(reference.prsquared)
    assert out.result["auc"] == pytest.approx(roc_auc_score(frame["b"], reference.predict()))
    assert out.result["lr_p"] == pytest.approx(reference.llr_pvalue)
    fitted = offered(out, "predicciones")
    assert fitted["probabilidad"].to_numpy() == pytest.approx(reference.predict())


def test_logistic_regression_with_factors_and_odds_ratios(frame):
    out = model("binary", frame, {"response": "b", "predictors": ["x1", "g"]})
    assert set(coefficients(out)) == {"Intercept", "g = B", "g = C", "x1"}
    reference = smf.logit("b ~ x1 + C(g)", frame).fit(disp=False)
    assert coefficients(out)["g = B"] == pytest.approx(reference.params["C(g)[T.B]"], rel=1e-5)
    table = next(b for b in out.blocks if b["type"] == "table" and b["title"] == "Coeficientes")
    assert table["columns"][7] == "Odds ratio"
    row = next(r for r in table["rows"] if r[0] == "x1")
    assert row[7] == pytest.approx(np.exp(coefficients(out)["x1"]))


def test_event_level_and_text_responses(frame):
    text = frame.assign(resultado=np.where(frame["b"] == 1, "si", "no"))
    default = model("binary", text, {"response": "resultado", "predictors": ["x1"]})
    assert default.result["events"] == int((frame["b"] == 1).sum())
    flipped = model("binary", text, {"response": "resultado", "predictors": ["x1"]}, {"event": "no"})
    assert coefficients(flipped)["x1"] == pytest.approx(-coefficients(default)["x1"], rel=1e-4)


def test_hosmer_lemeshow_and_the_probit_link(frame):
    reference = sm.Logit(frame["b"], sm.add_constant(frame[["x1", "x2"]])).fit(disp=False)
    statistic, p, groups = hosmer_lemeshow(frame["b"].to_numpy(), reference.predict())
    assert groups == 10 and 0 <= p <= 1 and statistic >= 0
    perfect = hosmer_lemeshow(np.array([0, 0, 1, 1] * 10), np.array([0.5] * 40))
    assert perfect[0] == pytest.approx(0)
    probit = model("binary", frame, {"response": "b", "predictors": ["x1", "x2"]}, {"link": "probit"})
    ref = sm.Probit(frame["b"], sm.add_constant(frame[["x1", "x2"]])).fit(disp=False)
    assert coefficients(probit)["x1"] == pytest.approx(ref.params["x1"], rel=1e-4)


def test_binary_regression_refuses_bad_responses(frame):
    with pytest.raises(InvalidInputError, match="exactamente 2"):
        model("binary", frame, {"response": "g", "predictors": ["x1"]})
    with pytest.raises(InvalidInputError, match="exactamente 2"):
        model("binary", frame.assign(k=0), {"response": "k", "predictors": ["x1"]})


# ------------------------------------------------------------------ ordinal and multinomial
def test_ordinal_regression_matches_orderedmodel(frame):
    out = model("ordinal", frame, {"response": "r1", "predictors": ["x1", "x2"]})
    endog = pd.Series(pd.Categorical(frame["r1"], categories=sorted(frame["r1"].unique()), ordered=True))
    reference = OrderedModel(endog, frame[["x1", "x2"]], distr="logit").fit(method="bfgs", disp=False, maxiter=1000)
    assert coefficients(out)["x1"] == pytest.approx(reference.params["x1"], rel=1e-4)
    assert coefficients(out)["x2"] == pytest.approx(reference.params["x2"], rel=1e-4)
    assert 0.2 < out.result["accuracy"] <= 1
    probabilities = offered(out, "predicciones")[
        [c for c in offered(out, "predicciones").columns if c.startswith("prob_")]
    ]
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(frame)))
    with pytest.raises(InvalidInputError, match="al menos 3"):
        model("ordinal", frame, {"response": "b", "predictors": ["x1"]})


def test_multinomial_regression_matches_mnlogit(frame):
    out = model("multinomial", frame, {"response": "g", "predictors": ["x1", "x2"]})
    codes = frame["g"].map({"A": 0, "B": 1, "C": 2})
    reference = sm.MNLogit(codes, sm.add_constant(frame[["x1", "x2"]])).fit(disp=False, method="newton")
    table = next(b for b in out.blocks if b["type"] == "table" and b["title"].startswith("Coeficientes"))
    row = next(r for r in table["rows"] if r[0] == "B" and r[1] == "x1")
    assert row[2] == pytest.approx(reference.params.loc["x1", 0], rel=1e-4)
    assert out.result["llr_p"] == pytest.approx(reference.llr_pvalue)
    with pytest.raises(InvalidInputError, match="no existe"):
        model("multinomial", frame, {"response": "g", "predictors": ["x1"]}, {"reference": "Z"})


# ------------------------------------------------------------------ counts and GLM
def test_poisson_regression_matches_glm_and_detects_overdispersion(frame):
    out = model("count", frame, {"response": "count", "predictors": ["x1", "x2"]})
    reference = smf.glm("count ~ x1 + x2", frame, family=sm.families.Poisson()).fit()
    assert coefficients(out)["x1"] == pytest.approx(reference.params["x1"], rel=1e-5)
    assert out.result["dispersion"] == pytest.approx(reference.pearson_chi2 / reference.df_resid)
    rng = np.random.default_rng(5)
    x = rng.normal(size=600)
    mu = np.exp(0.5 + 0.4 * x)
    over = pd.DataFrame({"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu))})
    poisson = model("count", over, {"response": "y", "predictors": ["x"]})
    assert poisson.result["dispersion"] > 1.5 and poisson.result["cameron_trivedi_p"] < 0.01
    assert any(b["type"] == "alert" and "sobredispersión" in b["message"] for b in poisson.blocks)
    negbin = model("count", over, {"response": "y", "predictors": ["x"]}, {"family": "negbin"})
    assert coefficients(negbin)["x"] == pytest.approx(0.4, abs=0.08)
    quasi = model("count", over, {"response": "y", "predictors": ["x"]}, {"family": "quasi"})
    plain_se = smf.glm("y ~ x", over, family=sm.families.Poisson()).fit().bse["x"]
    quasi_se = next(
        r
        for r in next(b for b in quasi.blocks if b["type"] == "table" and b["title"] == "Coeficientes")["rows"]
        if r[0] == "x"
    )[2]
    assert quasi_se > 1.3 * plain_se


def test_count_regression_with_exposure_and_validation(frame):
    exposure = frame.assign(t_obs=np.linspace(1, 3, len(frame)))
    out = model("count", exposure, {"response": "count", "predictors": ["x1"], "exposure": "t_obs"})
    reference = smf.glm("count ~ x1", exposure, family=sm.families.Poisson(), offset=np.log(exposure["t_obs"])).fit()
    assert coefficients(out)["x1"] == pytest.approx(reference.params["x1"], rel=1e-5)
    with pytest.raises(InvalidInputError, match="conteos"):
        model("count", frame, {"response": "x1", "predictors": ["x2"]})


def test_glm_families_match_statsmodels(frame):
    positive = frame.assign(gain=np.exp(frame["x1"] / 5) + frame["x3"])
    out = model("glm", positive, {"response": "gain", "predictors": ["x1", "x3"]}, {"family": "gamma", "link": "log"})
    reference = smf.glm("gain ~ x1 + x3", positive, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    assert coefficients(out)["x1"] == pytest.approx(reference.params["x1"], rel=1e-5)
    assert out.result["deviance"] == pytest.approx(reference.deviance) and out.result["aic"] == pytest.approx(
        reference.aic
    )
    gaussian = model("glm", frame, {"response": "y", "predictors": ["x1", "x2"]}, {"family": "gaussian"})
    assert coefficients(gaussian)["x1"] == pytest.approx(smf.ols("y ~ x1 + x2", frame).fit().params["x1"], rel=1e-6)
    with pytest.raises(InvalidInputError, match="liga"):
        model("glm", frame, {"response": "count", "predictors": ["x1"]}, {"family": "poisson", "link": "logit"})
    with pytest.raises(InvalidInputError, match="estrictamente positiva"):
        model("glm", frame.assign(neg=frame["y"] - 100), {"response": "neg", "predictors": ["x1"]}, {"family": "gamma"})


# ------------------------------------------------------------------ quantile, robust, weighted, polynomial
def test_quantile_regression_matches_quantreg(frame):
    out = model("quantile", frame, {"response": "y", "predictors": ["x1", "x2"]}, {"quantiles": ["0.1", "0.5"]})
    for q in (0.1, 0.5):
        reference = smf.quantreg("y ~ x1 + x2", frame).fit(q=q, max_iter=2000)
        assert out.result["coefficients"][f"{q:g}"]["x1"] == pytest.approx(reference.params["x1"], rel=1e-4)
    with pytest.raises(InvalidInputError, match="al menos un cuantil"):
        model("quantile", frame, {"response": "y", "predictors": ["x1"]}, {"quantiles": []})


def test_robust_standard_errors_and_m_estimation(frame):
    hc3 = model("robust", frame, {"response": "y", "predictors": ["x1", "x2"]}, {"method": "hc3"})
    reference = smf.ols("y ~ x1 + x2", frame).fit(cov_type="HC3")
    assert hc3.result["standard_errors"]["x1"] == pytest.approx(reference.bse["x1"])
    assert hc3.result["coefficients"]["x1"] == pytest.approx(smf.ols("y ~ x1 + x2", frame).fit().params["x1"])
    hac = model("robust", frame, {"response": "y", "predictors": ["x1"]}, {"method": "hac", "lags": 3})
    assert hac.result["standard_errors"]["x1"] == pytest.approx(
        smf.ols("y ~ x1", frame).fit(cov_type="HAC", cov_kwds={"maxlags": 3}).bse["x1"]
    )
    dirty = frame.assign(y=frame["y"].where(frame.index % 40 != 0, frame["y"] + 80))
    huber = model("robust", dirty, {"response": "y", "predictors": ["x1", "x2"]}, {"method": "huber"})
    plain = smf.ols("y ~ x1 + x2", dirty).fit()
    truth = smf.ols("y ~ x1 + x2", frame).fit().params["x2"]
    assert abs(huber.result["coefficients"]["x2"] - truth) < abs(plain.params["x2"] - truth)
    weights = offered(huber, "predicciones")["peso_robusto"]
    assert weights[dirty.index % 40 == 0].mean() < 0.5 < weights[dirty.index % 40 != 0].mean()


def test_weighted_least_squares_matches_statsmodels(frame):
    weighted = frame.assign(w=1 / (1 + frame["x3"]))
    out = model("wls", weighted, {"response": "y", "predictors": ["x1", "x2"], "weights": "w"})
    reference = sm.WLS(weighted["y"], sm.add_constant(weighted[["x1", "x2"]]), weights=weighted["w"]).fit()
    assert coefficients(out)["x1"] == pytest.approx(reference.params["x1"], rel=1e-6)
    estimated = model("wls", frame, {"response": "y", "predictors": ["x1", "x2"]})
    assert "pesos estimados" in estimated.subtitle
    with pytest.raises(InvalidInputError, match="positivos"):
        model("wls", weighted.assign(w=-1.0), {"response": "y", "predictors": ["x1"], "weights": "w"})


def test_polynomial_regression_matches_polyfit_and_the_sequential_test():
    rng = np.random.default_rng(8)
    x = np.linspace(-3, 3, 120)
    data = pd.DataFrame({"x": x, "y": 1 + 0.5 * x - 0.8 * x**2 + rng.normal(0, 0.5, 120)})
    out = model("polynomial", data, {"response": "y", "predictors": ["x"]}, {"degree": 3})
    quadratic = model("polynomial", data, {"response": "y", "predictors": ["x"]}, {"degree": 2})
    fit = np.polyfit(x, data["y"], 2)
    assert coefficients(quadratic)["x²"] == pytest.approx(fit[0], rel=1e-5) and coefficients(quadratic)[
        "x"
    ] == pytest.approx(fit[1], rel=1e-5)
    sequence = next(b for b in out.blocks if b["type"] == "table" and b["title"].startswith("Selección del grado"))[
        "rows"
    ]
    assert getattr(sequence[1][2], "value", sequence[1][2]) < 0.001  # degree 2 is needed
    assert getattr(sequence[2][2], "value", sequence[2][2]) > 0.01  # degree 3 adds nothing
    with pytest.raises(InvalidInputError, match="numérico"):
        model("polynomial", standard_frame(), {"response": "y", "predictors": ["g"]}, {"degree": 2})


def test_interactions_are_built_from_the_predictors(frame):
    out = model(
        "polynomial", frame, {"response": "y", "predictors": ["x1", "x2"]}, {"degree": 1, "interactions": "two"}
    )
    assert "x1 × x2" in coefficients(out)
    reference = smf.ols("y ~ x1 + x2 + x1:x2", frame).fit()
    assert coefficients(out)["x1 × x2"] == pytest.approx(reference.params["x1:x2"], rel=1e-6)


# ------------------------------------------------------------------ nonlinear
def test_nonlinear_regression_recovers_the_parameters():
    rng = np.random.default_rng(3)
    x = np.linspace(0.5, 20, 80)
    data = pd.DataFrame({"x": x, "y": 12 * x / (3 + x) + rng.normal(0, 0.15, 80)})
    out = model("nonlinear", data, {"x": "x", "y": "y"}, {"model": "michaelis"})
    assert out.result["parameters"]["a"] == pytest.approx(12, rel=0.03) and out.result["parameters"][
        "b"
    ] == pytest.approx(3, rel=0.08)
    assert out.result["r2"] > 0.99
    custom = model(
        "nonlinear",
        data,
        {"x": "x", "y": "y"},
        {"model": "custom", "expression": "a * x / (b + x)", "parameters": "a, b", "start": "10, 2"},
    )
    assert custom.result["parameters"]["a"] == pytest.approx(out.result["parameters"]["a"], rel=1e-4)
    growth = pd.DataFrame({"x": x, "y": 2 * np.exp(0.15 * x) + rng.normal(0, 0.1, 80)})
    expo = model("nonlinear", growth, {"x": "x", "y": "y"}, {"model": "exponential"})
    assert expo.result["parameters"]["b"] == pytest.approx(0.15, rel=0.03)


def test_nonlinear_regression_refuses_hostile_or_broken_input():
    data = pd.DataFrame({"x": np.linspace(1, 10, 30), "y": np.linspace(1, 10, 30) ** 2})
    for expression in ("__import__('os').getcwd() + a", "a.__class__", "a * open('f')"):
        with pytest.raises(InvalidInputError):
            model(
                "nonlinear",
                data,
                {"x": "x", "y": "y"},
                {"model": "custom", "expression": expression, "parameters": "a"},
            )
    with pytest.raises(InvalidInputError, match="valores iniciales"):
        model("nonlinear", data, {"x": "x", "y": "y"}, {"model": "power", "start": "1"})
    with pytest.raises(InvalidInputError, match="Indique"):
        model("nonlinear", data, {"x": "x", "y": "y"}, {"model": "custom"})


def test_the_cameron_trivedi_test_detects_extra_dispersion():
    rng = np.random.default_rng(2)
    mu = np.full(2000, 4.0)
    assert cameron_trivedi(rng.poisson(mu).astype(float), mu)[1] > 0.01
    assert cameron_trivedi(rng.negative_binomial(2, 2 / 6, 2000).astype(float), mu)[1] < 0.001
