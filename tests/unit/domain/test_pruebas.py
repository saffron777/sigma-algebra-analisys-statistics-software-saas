"""The additional hypothesis tests against scipy / statsmodels and published values."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats import contingency_tables as ct
from statsmodels.stats import rates
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.weightstats import ttost_ind, ttost_paired

from sigma.domain.analysis.pruebas import icc_table
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def go(analysis_id, frame, roles=None, params=None):
    return run("pruebas", analysis_id, frame, roles, params)


def table(output, title):
    block = next(b for b in output.blocks if b["type"] == "table" and b["title"] == title)
    return block["rows"]


def value(cell):
    return getattr(cell, "value", cell)


# ------------------------------------------------------------------ normality
def test_anderson_darling_matches_scipy(frame):
    out = go("anderson", frame, {"var": "y"}, {"dist": "norm"})
    reference = stats.anderson(frame["y"].to_numpy(), dist="norm", method="interpolate")
    assert out.result["statistic"] == pytest.approx(reference.statistic) and out.result["p"] == pytest.approx(
        reference.pvalue
    )
    skewed = frame.assign(v=np.exp(frame["x2"]))
    assert go("anderson", skewed, {"var": "v"}, {"dist": "norm"}).result["p"] <= 0.02
    with pytest.raises(InvalidInputError, match="al menos"):
        go("anderson", frame.head(4), {"var": "y"})


def test_lilliefors_matches_statsmodels(frame):
    out = go("lilliefors", frame, {"var": "x1"})
    statistic, pvalue = lilliefors(frame["x1"].to_numpy(), dist="norm", pvalmethod="table")
    assert out.result["statistic"] == pytest.approx(statistic) and out.result["p"] == pytest.approx(pvalue)
    with pytest.raises(InvalidInputError, match="no negativos"):
        go("lilliefors", frame.assign(v=frame["x1"] - 12), {"var": "v"}, {"dist": "exp"})


def test_kolmogorov_smirnov_one_and_two_samples(frame):
    out = go("ks_one", frame, {"var": "x1"}, {"dist": "norm", "loc": 10, "scale": 2})
    reference = stats.kstest(frame["x1"], stats.norm(10, 2).cdf)
    assert out.result["statistic"] == pytest.approx(reference.statistic) and out.result["p"] == pytest.approx(
        reference.pvalue
    )
    uniform = go("ks_one", frame, {"var": "x3"}, {"dist": "uniform", "loc": 0, "scale": 10})
    assert uniform.result["p"] == pytest.approx(stats.kstest(frame["x3"], stats.uniform(0, 10).cdf).pvalue)
    two = go("ks_two", frame, {"var": "y", "group": "h"})
    a, b = (g["y"] for _, g in frame.groupby("h"))
    assert two.result["p"] == pytest.approx(stats.ks_2samp(a, b).pvalue)
    with pytest.raises(InvalidInputError, match="2 categorías"):
        go("ks_two", frame, {"var": "y", "group": "g"})


def test_jarque_bera_matches_scipy(frame):
    out = go("jarque_bera", frame, {"var": "y"})
    reference = stats.jarque_bera(frame["y"])
    assert out.result["statistic"] == pytest.approx(reference.statistic) and out.result["p"] == pytest.approx(
        reference.pvalue
    )
    assert out.result["skew"] == pytest.approx(stats.skew(frame["y"]))


# ------------------------------------------------------------------ variances
def test_the_four_variance_tests_match_scipy(frame):
    out = go("variances", frame, {"var": "y", "group": "g"})
    samples = [g["y"].to_numpy() for _, g in frame.groupby("g")]
    assert out.result["levene_p"] == pytest.approx(stats.levene(*samples, center="mean").pvalue)
    assert out.result["brown_forsythe_p"] == pytest.approx(stats.levene(*samples, center="median").pvalue)
    assert out.result["bartlett_p"] == pytest.approx(stats.bartlett(*samples).pvalue)
    assert out.result["fligner_p"] == pytest.approx(stats.fligner(*samples).pvalue)
    rows = table(out, "Desviaciones estándar por grupo")
    first = samples[0]
    df = len(first) - 1
    assert value(rows[0][4]) == pytest.approx(first.std(ddof=1) * np.sqrt(df / stats.chi2.ppf(1 - 0.05 / 6, df)))


def test_unequal_variances_are_detected():
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"v": np.r_[rng.normal(0, 1, 80), rng.normal(0, 4, 80)], "g": ["a"] * 80 + ["b"] * 80})
    assert go("variances", data, {"var": "v", "group": "g"}).result["brown_forsythe_p"] < 0.001


# ------------------------------------------------------------------ nonparametric
def test_sign_test_matches_the_binomial_test(frame):
    out = go("sign_test", frame, {"var": "x1"}, {"median0": 10})
    centred = frame["x1"] - 10
    reference = stats.binomtest(int((centred > 0).sum()), int((centred != 0).sum()), 0.5)
    assert out.result["p"] == pytest.approx(reference.pvalue)
    assert out.result["positive"] + out.result["negative"] == len(frame)
    confidence, low, high = (value(c) for c in table(out, "Intervalo de confianza para la mediana")[0])
    assert low <= frame["x1"].median() <= high and confidence >= 0.94
    paired = go("sign_test", frame, {"var": "x1", "var2": "x2"})
    differences = frame["x1"] - frame["x2"]
    assert paired.result["positive"] == int((differences > 0).sum())


def test_mood_median_friedman_and_runs_match_references(frame):
    mood = go("mood_median", frame, {"var": "y", "group": "g"})
    reference = stats.median_test(*[g["y"] for _, g in frame.groupby("g")])
    assert mood.result["statistic"] == pytest.approx(reference[0]) and mood.result["grand_median"] == pytest.approx(
        reference[2]
    )
    friedman = go("friedman", frame, {"vars": ["r1", "r2", "r3"]})
    reference = stats.friedmanchisquare(frame["r1"], frame["r2"], frame["r3"])
    assert friedman.result["statistic"] == pytest.approx(reference.statistic) and friedman.result["p"] == pytest.approx(
        reference.pvalue
    )
    assert friedman.result["kendall_w"] == pytest.approx(reference.statistic / (len(frame) * 2))
    runs = go("runs", frame, {"var": "s"}, {"cutoff": "median"})
    z, p = runstest_1samp(frame["s"].to_numpy(), cutoff=frame["s"].median(), correction=False)
    assert runs.result["z"] == pytest.approx(z) and runs.result["p"] == pytest.approx(p)
    assert runs.result["runs"] < runs.result["expected"]  # a trending series has few runs


def test_mcnemar_and_cochran_q_match_statsmodels(frame):
    out = go("mcnemar", frame, {"a": "c1", "b": "c2"}, {"method": "exact"})
    table_ = pd.crosstab(frame["c1"], frame["c2"]).iloc[::-1, ::-1].to_numpy()
    assert out.result["p"] == pytest.approx(ct.mcnemar(table_, exact=True).pvalue)
    assert out.result["discordant"] == table_[0, 1] + table_[1, 0]
    chi = go("mcnemar", frame, {"a": "c1", "b": "c2"}, {"method": "chi2"})
    assert chi.result["p"] == pytest.approx(ct.mcnemar(table_, exact=False, correction=True).pvalue)
    cochran = go("cochran_q", frame, {"vars": ["c1", "c2", "c3"]})
    reference = ct.cochrans_q(frame[["c1", "c2", "c3"]].to_numpy())
    assert cochran.result["statistic"] == pytest.approx(reference.statistic) and cochran.result["p"] == pytest.approx(
        reference.pvalue
    )
    with pytest.raises(InvalidInputError, match="0 .* 1"):
        go("cochran_q", frame, {"vars": ["r1", "r2", "r3"]})


# ------------------------------------------------------------------ rates and tables
def test_poisson_rates_match_statsmodels():
    one = go("poisson_rate", None, params={"design": "one", "count1": 5, "exposure1": 10, "rate0": 0.3})
    assert one.result["p"] == pytest.approx(rates.test_poisson(5, 10, 0.3, method="exact-c").pvalue)
    two = go(
        "poisson_rate", None, params={"design": "two", "count1": 10, "exposure1": 20, "count2": 15, "exposure2": 22}
    )
    reference = rates.test_poisson_2indep(10, 20, 15, 22, value=1, method="exact-cond")
    assert two.result["p"] == pytest.approx(reference.pvalue) and two.result["ratio"] == pytest.approx(reference.ratio)
    low, high = rates.confint_poisson_2indep(10, 20, 15, 22, method="score")
    assert (
        value(table(two, "Razón de tasas (1 / 2)")[0][1]),
        value(table(two, "Razón de tasas (1 / 2)")[0][2]),
    ) == pytest.approx((low, high))


def test_odds_ratio_of_a_known_table():
    # exposed: 20 events, 10 without; unexposed: 8 events, 25 without -> OR = 20*25/(10*8) = 6.25, RR = 2.75
    rows = [("e", "ev")] * 20 + [("e", "no")] * 10 + [("u", "ev")] * 8 + [("u", "no")] * 25
    data = pd.DataFrame(rows, columns=["expo", "res"])
    out = go("odds_ratio", data, {"exposure": "expo", "outcome": "res"})
    assert out.result["odds_ratio"] == pytest.approx(6.25)
    # relabel so that the sort order of the levels is reversed: the same table must give the same answer
    flipped = data.replace({"expo": {"e": "z", "u": "a"}, "res": {"ev": "z", "no": "a"}})
    out = go("odds_ratio", flipped, {"exposure": "expo", "outcome": "res"})
    assert out.result["odds_ratio"] == pytest.approx(6.25) and out.result["risk_ratio"] == pytest.approx(2.75)
    reference = ct.Table2x2(np.array([[20, 10], [8, 25]]))
    medidas = table(out, "Medidas de asociación")
    assert (value(medidas[0][2]), value(medidas[0][3])) == pytest.approx(reference.oddsratio_confint())
    assert out.result["fisher_p"] == pytest.approx(stats.fisher_exact([[20, 10], [8, 25]]).pvalue)


def test_cmh_matches_statsmodels(frame):
    out = go("cmh", frame, {"exposure": "h", "outcome": "b", "stratum": "g"})
    tables = []
    for _, part in frame.groupby("g"):
        tables.append(pd.crosstab(part["h"], part["b"]).iloc[::-1, ::-1].to_numpy())
    reference = ct.StratifiedTable(tables)
    assert out.result["pooled_or"] == pytest.approx(reference.oddsratio_pooled)
    assert out.result["cmh_p"] == pytest.approx(reference.test_null_odds(correction=False).pvalue)
    assert out.result["breslow_day_p"] == pytest.approx(reference.test_equal_odds(adjust=False).pvalue)
    with pytest.raises(InvalidInputError, match="2 estratos"):
        go("cmh", frame.assign(k="solo"), {"exposure": "h", "outcome": "b", "stratum": "k"})


# ------------------------------------------------------------------ agreement
def test_cohens_kappa_of_a_known_table():
    rows = [("a", "a")] * 20 + [("a", "b")] * 5 + [("b", "a")] * 10 + [("b", "b")] * 15
    data = pd.DataFrame(rows, columns=["r1", "r2"])
    out = go("kappa", data, {"raters": ["r1", "r2"]})
    assert out.result["kappa"] == pytest.approx(0.4) and out.result["agreement"] == pytest.approx(35 / 50)
    ordinal = go("kappa", standard_frame(), {"raters": ["r1", "r2"]}, {"weights": "quadratic"})
    plain = go("kappa", standard_frame(), {"raters": ["r1", "r2"]})
    assert ordinal.result["kappa"] > plain.result["kappa"]  # near misses count for something when ordered


def test_fleiss_kappa_of_the_wikipedia_example():
    counts = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]  # fmt: skip
    ratings = [[category + 1 for category, n in enumerate(row) for _ in range(n)] for row in counts]
    data = pd.DataFrame(ratings, columns=[f"juez{i}" for i in range(14)])
    out = go("kappa", data, {"raters": list(data.columns)})
    assert out.result["kappa"] == pytest.approx(0.210, abs=0.001)


def test_intraclass_correlations_of_shrout_and_fleiss():
    scores = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float
    )
    results = {r["label"]: r for r in icc_table(scores)}
    expected = {
        "ICC(1,1)": 0.166,
        "ICC(2,1)": 0.290,
        "ICC(3,1)": 0.715,
        "ICC(1,k)": 0.443,
        "ICC(2,k)": 0.620,
        "ICC(3,k)": 0.909,
    }
    for label, icc in expected.items():
        assert results[label]["icc"] == pytest.approx(icc, abs=0.001), label
    assert results["ICC(3,1)"]["p"] == pytest.approx(stats.f.sf(results["ICC(3,1)"]["f"], 5, 15))
    assert results["ICC(3,1)"]["low"] < 0.715 < results["ICC(3,1)"]["high"]
    page = go("icc", pd.DataFrame(scores, columns=list("abcd")), {"raters": list("abcd")})
    assert page.result["icc"]["ICC(2,1)"] == pytest.approx(0.290, abs=0.001)


# ------------------------------------------------------------------ equivalence and multivariate
def test_tost_matches_statsmodels(frame):
    out = go("tost", frame, {"var": "y", "group": "h"}, {"design": "two", "low": -2, "upp": 2})
    a, b = (g["y"].to_numpy() for _, g in frame.groupby("h"))
    reference = ttost_ind(a, b, -2, 2, usevar="unequal")
    assert out.result["p"] == pytest.approx(reference[0])
    paired = go("tost", frame, {"var": "x1", "var2": "x2"}, {"design": "paired", "low": -6, "upp": 6})
    assert paired.result["p"] == pytest.approx(ttost_paired(frame["x1"], frame["x2"], -6, 6)[0])
    assert (
        paired.result["equivalent"] is True
        and go("tost", frame, {"var": "x1", "var2": "x2"}, {"design": "paired", "low": 1, "upp": 3}).result[
            "equivalent"
        ]
        is False
    )
    with pytest.raises(InvalidInputError, match="menor"):
        go("tost", frame, {"var": "y", "group": "h"}, {"low": 2, "upp": -2})
    with pytest.raises(InvalidInputError, match="grupos"):
        go("tost", frame, {"var": "y"}, {"design": "two"})


def test_hotelling_matches_manova(frame):
    out = go("hotelling", frame, {"vars": ["x1", "x2", "x3"], "group": "h"})
    fit = MANOVA.from_formula("x1 + x2 + x3 ~ C(h)", data=frame).mv_test()
    wilks_p = fit.results["C(h)"]["stat"].loc["Wilks' lambda", "Pr > F"]
    assert out.result["p"] == pytest.approx(wilks_p)
    x = frame[["x1", "x2", "x3"]].to_numpy()
    n = len(x)
    diff = x.mean(axis=0) - np.array([10.0, 5.0, 5.0])
    t2 = n * diff @ np.linalg.solve(np.cov(x, rowvar=False), diff)
    one = go("hotelling", frame, {"vars": ["x1", "x2", "x3"]}, {"design": "one", "mu0": "10, 5, 5"})
    assert one.result["t2"] == pytest.approx(t2)
    with pytest.raises(InvalidInputError, match="3 valores"):
        go("hotelling", frame, {"vars": ["x1", "x2", "x3"]}, {"design": "one", "mu0": "1, 2"})


def test_partial_correlation_matches_the_closed_form(frame):
    out = go("partial_corr", frame, {"x": "x1", "y": "y", "covars": ["x2"]})
    r = frame[["x1", "y", "x2"]].corr()
    expected = (r.at["x1", "y"] - r.at["x1", "x2"] * r.at["y", "x2"]) / np.sqrt(
        (1 - r.at["x1", "x2"] ** 2) * (1 - r.at["y", "x2"] ** 2)
    )
    assert out.result["partial"] == pytest.approx(expected)
    n = len(frame)
    t = expected * np.sqrt((n - 3) / (1 - expected**2))
    assert out.result["p"] == pytest.approx(2 * stats.t.sf(abs(t), n - 3))
    spearman = go("partial_corr", frame, {"x": "x1", "y": "y", "covars": ["x2", "x3"]}, {"method": "spearman"})
    ranks = frame[["x1", "y", "x2", "x3"]].rank()
    design = np.column_stack([np.ones(n), ranks[["x2", "x3"]]])
    residuals = [
        ranks[c].to_numpy() - design @ np.linalg.lstsq(design, ranks[c].to_numpy(), rcond=None)[0] for c in ("x1", "y")
    ]
    assert spearman.result["partial"] == pytest.approx(np.corrcoef(*residuals)[0, 1])


def test_one_variable_confidence_intervals(frame):
    out = go("ci_one", frame, {"var": "y"}, {"confidence": 90})
    rows = table(out, "Intervalos de confianza")
    low, high = stats.t.interval(0.9, len(frame) - 1, loc=frame["y"].mean(), scale=stats.sem(frame["y"]))
    assert (value(rows[0][2]), value(rows[0][3])) == pytest.approx((low, high))
    assert value(rows[1][2]) < frame["y"].std() < value(rows[1][3])
    assert value(rows[3][2]) <= frame["y"].median() <= value(rows[3][3])
