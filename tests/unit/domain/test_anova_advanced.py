"""Advanced ANOVA against statsmodels / scipy and the textbook formulas."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from sigma.domain.analysis.anova_advanced import box_m, helmert, sphericity
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def anova(analysis_id, frame, roles=None, params=None):
    return run("anova_avanzado", analysis_id, frame, roles, params)


def table(output, title_start):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"].startswith(title_start))["rows"]


def value(cell):
    return getattr(cell, "value", cell)


@pytest.fixture(scope="module")
def balanced():
    rng = np.random.default_rng(41)
    rows = []
    for a in ("a1", "a2", "a3"):
        for b in ("b1", "b2"):
            effect = (
                {"a1": 0, "a2": 1.5, "a3": 3}[a] + (1.0 if (a == "a3" and b == "b2") else 0) + {"b1": 0, "b2": 0.5}[b]
            )
            rows += [(a, b, effect + rng.normal(0, 1)) for _ in range(12)]
    return pd.DataFrame(rows, columns=["A", "B", "y"])


# ------------------------------------------------------------------ N-way ANOVA and ANCOVA
def test_two_way_anova_matches_statsmodels_for_every_ss_type(balanced):
    for label, typ in (("I", 1), ("II", 2)):
        out = anova(
            "glm_anova", balanced, {"response": "y", "factors": ["A", "B"]}, {"ss_type": label, "interactions": "full"}
        )
        reference = anova_lm(smf.ols("y ~ C(A) * C(B)", balanced).fit(), typ=typ)
        rows = {r[0]: r for r in table(out, "Análisis de varianza")}
        assert rows["A"][2] == pytest.approx(reference.loc["C(A)", "sum_sq"]) and value(rows["A"][5]) == pytest.approx(
            reference.loc["C(A)", "PR(>F)"]
        )
        assert rows["A × B"][4] == pytest.approx(reference.loc["C(A):C(B)", "F"])
    third = anova("glm_anova", balanced, {"response": "y", "factors": ["A", "B"]}, {"ss_type": "III"})
    reference = anova_lm(smf.ols("y ~ C(A, Sum) * C(B, Sum)", balanced).fit(), typ=3)
    rows = {r[0]: r for r in table(third, "Análisis de varianza")}
    assert rows["B"][2] == pytest.approx(reference.loc["C(B, Sum)", "sum_sq"]) and rows["Error"][1] == 66
    # partial eta squared = SS_effect / (SS_effect + SS_error)
    assert rows["A"][6] == pytest.approx(rows["A"][2] / (rows["A"][2] + rows["Error"][2]))


def test_unbalanced_data_gives_different_type_i_and_type_iii_results(frame):
    uneven = frame.iloc[:200]
    one = anova("glm_anova", uneven, {"response": "y", "factors": ["g", "h"]}, {"ss_type": "I"})
    three = anova("glm_anova", uneven, {"response": "y", "factors": ["g", "h"]}, {"ss_type": "III"})
    first = {r[0]: r for r in table(one, "Análisis de varianza")}
    last = {r[0]: r for r in table(three, "Análisis de varianza")}
    assert first["h"][2] != pytest.approx(last["h"][2])
    assert first["g"][2] == pytest.approx(last["g"][2], rel=0.2)


def test_marginal_means_of_a_balanced_design_are_the_averages(balanced):
    out = anova("glm_anova", balanced, {"response": "y", "factors": ["A", "B"]}, {})
    rows = table(out, "Medias marginales estimadas de «A»")
    expected = balanced.groupby("A")["y"].mean()
    assert [r[1] for r in rows] == pytest.approx(expected.tolist())
    crit = stats.t.ppf(0.975, 66)
    mse = smf.ols("y ~ C(A) * C(B)", balanced).fit().scale
    assert rows[0][2] == pytest.approx(np.sqrt(mse / 24)) and rows[0][4] - rows[0][1] == pytest.approx(
        crit * rows[0][2]
    )


def test_ancova_adjusts_the_means_and_checks_the_slopes(frame):
    out = anova("glm_anova", frame, {"response": "y", "factors": ["g"], "covariates": ["x1"]}, {"ss_type": "III"})
    fit = smf.ols("y ~ C(g) + x1", frame).fit()
    adjusted = {r[0]: r[1] for r in table(out, "Medias marginales estimadas de «g»")}
    for level in ("A", "B", "C"):
        grid = pd.DataFrame({"g": [level], "x1": [frame["x1"].mean()]})
        assert adjusted[level] == pytest.approx(float(fit.predict(grid).iloc[0]), rel=1e-6)
    slopes = next(r for r in table(out, "Comprobación de supuestos") if r[0].startswith("Homogeneidad de pendientes"))
    reference = anova_lm(fit, smf.ols("y ~ C(g) * x1", frame).fit())
    assert value(slopes[2]) == pytest.approx(reference.iloc[1]["Pr(>F)"])
    assert out.result["effects"]["x1"] < 0.001  # the covariate matters


def test_anova_input_checks(frame):
    with pytest.raises(InvalidInputError, match="máximo 4"):
        anova(
            "glm_anova",
            frame.assign(f1=frame["g"], f2=frame["g"], f3=frame["h"], f4=frame["h"]),
            {"response": "y", "factors": ["g", "h", "f1", "f2", "f3"]},
        )
    with pytest.raises(InvalidInputError, match="tantos parámetros"):
        anova(
            "glm_anova", pd.DataFrame({"y": np.arange(8.0), "f": list("abcdefgh")}), {"response": "y", "factors": ["f"]}
        )


# ------------------------------------------------------------------ repeated measures
def test_repeated_measures_match_statsmodels_and_the_sphericity_formulas(frame):
    names = ["r1", "r2", "r3"]
    out = anova("rm_anova", frame, {"vars": names}, {})
    long = frame[names].rename_axis("s").reset_index().melt(id_vars="s", var_name="c", value_name="y")
    reference = AnovaRM(long, "y", "s", within=["c"]).fit().anova_table
    assert out.result["f"] == pytest.approx(reference.loc["c", "F Value"]) and out.result["p"] == pytest.approx(
        reference.loc["c", "Pr > F"]
    )
    x = frame[names].to_numpy(float)
    k = 3
    # Greenhouse-Geisser from the original double-centred covariance formula
    s = np.cov(x, rowvar=False)
    centred = s - s.mean(axis=0, keepdims=True) - s.mean(axis=1, keepdims=True) + s.mean()
    eps_original = np.trace(centred) ** 2 / ((k - 1) * (centred**2).sum())
    assert out.result["epsilon_gg"] == pytest.approx(eps_original)
    assert 1 / (k - 1) <= out.result["epsilon_gg"] <= out.result["epsilon_hf"] <= 1


def test_mauchly_test_detects_unequal_variances_of_differences():
    rng = np.random.default_rng(52)
    n = 400
    subject = rng.normal(0, 2, n)
    spherical = np.column_stack([subject + rng.normal(0, 1, n) for _ in range(4)])
    assert sphericity(spherical)["p"] > 0.01 and sphericity(spherical)["gg"] > 0.9
    broken = spherical.copy()
    broken[:, 3] = broken[:, 3] * 4 + broken[:, 2] * 3
    result = sphericity(broken)
    assert result["p"] < 0.001 and result["gg"] < 0.85
    assert np.allclose(helmert(5) @ helmert(5).T, np.eye(4)) and np.allclose(helmert(5).sum(axis=1), 0)


def test_repeated_measures_pairwise_comparisons_are_adjusted(frame):
    out = anova("rm_anova", frame, {"vars": ["r1", "r2", "r3"]}, {"adjust": "bonferroni"})
    rows = table(out, "Comparaciones por pares")
    raw = [value(r[4]) for r in rows]
    assert [value(r[5]) for r in rows] == pytest.approx(list(multipletests(raw, method="bonferroni")[1]))
    assert rows[0][2] == pytest.approx((frame["r1"] - frame["r2"]).mean())
    two = anova("rm_anova", frame, {"vars": ["r1", "r2"]}, {})
    assert two.result["p"] == pytest.approx(stats.ttest_rel(frame["r1"], frame["r2"]).pvalue)
    with pytest.raises(InvalidInputError, match="3 sujetos"):
        anova("rm_anova", frame.head(2), {"vars": ["r1", "r2"]})


# ------------------------------------------------------------------ MANOVA
def test_manova_matches_statsmodels_and_box_m_matches_the_definition(frame):
    out = anova("manova", frame, {"responses": ["x1", "x2", "x3"], "factors": ["g"]}, {})
    reference = MANOVA.from_formula("x1 + x2 + x3 ~ C(g, Sum)", frame).mv_test().results["C(g, Sum)"]["stat"]
    wilks = reference.loc["Wilks' lambda", "Pr > F"]
    assert out.result["wilks_p"]["g"] == pytest.approx(wilks)
    rows = [r for r in table(out, "Pruebas multivariadas") if r[1].startswith("Pillai")]
    assert rows[0][2] == pytest.approx(reference.loc["Pillai's trace", "Value"])
    univariate = {r[0]: r for r in table(out, "Pruebas univariadas")}
    fit = smf.ols("x1 ~ C(g, Sum)", frame).fit()
    assert univariate["x1"][3] == pytest.approx(anova_lm(fit, typ=3).loc["C(g, Sum)", "F"])
    groups = [part[["x1", "x2", "x3"]].to_numpy() for _, part in frame.groupby("g")]
    chi2, p, df = box_m(groups)
    sizes = np.array([len(x) for x in groups])
    pooled = sum((len(x) - 1) * np.cov(x, rowvar=False) for x in groups) / (sizes.sum() - 3)
    m = (sizes.sum() - 3) * np.linalg.slogdet(pooled)[1] - sum(
        (len(x) - 1) * np.linalg.slogdet(np.cov(x, rowvar=False))[1] for x in groups
    )
    assert df == 12 and 0 <= chi2 <= m * 1.0001 and 0 <= p <= 1


def test_manova_finds_a_multivariate_effect_that_no_single_response_shows():
    rng = np.random.default_rng(61)
    n = 150
    a = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], n)
    b = rng.multivariate_normal([0.18, -0.18], [[1, 0.9], [0.9, 1]], n)  # opposed shifts against the correlation
    data = pd.DataFrame(np.vstack([a, b]), columns=["u", "v"]).assign(grupo=["a"] * n + ["b"] * n)
    out = anova("manova", data, {"responses": ["u", "v"], "factors": ["grupo"]})
    assert out.result["wilks_p"]["grupo"] < 1e-4
    univariate = {r[0]: value(r[4]) for r in table(out, "Pruebas univariadas")}
    assert (
        out.result["wilks_p"]["grupo"] < min(univariate.values()) / 100
    )  # far more convincing than any single response


# ------------------------------------------------------------------ nested
def test_nested_anova_matches_the_balanced_textbook_formulas():
    rng = np.random.default_rng(71)
    a_levels, b_levels, reps = 3, 4, 5
    rows = []
    for a in range(a_levels):
        for b in range(b_levels):
            effect = 2.0 * a + rng.normal(0, 1.5)
            rows += [(f"A{a}", f"{b + 1}", effect + rng.normal(0, 1)) for _ in range(reps)]
    data = pd.DataFrame(rows, columns=["A", "B", "y"])
    out = anova("nested", data, {"response": "y", "factor_a": "A", "factor_b": "B"}, {"b_random": True})
    y = data.assign(cell=data["A"] + data["B"]).groupby(["A", "B"])["y"]
    grand, a_mean = data["y"].mean(), data.groupby("A")["y"].mean()
    ss_a = b_levels * reps * ((a_mean - grand) ** 2).sum()
    ss_b = reps * sum(((y.mean().loc[a] - a_mean[a]) ** 2).sum() for a in a_mean.index)
    ss_e = sum(((g - g.mean()) ** 2).sum() for _, g in y)
    rows_ = {r[0]: r for r in table(out, "Análisis de varianza")}
    assert (
        rows_["A"][2] == pytest.approx(ss_a)
        and rows_["B(A)"][2] == pytest.approx(ss_b)
        and rows_["Error"][2] == pytest.approx(ss_e)
    )
    ms_a, ms_b, ms_e = ss_a / 2, ss_b / 9, ss_e / 48
    assert rows_["A"][5] == pytest.approx(ms_a / ms_b) and rows_["A"][6] == pytest.approx(stats.f.sf(ms_a / ms_b, 2, 9))
    assert rows_["B(A)"][5] == pytest.approx(ms_b / ms_e)
    components = {r[0]: r[1] for r in table(out, "Componentes de la varianza")}
    assert components["Error"] == pytest.approx(ms_e) and components["B(A)"] == pytest.approx(
        max((ms_b - ms_e) / reps, 0)
    )
    fixed = anova("nested", data, {"response": "y", "factor_a": "A", "factor_b": "B"}, {"b_random": False})
    assert {r[0]: r for r in table(fixed, "Análisis de varianza")}["A"][5] == pytest.approx(ms_a / ms_e)
    with pytest.raises(InvalidInputError, match="distinto"):
        anova("nested", data, {"response": "y", "factor_a": "A", "factor_b": "A"})


# ------------------------------------------------------------------ post hoc
def test_tukey_matches_statsmodels(frame):
    out = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "tukey"})
    reference = pairwise_tukeyhsd(frame["y"], frame["g"], alpha=0.05)
    rows = table(out, "Diferencias entre pares")
    assert len(rows) == len(reference.meandiffs) == 3
    for row, meandiff, padj, (lower, upper) in zip(
        rows, reference.meandiffs, reference.pvalues, reference.confint, strict=True
    ):
        assert row[2] == pytest.approx(-meandiff)  # statsmodels reports (second − first); here it is first − second
        assert value(row[5]) == pytest.approx(padj, abs=5e-3)
        assert (row[6], row[7]) == pytest.approx((-upper, -lower), abs=2e-3)
        assert (row[6] + row[7]) / 2 == pytest.approx(row[2])


def test_games_howell_reduces_to_welch_for_two_groups(frame):
    out = anova("posthoc", frame, {"response": "y", "factor": "h"}, {"method": "games_howell"})
    a, b = (g["y"] for _, g in frame.groupby("h"))
    assert value(table(out, "Diferencias entre pares")[0][5]) == pytest.approx(
        stats.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-4
    )
    three = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "games_howell"})
    assert three.result["comparisons"] == 3 and three.result["significant"] >= 2


def test_dunnett_and_the_p_value_adjustments(frame):
    out = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "dunnett", "control": "A"})
    samples = [g["y"].to_numpy() for _, g in frame.groupby("g")]
    reference = stats.dunnett(samples[1], samples[2], control=samples[0])
    assert [value(r[5]) for r in table(out, "Diferencias entre pares")] == pytest.approx(
        list(reference.pvalue), abs=1e-3
    )
    with pytest.raises(InvalidInputError, match="control"):
        anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "dunnett", "control": "Z"})
    bonferroni = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "bonferroni"})
    lsd = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "lsd"})
    holm = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "holm"})
    raw = [value(r[5]) for r in table(lsd, "Diferencias entre pares")]
    assert [value(r[5]) for r in table(bonferroni, "Diferencias entre pares")] == pytest.approx(
        list(multipletests(raw, method="bonferroni")[1])
    )
    assert [value(r[5]) for r in table(holm, "Diferencias entre pares")] == pytest.approx(
        list(multipletests(raw, method="holm")[1])
    )
    assert any(b["type"] == "alert" for b in lsd.blocks)  # LSD warns about the lack of adjustment


def test_scheffe_is_more_conservative_than_tukey(frame):
    scheffe = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "scheffe"})
    tukey = anova("posthoc", frame, {"response": "y", "factor": "g"}, {"method": "tukey"})
    for s_row, t_row in zip(
        table(scheffe, "Diferencias entre pares"), table(tukey, "Diferencias entre pares"), strict=True
    ):
        assert (s_row[7] - s_row[6]) >= (t_row[7] - t_row[6]) - 1e-9
    one_way = sm.stats.anova_lm(smf.ols("y ~ C(g)", frame).fit(), typ=2)
    assert one_way.loc["Residual", "df"] == 237
