import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sigma.domain.errors import InvalidInputError
from sigma.domain.statistics.hypothesis import list_tests, run_hypothesis_test


def run(test_id, frame=None, roles=None, params=None, summary=None, alpha=0.05, tail="two-sided"):
    return run_hypothesis_test(
        test_id, frame=frame, roles=roles or {}, params=params or {}, summary=summary, alpha=alpha, tail=tail
    )


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 30), rng.normal(11.5, 3, 25)
    frame = pd.DataFrame({"valor": np.r_[a, b], "grupo": ["A"] * 30 + ["B"] * 25})
    return frame, a, b


def test_one_sample_t_matches_scipy(sales_frame):
    result = run("t_one_sample", sales_frame, {"x": "precio"}, {"mu0": 48})
    expected = stats.ttest_1samp(sales_frame["precio"], 48)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.df == sales_frame["precio"].count() - 1
    ci = expected.confidence_interval(0.95)
    assert result.confidence_interval == pytest.approx([ci.low, ci.high])


@pytest.mark.parametrize("tail", ["two-sided", "left", "right"])
def test_welch_matches_scipy(two_groups, tail):
    frame, a, b = two_groups
    result = run("t_two_sample", frame, {"x": "valor", "group": "grupo"}, {"variance_assumption": "welch"}, tail=tail)
    alt = {"two-sided": "two-sided", "left": "less", "right": "greater"}[tail]
    expected = stats.ttest_ind(a, b, equal_var=False, alternative=alt)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.df == pytest.approx(expected.df)


def test_pooled_t_matches_scipy_and_summary_mode(two_groups):
    frame, a, b = two_groups
    result = run("t_two_sample", frame, {"x": "valor", "group": "grupo"}, {"variance_assumption": "pooled"})
    expected = stats.ttest_ind(a, b, equal_var=True)
    assert result.p_value == pytest.approx(expected.pvalue)
    from_summary = run(
        "t_two_sample",
        params={"variance_assumption": "pooled"},
        summary={"n1": 30, "mean1": a.mean(), "sd1": a.std(ddof=1), "n2": 25, "mean2": b.mean(), "sd2": b.std(ddof=1)},
    )
    assert from_summary.statistic == pytest.approx(result.statistic)


def test_paired_t_matches_scipy():
    rng = np.random.default_rng(3)
    before = rng.normal(100, 10, 20)
    after = before + rng.normal(2, 3, 20)
    frame = pd.DataFrame({"antes": before, "despues": after})
    result = run("t_paired", frame, {"x": "antes", "y": "despues"})
    expected = stats.ttest_rel(before, after)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)


def test_z_mean_known_value():
    result = run("z_mean", params={"mu0": 100, "sigma": 15}, summary={"n": 36, "mean": 105})
    assert result.statistic == pytest.approx(2.0)
    assert result.p_value == pytest.approx(0.0455003, abs=1e-6)
    assert result.reject_null is True
    assert result.critical_values == pytest.approx([-1.959964, 1.959964], abs=1e-5)


def test_one_proportion_z():
    result = run("z_prop_one", params={"p0": 0.5}, summary={"successes": 60, "n": 100})
    assert result.statistic == pytest.approx(2.0)
    assert result.p_value == pytest.approx(0.0455003, abs=1e-6)


def test_two_proportions_from_data(sales_frame):
    result = run("z_prop_two", sales_frame, {"x": "comprador", "group": "canal"})
    assert 0 <= result.p_value <= 1
    assert "p combinada" in result.sample_info


def test_binomial_exact_matches_scipy():
    result = run("binomial_exact", params={"p0": 0.5}, summary={"successes": 8, "n": 10}, tail="right")
    assert result.p_value == pytest.approx(stats.binomtest(8, 10, 0.5, alternative="greater").pvalue)


def test_variance_tests():
    chi = run("chi2_variance", params={"variance0": 4}, summary={"n": 25, "sd": 2.5})
    assert chi.statistic == pytest.approx(24 * 6.25 / 4)
    f = run("f_variance", summary={"n1": 10, "sd1": 3, "n2": 12, "sd2": 2})
    assert f.statistic == pytest.approx(9 / 4)
    assert f.df == [9, 11]


def test_chi2_independence_matches_scipy():
    table = [[20, 30], [30, 20]]
    result = run("chi2_independence", summary={"table": table})
    expected = stats.chi2_contingency(table, correction=False)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.effect_size == pytest.approx(0.2)  # Cramér's V of a balanced 2x2


def test_chi2_independence_from_dataset(sales_frame):
    result = run("chi2_independence", sales_frame, {"row": "region", "col": "canal"})
    assert result.df == 2
    assert len(result.tables) == 2


def test_chi2_goodness_uniform_default():
    result = run("chi2_goodness", summary={"observed": "20, 30, 50"}, params={})
    expected = stats.chisquare([20, 30, 50])
    assert result.statistic == pytest.approx(expected.statistic)


def test_fisher_matches_scipy():
    result = run("fisher_exact", summary={"table": "3 1; 1 3"})
    assert result.p_value == pytest.approx(stats.fisher_exact([[3, 1], [1, 3]])[1])


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_correlation_matches_scipy(sales_frame, method):
    result = run("correlation", sales_frame, {"x": "precio", "y": "ventas"}, {"method": method})
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    expected = func(sales_frame["precio"], sales_frame["ventas"])
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)


def test_mann_whitney_matches_scipy(two_groups):
    frame, a, b = two_groups
    result = run("mann_whitney", frame, {"x": "valor", "group": "grupo"})
    expected = stats.mannwhitneyu(a, b)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)


def test_wilcoxon_and_shapiro(sales_frame):
    w = run("wilcoxon", sales_frame, {"x": "precio", "y": "competencia"})
    assert 0 <= w.p_value <= 1
    s = run("shapiro", sales_frame, {"x": "precio"})
    assert s.p_value == pytest.approx(stats.shapiro(sales_frame["precio"]).pvalue)


def test_input_validation(sales_frame):
    with pytest.raises(InvalidInputError):
        run("t_two_sample", sales_frame, {"x": "ventas", "group": "region"})  # 3 levels
    with pytest.raises(InvalidInputError):
        run("t_one_sample", sales_frame, {"x": "region"})  # not numeric
    with pytest.raises(InvalidInputError):
        run("t_one_sample", sales_frame, {"x": "precio"}, alpha=1.5)
    with pytest.raises(InvalidInputError):
        run("nope", sales_frame)


def test_every_catalogue_entry_is_well_formed():
    ids = [spec.id for spec in list_tests()]
    assert len(ids) == len(set(ids)) == 16
