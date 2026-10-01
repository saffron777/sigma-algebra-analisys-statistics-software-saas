import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sigma.domain.errors import InvalidInputError
from sigma.domain.statistics.descriptive import (
    correlation_matrix,
    describe,
    fit_distributions,
    frequency_table,
    summarize_numeric,
)


def test_numeric_summary_matches_pandas_and_scipy(sales_frame):
    values = sales_frame["precio"]
    summary = summarize_numeric(values, "precio", 0.95)
    assert summary.n == 200
    assert summary.mean == pytest.approx(values.mean())
    assert summary.std == pytest.approx(values.std(ddof=1))
    assert summary.median == pytest.approx(values.median())
    assert summary.skewness == pytest.approx(stats.skew(values, bias=False))
    assert summary.kurtosis == pytest.approx(stats.kurtosis(values, bias=False))
    low, high = stats.t.interval(0.95, 199, loc=values.mean(), scale=stats.sem(values))
    assert (summary.ci_low, summary.ci_high) == pytest.approx((low, high))
    assert summary.iqr == pytest.approx(values.quantile(0.75) - values.quantile(0.25))


def test_missing_values_are_counted():
    summary = summarize_numeric(pd.Series([1.0, 2.0, None, 4.0]), "x")
    assert (summary.n, summary.n_missing) == (3, 1)


def test_empty_column_raises():
    with pytest.raises(InvalidInputError):
        summarize_numeric(pd.Series([None, None], dtype=float), "x")


def test_frequency_table_percentages():
    table = frequency_table(pd.Series(["a", "b", "a", "a", None]), "x")
    assert table.n == 4 and table.n_missing == 1
    assert table.rows[0].value == "a" and table.rows[0].percent == pytest.approx(75.0)
    assert table.rows[-1].cumulative_percent == pytest.approx(100.0)


def test_describe_with_groups_normality_and_fit(sales_frame):
    result = describe(
        sales_frame,
        ["precio", "ventas"],
        ["region"],
        group_by="canal",
        run_normality=True,
        fit_candidates=["normal", "lognormal"],
        correlation_method="pearson",
    )
    groups = {s.group for s in result.numeric}
    assert groups == {None, "Online", "Tienda"}
    assert result.fits and result.fits[0].distribution in {"normal", "lognormal"}
    assert result.correlation is not None and result.correlation.matrix[0][0] == 1.0
    assert result.normality, "expected Shapiro / D'Agostino results"


def test_distribution_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(5, 2, 2000))
    best = fit_distributions(data, "x", ["normal", "uniform"])[0]
    assert best.distribution == "normal"
    assert best.params["mu"] == pytest.approx(5, abs=0.2)
    assert best.params["sigma"] == pytest.approx(2, abs=0.2)
    assert best.ks_p_value > 0.01


def test_correlation_pvalues_match_scipy(sales_frame):
    result = correlation_matrix(sales_frame, ["precio", "ventas"], "pearson")
    expected = stats.pearsonr(sales_frame["precio"], sales_frame["ventas"])
    assert result.matrix[0][1] == pytest.approx(expected.statistic)
    assert result.p_values[0][1] == pytest.approx(expected.pvalue)


def test_non_numeric_column_is_rejected(sales_frame):
    with pytest.raises(InvalidInputError):
        describe(sales_frame, ["region"], [])
