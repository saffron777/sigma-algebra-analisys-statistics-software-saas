"""Bootstrap, permutation, Monte Carlo and random data: reproducible with a seed and consistent with theory."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def resample(analysis_id, frame, roles=None, params=None):
    return run("remuestreo", analysis_id, frame, roles, params)


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_matches_scipy_with_the_same_seed(frame):
    out = resample(
        "bootstrap", frame, {"var": "y"}, {"statistic": "median", "resamples": 3000, "seed": 7, "method": "percentile"}
    )
    reference = stats.bootstrap(
        (frame["y"].to_numpy(),),
        np.median,
        n_resamples=3000,
        confidence_level=0.95,
        method="percentile",
        vectorized=True,
        random_state=np.random.default_rng(7),
    )
    assert out.result["low"] == pytest.approx(reference.confidence_interval.low)
    assert out.result["high"] == pytest.approx(reference.confidence_interval.high)
    assert out.result["se"] == pytest.approx(reference.standard_error)
    again = resample(
        "bootstrap", frame, {"var": "y"}, {"statistic": "median", "resamples": 3000, "seed": 7, "method": "percentile"}
    )
    assert again.result == out.result
    other = resample(
        "bootstrap", frame, {"var": "y"}, {"statistic": "median", "resamples": 3000, "seed": 8, "method": "percentile"}
    )
    assert other.result["se"] != out.result["se"]  # the interval limits of a median are sample values and may coincide


def test_bootstrap_of_the_mean_agrees_with_the_t_interval(frame):
    out = resample("bootstrap", frame, {"var": "y"}, {"statistic": "mean", "resamples": 5000, "seed": 3})
    low, high = stats.t.interval(0.95, len(frame) - 1, loc=frame["y"].mean(), scale=stats.sem(frame["y"]))
    assert out.result["low"] == pytest.approx(low, abs=0.08) and out.result["high"] == pytest.approx(high, abs=0.08)
    assert out.result["se"] == pytest.approx(stats.sem(frame["y"]), rel=0.1)
    assert any(b["type"] == "table" and b["title"].startswith("Comparación") for b in out.blocks)


def test_bootstrap_of_other_statistics(frame):
    corr = resample("bootstrap", frame, {"var": "y", "var2": "x1"}, {"statistic": "correlation", "resamples": 2000})
    r = frame["y"].corr(frame["x1"])
    assert corr.result["estimate"] == pytest.approx(r) and corr.result["low"] < r < corr.result["high"]
    diff = resample("bootstrap", frame, {"var": "y", "group": "h"}, {"statistic": "mean_diff", "resamples": 2000})
    a, b = (g["y"] for _, g in frame.groupby("h"))
    assert diff.result["estimate"] == pytest.approx(a.mean() - b.mean())
    percentile = resample(
        "bootstrap", frame, {"var": "y"}, {"statistic": "percentile", "percentile": 90, "resamples": 1000}
    )
    assert percentile.result["estimate"] == pytest.approx(np.percentile(frame["y"], 90))
    for stat in ("std", "trimmed", "iqr"):
        assert (
            resample("bootstrap", frame, {"var": "y"}, {"statistic": stat, "resamples": 500}).result["low"]
            < frame["y"].max()
        )


def test_bootstrap_checks_what_it_needs(frame):
    with pytest.raises(InvalidInputError, match="segunda columna"):
        resample("bootstrap", frame, {"var": "y"}, {"statistic": "correlation"})
    with pytest.raises(InvalidInputError, match="grupos"):
        resample("bootstrap", frame, {"var": "y"}, {"statistic": "mean_diff"})
    with pytest.raises(InvalidInputError, match="2 categorías"):
        resample("bootstrap", frame, {"var": "y", "group": "g"}, {"statistic": "mean_diff"})
    big = pd.DataFrame({"v": np.arange(1000.0)})
    with pytest.raises(InvalidInputError, match="demasiados"):
        resample("bootstrap", big, {"var": "v"}, {"resamples": 200000})


# ------------------------------------------------------------------ permutation
def test_permutation_matches_scipy_and_the_t_test(frame):
    params = {"design": "mean_diff", "resamples": 4999, "seed": 5}
    out = resample("permutation", frame, {"var": "y", "group": "h"}, params)
    a, b = (g["y"].to_numpy() for _, g in frame.groupby("h"))
    reference = stats.permutation_test(
        (a, b),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        permutation_type="independent",
        vectorized=True,
        n_resamples=4999,
        random_state=np.random.default_rng(5),
    )
    assert out.result["observed"] == pytest.approx(reference.statistic) and out.result["p"] == pytest.approx(
        reference.pvalue
    )
    welch = stats.ttest_ind(a, b, equal_var=False).pvalue
    assert out.result["p"] == pytest.approx(welch, abs=0.05)


def test_permutation_finds_real_effects_and_ignores_noise(frame):
    paired = resample("permutation", frame, {"var": "x1", "var2": "x2"}, {"design": "paired", "resamples": 2000})
    assert paired.result["p"] < 0.01  # x1 averages 10 and x2 averages 5
    corr = resample("permutation", frame, {"var": "y", "var2": "x1"}, {"design": "correlation", "resamples": 2000})
    assert corr.result["p"] < 0.01 and corr.result["observed"] == pytest.approx(frame["y"].corr(frame["x1"]))
    noise = resample("permutation", frame, {"var": "x3", "var2": "x2"}, {"design": "correlation", "resamples": 2000})
    assert noise.result["p"] > 0.01
    median = resample("permutation", frame, {"var": "y", "group": "h"}, {"design": "median_diff", "resamples": 1000})
    assert median.result["permutations"] == 1000 or median.result["permutations"] > 900


def test_permutation_is_exact_for_tiny_samples():
    data = pd.DataFrame({"v": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0], "g": list("aaabbb")})
    out = resample("permutation", data, {"var": "v", "group": "g"}, {"design": "mean_diff", "resamples": 999})
    assert out.result["p"] == pytest.approx(2 / 20)  # 20 ways to split 6 into 3+3, two of them as extreme as observed
    assert out.result["permutations"] == 20


# ------------------------------------------------------------------ Monte Carlo
def test_the_sample_mean_follows_the_central_limit_theorem():
    out = resample(
        "monte_carlo", None, params={"distribution": "expon", "a": 2.0, "n": 40, "replications": 20000, "seed": 1}
    )
    assert out.result["mean"] == pytest.approx(2.0, abs=0.02)
    assert out.result["se"] == pytest.approx(2.0 / np.sqrt(40), rel=0.03)
    skew = resample(
        "monte_carlo", None, params={"distribution": "expon", "a": 2.0, "n": 2, "replications": 20000, "seed": 1}
    )
    assert skew.result["se"] > out.result["se"]


def test_other_sampling_distributions():
    variance = resample(
        "monte_carlo",
        None,
        params={
            "distribution": "norm",
            "a": 0,
            "b": 3,
            "n": 20,
            "statistic": "variance",
            "replications": 20000,
            "seed": 2,
        },
    )
    assert variance.result["mean"] == pytest.approx(9.0, rel=0.03)  # the sample variance is unbiased
    maximum = resample(
        "monte_carlo",
        None,
        params={
            "distribution": "uniform",
            "a": 0,
            "b": 1,
            "n": 9,
            "statistic": "max",
            "replications": 20000,
            "seed": 2,
        },
    )
    assert maximum.result["mean"] == pytest.approx(9 / 10, abs=0.005)  # E[max] = n / (n + 1)
    same = resample(
        "monte_carlo",
        None,
        params={
            "distribution": "uniform",
            "a": 0,
            "b": 1,
            "n": 9,
            "statistic": "max",
            "replications": 20000,
            "seed": 2,
        },
    )
    assert same.result == maximum.result
    for params, fragment in (
        ({"distribution": "uniform", "a": 5, "b": 1}, "parámetros"),
        ({"distribution": "binom", "a": 10, "b": 1.5}, "parámetros"),
        ({"distribution": "expon", "a": -1}, "parámetros"),
    ):
        with pytest.raises(InvalidInputError, match=fragment):
            resample("monte_carlo", None, params=params)
    with pytest.raises(InvalidInputError, match="demasiados"):
        resample("monte_carlo", None, params={"n": 100000, "replications": 100000})


# ------------------------------------------------------------------ random data
def test_random_data_has_the_requested_shape_and_moments():
    out = resample(
        "random_data",
        None,
        params={"distribution": "norm", "a": 10, "b": 2, "n": 5000, "columns": 3, "seed": 4, "prefix": "z"},
    )
    data = offered(out)
    assert list(data.columns) == ["z1", "z2", "z3"] and len(data) == 5000
    assert data.mean().to_numpy() == pytest.approx([10, 10, 10], abs=0.15) and data.std().to_numpy() == pytest.approx(
        [2, 2, 2], abs=0.15
    )
    assert not data["z1"].equals(data["z2"])
    again = offered(
        resample(
            "random_data",
            None,
            params={"distribution": "norm", "a": 10, "b": 2, "n": 5000, "columns": 3, "seed": 4, "prefix": "z"},
        )
    )
    pd.testing.assert_frame_equal(data, again)
    counts = offered(resample("random_data", None, params={"distribution": "poisson", "a": 3, "n": 4000}))
    assert counts["x1"].mean() == pytest.approx(3, abs=0.15) and counts["x1"].dtype.kind == "i"
    dice = offered(resample("random_data", None, params={"distribution": "randint", "a": 1, "b": 6, "n": 3000}))
    assert set(dice["x1"].unique()) == {1, 2, 3, 4, 5, 6}
