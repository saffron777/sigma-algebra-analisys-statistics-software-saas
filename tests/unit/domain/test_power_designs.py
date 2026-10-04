"""More power designs against closed forms, published sample sizes and a simulation."""

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.proportion import proportion_effectsize

from sigma.domain.analysis.power_designs import tost_power
from sigma.domain.errors import ComputationError, InvalidInputError
from tests.analysis_data import run


def power(analysis_id, params):
    return run("potencia", analysis_id, None, {}, params)


def test_one_proportion_matches_the_arcsine_formula():
    out = power("prop_power", {"design": "one", "p0": 0.5, "p1": 0.6})
    h = proportion_effectsize(0.6, 0.5)
    expected = math.ceil(((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / h) ** 2)
    assert out.result["n"] == expected and out.result["power"] >= 0.8
    # one less observation falls short of the target
    below = power("prop_power", {"design": "one", "p0": 0.5, "p1": 0.6, "solve_for": "power", "n": expected - 1})
    assert below.result["power"] < 0.8 <= out.result["power"]


def test_two_proportions_need_about_388_per_group():
    out = power("prop_power", {"design": "two", "p0": 0.5, "p1": 0.6})
    assert abs(out.result["n"] - 388) <= 3  # 387.4 with the pooled-variance formula
    one_sided = power("prop_power", {"design": "two", "p0": 0.5, "p1": 0.6, "alternative": "larger"})
    assert one_sided.result["n"] < out.result["n"]


def test_proportion_effect_round_trip():
    n = power("prop_power", {"design": "one", "p0": 0.5, "p1": 0.62}).result["n"]
    back = power("prop_power", {"design": "one", "p0": 0.5, "solve_for": "effect", "n": n})
    assert back.result["p1"] == pytest.approx(0.62, abs=0.005)
    with pytest.raises(InvalidInputError, match="distinta"):
        power("prop_power", {"p0": 0.5, "p1": 0.5})
    with pytest.raises(ComputationError):
        power("prop_power", {"design": "one", "p0": 0.97, "solve_for": "effect", "n": 5})


def test_anova_needs_180_observations_for_a_medium_effect_in_four_groups():
    out = power("anova_power", {"k": 4, "effect": 0.25})
    assert out.result["n_per_group"] == 45 and out.result["total"] == 180
    reference = FTestAnovaPower().power(effect_size=0.25, nobs=180, alpha=0.05, k_groups=4)
    assert out.result["power"] == pytest.approx(reference)
    effect = power("anova_power", {"k": 4, "solve_for": "effect", "n": 45})
    assert effect.result["effect_f"] == pytest.approx(0.25, abs=0.005)
    assert power("anova_power", {"k": 4, "solve_for": "power", "n": 10, "effect": 0.25}).result["power"] < 0.5


def test_correlation_needs_85_pairs_for_r_03():
    out = power("corr_power", {"r": 0.3})
    assert out.result["n"] == 85
    assert power("corr_power", {"r": -0.3}).result["n"] == 85
    assert power("corr_power", {"r": 0.3, "alternative": "larger"}).result["n"] < 85
    assert power("corr_power", {"solve_for": "effect", "n": 85}).result["r"] == pytest.approx(0.3, abs=0.01)
    assert power("corr_power", {"solve_for": "power", "n": 85, "r": 0.3}).result["power"] == pytest.approx(
        0.8, abs=0.01
    )
    with pytest.raises(InvalidInputError, match="0.999"):
        power("corr_power", {"r": 1.2})
    with pytest.raises(InvalidInputError, match="mayor que 3"):
        power("corr_power", {"solve_for": "power", "n": 3, "r": 0.3})


def test_tost_power_matches_a_simulation():
    rng = np.random.default_rng(11)
    n, sd, margin, delta, alpha = 30, 3.0, 2.0, 0.5, 0.05
    reps = 40000
    a = rng.normal(0, sd, (reps, n))
    b = rng.normal(delta, sd, (reps, n))
    diff = b.mean(axis=1) - a.mean(axis=1)
    pooled = np.sqrt((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2)
    se = pooled * math.sqrt(2 / n)
    t = stats.t.ppf(1 - alpha, 2 * n - 2)
    simulated = np.mean((diff + margin) / se > t) * 0 + np.mean(
        ((diff + margin) / se > t) & ((diff - margin) / se < -t)
    )
    assert tost_power(delta, margin, sd, n, alpha) == pytest.approx(simulated, abs=0.01)


def test_tost_sample_size_reaches_the_target_and_validates():
    out = power("tost_power", {"margin": 2.0, "difference": 0.5, "sd": 3.0})
    n = out.result["n"]
    assert tost_power(0.5, 2.0, 3.0, n, 0.05) >= 0.8 > tost_power(0.5, 2.0, 3.0, n - 1, 0.05)
    # a larger true difference (closer to the margin) needs more observations
    assert power("tost_power", {"margin": 2.0, "difference": 1.0, "sd": 3.0}).result["n"] > n
    with pytest.raises(InvalidInputError, match="dentro"):
        power("tost_power", {"margin": 2.0, "difference": 2.5, "sd": 3.0})
    with pytest.raises(ComputationError):
        power("tost_power", {"margin": 0.001, "difference": 0.0, "sd": 100.0})
