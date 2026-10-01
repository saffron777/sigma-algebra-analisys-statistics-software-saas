import math

import pytest
from scipy import stats

from sigma.domain.distributions import REGISTRY, get_distribution, list_distributions
from sigma.domain.distributions.probability import (
    critical_values,
    evaluate_at,
    moments,
    p_value,
    probability_between,
    quantile,
)
from sigma.domain.errors import InvalidInputError


def test_registry_has_fifteen_distributions():
    assert len(list_distributions()) == 15
    for spec in list_distributions():
        dist = spec.build(spec.defaults())  # every default parameter set must be valid
        assert math.isfinite(float(dist.mean()))


def test_standard_normal_critical_value_is_1_96():
    dist = get_distribution("normal").build({"mu": 0, "sigma": 1})
    low, high = critical_values(dist, 0.05, "two-sided")
    assert high == pytest.approx(1.959964, abs=1e-6)
    assert low == pytest.approx(-1.959964, abs=1e-6)


def test_t_critical_value_df10():
    dist = get_distribution("t").build({"df": 10})
    assert critical_values(dist, 0.05, "right")[0] == pytest.approx(1.812461, abs=1e-6)
    assert critical_values(dist, 0.05, "two-sided")[1] == pytest.approx(2.228139, abs=1e-6)


def test_two_sided_p_value_matches_scipy():
    dist = stats.norm()
    assert p_value(dist, 2.0, "two-sided") == pytest.approx(2 * stats.norm.sf(2.0))
    assert p_value(dist, 2.0, "right") == pytest.approx(stats.norm.sf(2.0))
    assert p_value(dist, 2.0, "left") == pytest.approx(stats.norm.cdf(2.0))


def test_discrete_right_tail_is_inclusive():
    dist = stats.binom(10, 0.5)
    assert p_value(dist, 8, "right", discrete=True) == pytest.approx(stats.binom.sf(7, 10, 0.5))


def test_evaluate_and_between_binomial():
    values = evaluate_at("binomial", {"n": 10, "p": 0.5}, 5)
    assert values["density"] == pytest.approx(0.24609375)
    assert probability_between("binomial", {"n": 10, "p": 0.5}, 3, 5) == pytest.approx(
        stats.binom.pmf([3, 4, 5], 10, 0.5).sum()
    )


def test_quantile_roundtrip():
    q = quantile("chi2", {"df": 5}, 0.95)
    assert q == pytest.approx(11.0705, abs=1e-4)


def test_moments_of_exponential():
    result = moments(get_distribution("exponential").build({"rate": 2.0}))
    assert result["mean"] == pytest.approx(0.5)
    assert result["variance"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("key", "params"),
    [
        ("normal", {"mu": 0, "sigma": -1}),
        ("binomial", {"n": 5.5, "p": 0.5}),
        ("beta", {"alpha": 1}),
        ("uniform", {"a": 3, "b": 1}),
        ("hypergeometric", {"population": 10, "successes": 20, "draws": 3}),
    ],
)
def test_invalid_parameters_raise(key, params):
    with pytest.raises(InvalidInputError):
        REGISTRY[key].build(params)
