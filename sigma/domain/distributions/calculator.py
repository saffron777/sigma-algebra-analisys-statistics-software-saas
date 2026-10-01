"""Distribution calculator: evaluates one distribution in one of several modes."""

from __future__ import annotations

from sigma.domain.distributions.probability import (
    critical_values,
    evaluate_at,
    moments,
    p_value,
    probability_between,
    quantile,
    validate_tail,
)
from sigma.domain.distributions.registry import get_distribution
from sigma.domain.distributions.spec import DISCRETE
from sigma.domain.errors import InvalidInputError
from sigma.domain.results import DistributionCalcResult

MODES = ("point", "between", "quantile", "pvalue")


def _required(value: float | None, label: str) -> float:
    if value is None:
        raise InvalidInputError(f"Falta el valor «{label}».")
    return float(value)


def calculate(
    key: str,
    params: dict[str, float],
    mode: str,
    x: float | None = None,
    a: float | None = None,
    b: float | None = None,
    q: float | None = None,
    statistic: float | None = None,
    tail: str = "two-sided",
    alpha: float = 0.05,
) -> DistributionCalcResult:
    if mode not in MODES:
        raise InvalidInputError(f"Modo de cálculo inválido «{mode}».")
    spec = get_distribution(key)
    dist = spec.build(params)
    discrete = spec.kind == DISCRETE
    result = DistributionCalcResult(
        distribution=key,
        label=spec.label,
        kind=spec.kind,
        params=dict(params),
        mode=mode,
        values={},
        moments=moments(dist),
    )
    if mode == "point":
        point = _required(x, "x")
        result.values = dict(evaluate_at(key, params, point))
        result.marker = point
    elif mode == "between":
        low, high = _required(a, "límite inferior (a)"), _required(b, "límite superior (b)")
        result.values = {"probability": probability_between(key, params, low, high)}
        result.bounds = [low, high]
    elif mode == "quantile":
        prob = _required(q, "probabilidad (p)")
        value = quantile(key, params, prob)
        result.values = {"quantile": value, "probability": prob}
        result.marker = value
    else:
        stat = _required(statistic, "estadístico")
        validate_tail(tail)
        result.values = {"p_value": p_value(dist, stat, tail, discrete)}
        result.tail, result.alpha, result.marker = tail, alpha, stat
        if not discrete:
            result.critical_values = critical_values(dist, alpha, tail)
    return result
