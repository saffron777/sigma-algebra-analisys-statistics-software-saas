"""Probability computations shared by the distribution calculator and every test module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sigma.domain.distributions.registry import get_distribution
from sigma.domain.distributions.spec import DISCRETE, DistributionRef
from sigma.domain.errors import InvalidInputError

TAILS = ("two-sided", "left", "right")


def frozen_from_ref(ref: DistributionRef) -> Any:
    return get_distribution(ref.key).build(dict(ref.params))


def validate_tail(tail: str) -> str:
    if tail not in TAILS:
        raise InvalidInputError(f"Cola inválida «{tail}». Use: {', '.join(TAILS)}.")
    return tail


def p_value(dist: Any, statistic: float, tail: str = "two-sided", discrete: bool = False) -> float:
    """P-value of ``statistic`` under ``dist``.

    For discrete distributions the right tail is P(X >= x) (inclusive), the usual convention
    for exact tests. Two-sided p-values for asymmetric distributions use 2*min(left, right).
    """
    validate_tail(tail)
    left = float(dist.cdf(statistic))
    right = float(dist.sf(statistic - 1)) if discrete else float(dist.sf(statistic))
    if tail == "left":
        return min(1.0, left)
    if tail == "right":
        return min(1.0, right)
    return min(1.0, 2.0 * min(left, right))


def critical_values(dist: Any, alpha: float, tail: str = "two-sided") -> list[float]:
    """Boundaries of the rejection region at significance ``alpha``."""
    validate_tail(tail)
    if not 0 < alpha < 1:
        raise InvalidInputError("El nivel de significancia α debe estar entre 0 y 1.")
    if tail == "left":
        return [float(dist.ppf(alpha))]
    if tail == "right":
        return [float(dist.isf(alpha))]
    return [float(dist.ppf(alpha / 2)), float(dist.isf(alpha / 2))]


def moments(dist: Any) -> dict[str, float]:
    mean, var, skew, kurt = (float(v) for v in dist.stats(moments="mvsk"))
    return {
        "mean": mean,
        "variance": var,
        "std": math.sqrt(var) if var >= 0 else float("nan"),
        "skewness": skew,
        "excess_kurtosis": kurt,
    }


@dataclass(frozen=True)
class PlotSupport:
    """Region of the x axis worth drawing for a distribution."""

    low: float
    high: float
    discrete: bool


def plot_support(dist: Any, discrete: bool, extra_points: tuple[float, ...] = ()) -> PlotSupport:
    low = float(dist.ppf(0.0005 if not discrete else 0.0001))
    high = float(dist.isf(0.0005 if not discrete else 0.0001))
    if discrete:
        low, high = math.floor(low), math.ceil(high)
    for point in extra_points:
        if math.isfinite(point):
            span = (high - low) or 1.0
            low = min(low, point - 0.05 * span)
            high = max(high, point + 0.05 * span)
    if high <= low:
        high = low + 1
    return PlotSupport(low, high, discrete)


def evaluate_at(key: str, params: dict[str, float], x: float) -> dict[str, float | None]:
    spec = get_distribution(key)
    dist = spec.build(params)
    discrete = spec.kind == DISCRETE
    if discrete and x != int(x):
        raise InvalidInputError("En una distribución discreta, x debe ser un entero.")
    density = dist.pmf(x) if discrete else dist.pdf(x)
    return {
        "density": float(density),
        "cdf": float(dist.cdf(x)),
        "survival": float(dist.sf(x)),
        "survival_inclusive": float(dist.sf(x - 1)) if discrete else float(dist.sf(x)),
    }


def probability_between(key: str, params: dict[str, float], a: float, b: float) -> float:
    if b < a:
        raise InvalidInputError("El límite inferior no puede ser mayor que el superior.")
    spec = get_distribution(key)
    dist = spec.build(params)
    if spec.kind == DISCRETE:
        return float(dist.cdf(b) - dist.cdf(a - 1))
    return float(dist.cdf(b) - dist.cdf(a))


def quantile(key: str, params: dict[str, float], q: float) -> float:
    if not 0 < q < 1:
        raise InvalidInputError("La probabilidad debe estar estrictamente entre 0 y 1.")
    return float(get_distribution(key).build(params).ppf(q))


def curve(dist: Any, support: PlotSupport, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """x/density arrays used to draw the distribution."""
    if support.discrete:
        x = np.arange(support.low, support.high + 1)
        return x, np.asarray(dist.pmf(x), dtype=float)
    x = np.linspace(support.low, support.high, points)
    return x, np.asarray(dist.pdf(x), dtype=float)
