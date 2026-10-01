"""Hypothesis tests.

Every test is declared as a :class:`TestSpec`: the roles/parameters the UI must ask for, how to
turn dataset columns into test inputs (``prepare``) and how to compute the result (``compute``).
Tests that support it can also be run from summary statistics instead of raw data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.distributions import DistributionRef
from sigma.domain.distributions.probability import TAILS, critical_values, p_value
from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.domain.results import HypothesisTestResult, TableData
from sigma.domain.statistics.common import (
    check_alpha,
    label_of,
    numeric_series,
    require_columns,
)

ALL_TAILS = TAILS
RIGHT_ONLY = ("right",)


# ============================================================================ declarations
@dataclass(frozen=True)
class RoleSpec:
    name: str
    label: str
    kind: str = "numeric"  # numeric | categorical | any
    optional: bool = False


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str = "number"  # number | select | text | matrix
    default: Any = None
    options: tuple[tuple[str, str], ...] = ()
    help: str = ""
    optional: bool = False


@dataclass(frozen=True)
class Settings:
    alpha: float
    tail: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TestSpec:
    id: str
    label: str
    group: str
    description: str
    roles: tuple[RoleSpec, ...]
    params: tuple[FieldSpec, ...]
    summary_fields: tuple[FieldSpec, ...]
    tails: tuple[str, ...]
    prepare: Callable[[pd.DataFrame, dict[str, str], dict[str, Any]], dict[str, Any]]
    compute: Callable[[dict[str, Any], Settings], HypothesisTestResult]
    __test__ = False  # not a pytest class


# ============================================================================ helpers
def _number(params: dict[str, Any], name: str, label: str, default: float | None = None) -> float:
    raw = params.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InvalidInputError(f"«{label}» debe ser un número.") from None
    if not math.isfinite(value):
        raise InvalidInputError(f"«{label}» debe ser un número finito.")
    return value


def _positive(value: float, label: str) -> float:
    if value <= 0:
        raise InvalidInputError(f"«{label}» debe ser mayor que 0.")
    return value


def _count(value: Any, label: str, minimum: int = 0) -> int:
    number = float(value)
    if number != int(number) or number < minimum:
        raise InvalidInputError(f"«{label}» debe ser un entero ≥ {minimum}.")
    return int(number)


def _numeric_inputs(raw: dict[str, Any], names: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, label in names.items():
        try:
            values[key] = float(raw[key])
        except (KeyError, TypeError, ValueError):
            raise InvalidInputError(f"Falta o es inválido el valor «{label}».") from None
    return values


def _clean(frame: pd.DataFrame, name: str, minimum: int = 2) -> np.ndarray:
    values = numeric_series(frame, name).dropna().to_numpy()
    if len(values) < minimum:
        raise InvalidInputError(f"«{name}» necesita al menos {minimum} observaciones válidas.")
    return values


def _two_samples(
    frame: pd.DataFrame, roles: dict[str, str], minimum: int = 2
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return both samples from either a group column (long) or two numeric columns (wide)."""
    x_name, y_name, group = roles.get("x"), roles.get("y"), roles.get("group")
    if not x_name:
        raise InvalidInputError("Seleccione la columna numérica «x».")
    if bool(y_name) == bool(group):
        raise InvalidInputError("Indique la columna de grupos o una segunda columna numérica (solo una de las dos).")
    if group:
        require_columns(frame, [group])
        values = numeric_series(frame, x_name)
        levels = frame[group].dropna().map(label_of)
        unique = sorted(levels.unique())
        if len(unique) != 2:
            raise InvalidInputError(
                f"La columna de grupos «{group}» debe tener exactamente 2 niveles (tiene {len(unique)})."
            )
        keys = frame[group].map(lambda v: label_of(v) if pd.notna(v) else None)
        a = values[keys == unique[0]].dropna().to_numpy()
        b = values[keys == unique[1]].dropna().to_numpy()
        labels = (f"{x_name} [{unique[0]}]", f"{x_name} [{unique[1]}]")
    else:
        a, b = _clean(frame, x_name, 0), _clean(frame, y_name, 0)  # type: ignore[arg-type]
        labels = (x_name, y_name)  # type: ignore[assignment]
    if len(a) < minimum or len(b) < minimum:
        raise InvalidInputError(f"Cada grupo necesita al menos {minimum} observaciones.")
    return a, b, labels[0], labels[1]


def _success_mask(series: pd.Series, success_value: str | None) -> pd.Series:
    """Boolean mask of 'success' observations of a binary/categorical column."""
    clean = series.dropna()
    labels = clean.map(label_of)
    if success_value not in (None, ""):
        return series.map(lambda v: pd.notna(v) and label_of(v) == str(success_value).strip())
    unique = set(labels.unique())
    if unique <= {"0", "1"} or unique <= {"True", "False"}:
        positive = "1" if "1" in unique or unique <= {"0", "1"} else "True"
        return series.map(lambda v: pd.notna(v) and label_of(v) == positive)
    raise InvalidInputError("Indique el «valor de éxito» (la categoría que cuenta como éxito).")


def _hypothesis_texts(symbol: str, value: str, tail: str) -> tuple[str, str]:
    h0 = f"H₀: {symbol} = {value}"
    h1 = {
        "two-sided": f"H₁: {symbol} ≠ {value}",
        "left": f"H₁: {symbol} < {value}",
        "right": f"H₁: {symbol} > {value}",
    }[tail]
    return h0, h1


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _conclusion(p: float, alpha: float, h0: str) -> tuple[bool, str]:
    reject = p < alpha
    relation = "<" if reject else "≥"
    if reject:
        text = (
            f"Como p = {_fmt(p)} {relation} α = {_fmt(alpha)}, se rechaza {h0.split(':')[0]}: "
            "hay evidencia estadística suficiente a favor de la hipótesis alternativa."
        )
    else:
        text = (
            f"Como p = {_fmt(p)} {relation} α = {_fmt(alpha)}, no se rechaza {h0.split(':')[0]}: "
            "no hay evidencia suficiente para respaldar la hipótesis alternativa."
        )
    return reject, text


def _result(
    spec_id: str,
    name: str,
    h0: str,
    h1: str,
    stat_name: str,
    statistic: float,
    settings: Settings,
    *,
    p: float | None = None,
    frozen: Any = None,
    ref: DistributionRef | None = None,
    df: float | list[float] | None = None,
    tail: str | None = None,
    **extra: Any,
) -> HypothesisTestResult:
    tail = tail or settings.tail
    if p is None:
        if frozen is None:
            raise ComputationError("No se pudo calcular el valor p.")
        p = p_value(frozen, statistic, tail)
    crit = critical_values(frozen, settings.alpha, tail) if frozen is not None else []
    reject, conclusion = _conclusion(p, settings.alpha, h0)
    return HypothesisTestResult(
        test_id=spec_id,
        test_name=name,
        null_hypothesis=h0,
        alternative_hypothesis=h1,
        statistic_name=stat_name,
        statistic=float(statistic),
        df=df,
        p_value=float(p),
        alpha=settings.alpha,
        tail=tail,
        reject_null=reject,
        conclusion=conclusion,
        critical_values=crit,
        distribution=ref,
        **extra,
    )


def _mean_summary(arr: np.ndarray) -> dict[str, float]:
    return {"n": float(len(arr)), "mean": float(np.mean(arr)), "sd": float(np.std(arr, ddof=1))}


def _require_variance(sd: float, label: str = "la muestra") -> None:
    if not sd > 0:
        raise ComputationError(f"La desviación estándar de {label} es cero: no se puede calcular el estadístico.")


# ============================================================================ z for mean
def _prepare_one_mean(frame, roles, params):
    return _mean_summary(_clean(frame, roles.get("x", "")))


def _compute_z_mean(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n, mean = float(inp["n"]), float(inp["mean"])
    mu0 = _number(s.params, "mu0", "μ₀", 0.0)
    sigma = _positive(_number(s.params, "sigma", "σ conocida"), "σ conocida")
    if n < 1:
        raise InvalidInputError("n debe ser al menos 1.")
    se = sigma / math.sqrt(n)
    z = (mean - mu0) / se
    h0, h1 = _hypothesis_texts("μ", _fmt(mu0), s.tail)
    zc = stats.norm.isf(s.alpha / 2)
    return _result(
        "z_mean",
        "Prueba z para una media (σ conocida)",
        h0,
        h1,
        "z",
        z,
        s,
        frozen=stats.norm(),
        ref=DistributionRef("normal", {"mu": 0.0, "sigma": 1.0}),
        estimate_label="Media muestral",
        estimate=mean,
        confidence_interval=[mean - zc * se, mean + zc * se],
        confidence=1 - s.alpha,
        effect_size_label="d de Cohen",
        effect_size=(mean - mu0) / sigma,
        sample_info={"n": int(n), "media": mean, "σ": sigma, "error estándar": se},
    )


# ============================================================================ t one sample
def _compute_t_one(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n, mean, sd = float(inp["n"]), float(inp["mean"]), float(inp["sd"])
    if n < 2:
        raise InvalidInputError("n debe ser al menos 2.")
    _require_variance(sd)
    mu0 = _number(s.params, "mu0", "μ₀", 0.0)
    se = sd / math.sqrt(n)
    t = (mean - mu0) / se
    df = n - 1
    dist = stats.t(df)
    tc = dist.isf(s.alpha / 2)
    h0, h1 = _hypothesis_texts("μ", _fmt(mu0), s.tail)
    return _result(
        "t_one_sample",
        "Prueba t para una media",
        h0,
        h1,
        "t",
        t,
        s,
        df=df,
        frozen=dist,
        ref=DistributionRef("t", {"df": df}),
        estimate_label="Media muestral",
        estimate=mean,
        confidence_interval=[mean - tc * se, mean + tc * se],
        confidence=1 - s.alpha,
        effect_size_label="d de Cohen",
        effect_size=(mean - mu0) / sd,
        sample_info={"n": int(n), "media": mean, "desv. estándar": sd, "error estándar": se},
    )


# ============================================================================ t two samples
def _prepare_two_means(frame, roles, params):
    a, b, la, lb = _two_samples(frame, roles)
    sa, sb = _mean_summary(a), _mean_summary(b)
    return {
        "n1": sa["n"],
        "mean1": sa["mean"],
        "sd1": sa["sd"],
        "n2": sb["n"],
        "mean2": sb["mean"],
        "sd2": sb["sd"],
        "label1": la,
        "label2": lb,
    }


def _compute_t_two(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n1, m1, s1, n2, m2, s2 = (float(inp[k]) for k in ("n1", "mean1", "sd1", "n2", "mean2", "sd2"))
    if n1 < 2 or n2 < 2:
        raise InvalidInputError("Cada muestra necesita al menos 2 observaciones.")
    _require_variance(max(s1, s2), "las muestras")
    diff0 = _number(s.params, "diff0", "diferencia hipotética", 0.0)
    pooled_mode = str(s.params.get("variance_assumption", "welch")) == "pooled"
    v1, v2 = s1**2 / n1, s2**2 / n2
    if pooled_mode:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        name = "Prueba t para dos medias (varianzas iguales)"
    else:
        se = math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
        name = "Prueba t para dos medias (Welch)"
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    diff = m1 - m2
    t = (diff - diff0) / se
    dist = stats.t(df)
    tc = dist.isf(s.alpha / 2)
    h0, h1 = _hypothesis_texts("μ₁ − μ₂", _fmt(diff0), s.tail)
    l1, l2 = inp.get("label1", "Muestra 1"), inp.get("label2", "Muestra 2")
    return _result(
        "t_two_sample",
        name,
        h0,
        h1,
        "t",
        t,
        s,
        df=df,
        frozen=dist,
        ref=DistributionRef("t", {"df": df}),
        estimate_label="Diferencia de medias (1 − 2)",
        estimate=diff,
        confidence_interval=[diff - tc * se, diff + tc * se],
        confidence=1 - s.alpha,
        effect_size_label="d de Cohen",
        effect_size=diff / sp if sp > 0 else None,
        sample_info={
            f"n ({l1})": int(n1),
            f"media ({l1})": m1,
            f"desv. ({l1})": s1,
            f"n ({l2})": int(n2),
            f"media ({l2})": m2,
            f"desv. ({l2})": s2,
        },
    )


# ============================================================================ t paired
def _prepare_paired(frame, roles, params):
    x_name, y_name = roles.get("x"), roles.get("y")
    if not x_name or not y_name:
        raise InvalidInputError("Seleccione las dos columnas pareadas (antes y después).")
    pair = pd.DataFrame({"x": numeric_series(frame, x_name), "y": numeric_series(frame, y_name)}).dropna()
    if len(pair) < 2:
        raise InvalidInputError("Se necesitan al menos 2 pares completos.")
    diff = (pair["x"] - pair["y"]).to_numpy()
    summary = _mean_summary(diff)
    return {"n": summary["n"], "mean_diff": summary["mean"], "sd_diff": summary["sd"], "diffs": diff}


def _compute_t_paired(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n, mean, sd = float(inp["n"]), float(inp["mean_diff"]), float(inp["sd_diff"])
    if n < 2:
        raise InvalidInputError("n debe ser al menos 2.")
    _require_variance(sd, "las diferencias")
    diff0 = _number(s.params, "diff0", "diferencia hipotética", 0.0)
    se = sd / math.sqrt(n)
    t = (mean - diff0) / se
    df = n - 1
    dist = stats.t(df)
    tc = dist.isf(s.alpha / 2)
    h0, h1 = _hypothesis_texts("μ_d", _fmt(diff0), s.tail)
    return _result(
        "t_paired",
        "Prueba t para muestras pareadas",
        h0,
        h1,
        "t",
        t,
        s,
        df=df,
        frozen=dist,
        ref=DistributionRef("t", {"df": df}),
        estimate_label="Diferencia media (x − y)",
        estimate=mean,
        confidence_interval=[mean - tc * se, mean + tc * se],
        confidence=1 - s.alpha,
        effect_size_label="d de Cohen (diferencias)",
        effect_size=(mean - diff0) / sd,
        sample_info={"pares": int(n), "diferencia media": mean, "desv. de diferencias": sd},
    )


# ============================================================================ proportions
def _wilson(successes: float, n: float, confidence: float) -> list[float]:
    z = stats.norm.isf((1 - confidence) / 2)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [centre - half, centre + half]


def _cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _prepare_one_prop(frame, roles, params):
    name = roles.get("x")
    if not name:
        raise InvalidInputError("Seleccione la columna de la variable.")
    require_columns(frame, [name])
    series = frame[name]
    mask = _success_mask(series, params.get("success_value"))
    valid = series.notna()
    return {"successes": float(mask[valid].sum()), "n": float(valid.sum())}


def _check_counts(successes: float, n: float) -> tuple[int, int]:
    n_i, x_i = _count(n, "n", 1), _count(successes, "éxitos", 0)
    if x_i > n_i:
        raise InvalidInputError("Los éxitos no pueden superar a n.")
    return x_i, n_i


def _compute_z_prop_one(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    x, n = _check_counts(inp["successes"], inp["n"])
    p0 = _number(s.params, "p0", "p₀", 0.5)
    if not 0 < p0 < 1:
        raise InvalidInputError("p₀ debe estar entre 0 y 1 (exclusivo).")
    phat = x / n
    z = (phat - p0) / math.sqrt(p0 * (1 - p0) / n)
    h0, h1 = _hypothesis_texts("p", _fmt(p0), s.tail)
    notes = []
    if n * p0 < 10 or n * (1 - p0) < 10:
        notes.append(
            "Aviso: n·p₀ o n·(1−p₀) es menor que 10; la aproximación normal puede ser pobre "
            "(considere la prueba binomial exacta)."
        )
    return _result(
        "z_prop_one",
        "Prueba z para una proporción",
        h0,
        h1,
        "z",
        z,
        s,
        frozen=stats.norm(),
        ref=DistributionRef("normal", {"mu": 0.0, "sigma": 1.0}),
        estimate_label="Proporción muestral",
        estimate=phat,
        confidence_interval=_wilson(x, n, 1 - s.alpha),
        confidence=1 - s.alpha,
        effect_size_label="h de Cohen",
        effect_size=_cohens_h(phat, p0),
        sample_info={"éxitos": x, "n": n, "p̂": phat},
        notes=notes,
    )


def _prepare_two_props(frame, roles, params):
    x_name, group = roles.get("x"), roles.get("group")
    if not x_name or not group:
        raise InvalidInputError("Seleccione la variable y la columna de grupos.")
    require_columns(frame, [x_name, group])
    unique = sorted(frame[group].dropna().map(label_of).unique())
    if len(unique) != 2:
        raise InvalidInputError(f"La columna de grupos debe tener exactamente 2 niveles (tiene {len(unique)}).")
    mask = _success_mask(frame[x_name], params.get("success_value"))
    keys = frame[group].map(lambda v: label_of(v) if pd.notna(v) else None)
    valid = frame[x_name].notna()
    out: dict[str, Any] = {}
    for i, level in enumerate(unique, start=1):
        in_group = (keys == level) & valid
        out[f"successes{i}"], out[f"n{i}"] = float(mask[in_group].sum()), float(in_group.sum())
        out[f"label{i}"] = level
    return out


def _compute_z_prop_two(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    x1, n1 = _check_counts(inp["successes1"], inp["n1"])
    x2, n2 = _check_counts(inp["successes2"], inp["n2"])
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        raise ComputationError("La proporción combinada es 0 o 1: no se puede calcular el estadístico z.")
    z = (p1 - p2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.isf(s.alpha / 2)
    h0, h1 = _hypothesis_texts("p₁ − p₂", "0", s.tail)
    notes = []
    if min(n1 * pooled, n1 * (1 - pooled), n2 * pooled, n2 * (1 - pooled)) < 5:
        notes.append("Aviso: alguna frecuencia esperada es menor que 5; considere la prueba exacta de Fisher.")
    return _result(
        "z_prop_two",
        "Prueba z para dos proporciones",
        h0,
        h1,
        "z",
        z,
        s,
        frozen=stats.norm(),
        ref=DistributionRef("normal", {"mu": 0.0, "sigma": 1.0}),
        estimate_label="Diferencia de proporciones (1 − 2)",
        estimate=p1 - p2,
        confidence_interval=[p1 - p2 - zc * se_unpooled, p1 - p2 + zc * se_unpooled],
        confidence=1 - s.alpha,
        effect_size_label="h de Cohen",
        effect_size=_cohens_h(p1, p2),
        sample_info={"p̂₁": p1, "n₁": n1, "p̂₂": p2, "n₂": n2, "p combinada": pooled},
        notes=notes,
    )


def _compute_binomial(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    x, n = _check_counts(inp["successes"], inp["n"])
    p0 = _number(s.params, "p0", "p₀", 0.5)
    if not 0 < p0 < 1:
        raise InvalidInputError("p₀ debe estar entre 0 y 1 (exclusivo).")
    alternative = {"two-sided": "two-sided", "left": "less", "right": "greater"}[s.tail]
    outcome = stats.binomtest(x, n, p0, alternative=alternative)
    ci = outcome.proportion_ci(confidence_level=1 - s.alpha, method="exact")
    h0, h1 = _hypothesis_texts("p", _fmt(p0), s.tail)
    return _result(
        "binomial_exact",
        "Prueba binomial exacta",
        h0,
        h1,
        "X (éxitos)",
        float(x),
        s,
        p=float(outcome.pvalue),
        ref=DistributionRef("binomial", {"n": n, "p": p0}),
        estimate_label="Proporción muestral",
        estimate=x / n,
        confidence_interval=[float(ci.low), float(ci.high)],
        confidence=1 - s.alpha,
        effect_size_label="h de Cohen",
        effect_size=_cohens_h(x / n, p0),
        sample_info={"éxitos": x, "n": n, "p̂": x / n},
        notes=["IC de Clopper-Pearson (exacto)."],
    )


# ============================================================================ variances
def _compute_chi2_var(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n, sd = float(inp["n"]), float(inp["sd"])
    if n < 2:
        raise InvalidInputError("n debe ser al menos 2.")
    _require_variance(sd)
    var0 = _positive(_number(s.params, "variance0", "σ₀² (varianza hipotética)"), "σ₀²")
    df = n - 1
    stat = df * sd**2 / var0
    dist = stats.chi2(df)
    h0, h1 = _hypothesis_texts("σ²", _fmt(var0), s.tail)
    lo = df * sd**2 / dist.isf(s.alpha / 2)
    hi = df * sd**2 / dist.ppf(s.alpha / 2)
    return _result(
        "chi2_variance",
        "Prueba χ² para una varianza",
        h0,
        h1,
        "χ²",
        stat,
        s,
        df=df,
        frozen=dist,
        ref=DistributionRef("chi2", {"df": df}),
        estimate_label="Varianza muestral",
        estimate=sd**2,
        confidence_interval=[lo, hi],
        confidence=1 - s.alpha,
        sample_info={"n": int(n), "varianza muestral": sd**2},
    )


def _prepare_two_sds(frame, roles, params):
    a, b, la, lb = _two_samples(frame, roles)
    return {
        "n1": float(len(a)),
        "sd1": float(np.std(a, ddof=1)),
        "n2": float(len(b)),
        "sd2": float(np.std(b, ddof=1)),
        "label1": la,
        "label2": lb,
    }


def _compute_f_var(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    n1, s1, n2, s2 = (float(inp[k]) for k in ("n1", "sd1", "n2", "sd2"))
    if n1 < 2 or n2 < 2:
        raise InvalidInputError("Cada muestra necesita al menos 2 observaciones.")
    _require_variance(min(s1, s2), "alguna muestra")
    f = s1**2 / s2**2
    dist = stats.f(n1 - 1, n2 - 1)
    h0, h1 = _hypothesis_texts("σ₁²/σ₂²", "1", s.tail)
    ci = [f / dist.isf(s.alpha / 2), f / dist.ppf(s.alpha / 2)]
    return _result(
        "f_variance",
        "Prueba F para el cociente de varianzas",
        h0,
        h1,
        "F",
        f,
        s,
        df=[n1 - 1, n2 - 1],
        frozen=dist,
        ref=DistributionRef("f", {"dfn": n1 - 1, "dfd": n2 - 1}),
        estimate_label="Cociente de varianzas (1/2)",
        estimate=f,
        confidence_interval=ci,
        confidence=1 - s.alpha,
        sample_info={"n₁": int(n1), "s₁²": s1**2, "n₂": int(n2), "s₂²": s2**2},
    )


# ============================================================================ chi-square family
def parse_matrix(raw: Any) -> np.ndarray:
    """Accept a list of lists or text ('1 2; 3 4' / one row per line) and return a float matrix."""
    if isinstance(raw, str):
        rows = [r for r in raw.replace(";", "\n").splitlines() if r.strip()]
        raw = [[c for c in r.replace(",", " ").replace("\t", " ").split() if c] for r in rows]
    try:
        matrix = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise InvalidInputError("La tabla de contingencia debe contener solo números.") from None
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise InvalidInputError("La tabla de contingencia debe tener al menos 2 filas y 2 columnas.")
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise InvalidInputError("Las frecuencias deben ser números no negativos.")
    return matrix


def _prepare_contingency(frame, roles, params):
    row, col = roles.get("row"), roles.get("col")
    if not row or not col:
        raise InvalidInputError("Seleccione las dos columnas categóricas.")
    require_columns(frame, [row, col])
    pair = frame[[row, col]].dropna()
    if pair.empty:
        raise InvalidInputError("No hay filas completas en las dos columnas elegidas.")
    table = pd.crosstab(pair[row].map(label_of), pair[col].map(label_of))
    return {"table": table.to_numpy(dtype=float), "row_labels": list(table.index), "col_labels": list(table.columns)}


def _summary_table(inp: dict[str, Any]) -> np.ndarray:
    raw = inp["table"]
    return raw if isinstance(raw, np.ndarray) else parse_matrix(raw)


def _compute_chi2_independence(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    table = _summary_table(inp)
    rows = inp.get("row_labels") or [f"Fila {i + 1}" for i in range(table.shape[0])]
    cols = inp.get("col_labels") or [f"Col. {j + 1}" for j in range(table.shape[1])]
    yates = str(s.params.get("yates", "no")) == "yes"
    try:
        chi2, p, df, expected = stats.chi2_contingency(table, correction=yates)
    except ValueError as error:
        raise ComputationError(f"No se pudo calcular χ²: {error}") from error
    n = float(table.sum())
    k = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else 0.0
    notes = []
    low = float((expected < 5).mean())
    if low > 0.2:
        notes.append(
            f"Aviso: {low:.0%} de las frecuencias esperadas son menores que 5; el resultado puede no ser fiable."
        )
    h0 = "H₀: las variables son independientes"
    h1 = "H₁: las variables están asociadas"
    tables = [
        TableData(
            "Frecuencias observadas",
            [""] + [str(c) for c in cols],
            [[str(r), *[float(v) for v in table[i]]] for i, r in enumerate(rows)],
        ),
        TableData(
            "Frecuencias esperadas",
            [""] + [str(c) for c in cols],
            [[str(r), *[float(v) for v in expected[i]]] for i, r in enumerate(rows)],
        ),
    ]
    return _result(
        "chi2_independence",
        "Prueba χ² de independencia",
        h0,
        h1,
        "χ²",
        float(chi2),
        s,
        df=float(df),
        p=float(p),
        frozen=stats.chi2(df),
        ref=DistributionRef("chi2", {"df": float(df)}),
        tail="right",
        effect_size_label="V de Cramér",
        effect_size=v,
        sample_info={"n": n, "filas": table.shape[0], "columnas": table.shape[1]},
        tables=tables,
        notes=notes,
    )


def _parse_vector(raw: Any, label: str) -> np.ndarray:
    if isinstance(raw, str):
        raw = [c for c in raw.replace(";", " ").replace(",", " ").split() if c]
    try:
        vector = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise InvalidInputError(f"«{label}» debe contener solo números.") from None
    if vector.ndim != 1 or len(vector) < 2:
        raise InvalidInputError(f"«{label}» necesita al menos 2 valores.")
    return vector


def _prepare_goodness(frame, roles, params):
    name = roles.get("x")
    if not name:
        raise InvalidInputError("Seleccione la columna categórica.")
    require_columns(frame, [name])
    counts = frame[name].dropna().map(label_of).value_counts().sort_index()
    if len(counts) < 2:
        raise InvalidInputError("La variable debe tener al menos 2 categorías.")
    return {"observed": counts.to_numpy(dtype=float), "labels": list(counts.index)}


def _compute_goodness(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    observed = _parse_vector(inp["observed"], "frecuencias observadas")
    if np.any(observed < 0):
        raise InvalidInputError("Las frecuencias observadas no pueden ser negativas.")
    raw_expected = s.params.get("expected")
    if raw_expected in (None, ""):
        proportions = np.full(len(observed), 1.0 / len(observed))
    else:
        proportions = _parse_vector(raw_expected, "proporciones esperadas")
        if len(proportions) != len(observed):
            raise InvalidInputError(f"Se esperaban {len(observed)} proporciones y se recibieron {len(proportions)}.")
        if np.any(proportions <= 0):
            raise InvalidInputError("Las proporciones esperadas deben ser positivas.")
        proportions = proportions / proportions.sum()
    expected = proportions * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    df = len(observed) - 1
    labels = inp.get("labels") or [f"Categoría {i + 1}" for i in range(len(observed))]
    notes = ["Las proporciones esperadas se normalizaron para sumar 1."] if raw_expected not in (None, "") else []
    if np.any(expected < 5):
        notes.append("Aviso: hay frecuencias esperadas menores que 5.")
    return _result(
        "chi2_goodness",
        "Prueba χ² de bondad de ajuste",
        "H₀: los datos siguen las proporciones esperadas",
        "H₁: los datos no siguen las proporciones esperadas",
        "χ²",
        float(chi2),
        s,
        df=float(df),
        p=float(p),
        frozen=stats.chi2(df),
        ref=DistributionRef("chi2", {"df": float(df)}),
        tail="right",
        sample_info={"n": float(observed.sum()), "categorías": len(observed)},
        tables=[
            TableData(
                "Observadas vs esperadas",
                ["Categoría", "Observada", "Esperada", "Proporción esperada"],
                [
                    [str(labels[i]), float(observed[i]), float(expected[i]), float(proportions[i])]
                    for i in range(len(observed))
                ],
            )
        ],
        notes=notes,
    )


def _compute_fisher(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    table = _summary_table(inp)
    if table.shape != (2, 2):
        raise InvalidInputError("La prueba exacta de Fisher requiere una tabla 2×2.")
    alternative = {"two-sided": "two-sided", "left": "less", "right": "greater"}[s.tail]
    odds, p = stats.fisher_exact(table, alternative=alternative)
    rows = inp.get("row_labels") or ["Fila 1", "Fila 2"]
    cols = inp.get("col_labels") or ["Col. 1", "Col. 2"]
    h0, h1 = _hypothesis_texts("razón de momios (OR)", "1", s.tail)
    return _result(
        "fisher_exact",
        "Prueba exacta de Fisher",
        h0,
        h1,
        "Razón de momios (OR)",
        float(odds),
        s,
        p=float(p),
        sample_info={"n": float(table.sum())},
        tables=[
            TableData(
                "Tabla 2×2",
                [""] + [str(c) for c in cols],
                [[str(r), *[float(v) for v in table[i]]] for i, r in enumerate(rows)],
            )
        ],
    )


# ============================================================================ correlation & non-parametric
def _prepare_xy_pairs(frame, roles, params):
    x_name, y_name = roles.get("x"), roles.get("y")
    if not x_name or not y_name:
        raise InvalidInputError("Seleccione las dos columnas numéricas.")
    pair = pd.DataFrame({"x": numeric_series(frame, x_name), "y": numeric_series(frame, y_name)}).dropna()
    if len(pair) < 3:
        raise InvalidInputError("Se necesitan al menos 3 pares completos.")
    return {"x": pair["x"].to_numpy(), "y": pair["y"].to_numpy(), "x_name": x_name, "y_name": y_name}


def _compute_correlation(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    x, y = np.asarray(inp["x"], dtype=float), np.asarray(inp["y"], dtype=float)
    method = str(s.params.get("method", "pearson"))
    if method not in ("pearson", "spearman", "kendall"):
        raise InvalidInputError("Método de correlación inválido.")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ComputationError("Una de las variables es constante: la correlación no está definida.")
    n = len(x)
    alternative = {"two-sided": "two-sided", "left": "less", "right": "greater"}[s.tail]
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    outcome = func(x, y, alternative=alternative)
    r, p = float(outcome.statistic), float(outcome.pvalue)
    symbol = {"pearson": "ρ", "spearman": "ρ_s", "kendall": "τ"}[method]
    h0, h1 = _hypothesis_texts(symbol, "0", s.tail)
    name = {
        "pearson": "Correlación de Pearson",
        "spearman": "Correlación de Spearman",
        "kendall": "Correlación τ de Kendall",
    }[method]
    extra: dict[str, Any] = {}
    t_stat: float | None = None
    if method in ("pearson", "spearman") and n > 3 and abs(r) < 1:
        t_stat = r * math.sqrt((n - 2) / (1 - r**2))
        extra = {
            "frozen": stats.t(n - 2),
            "ref": DistributionRef("t", {"df": float(n - 2)}),
            "df": float(n - 2),
            "notes": [f"Estadístico t = {t_stat:.4g} con {n - 2} gl; el valor p proviene de scipy."],
        }
    ci = None
    if method == "pearson" and n > 3 and abs(r) < 1:
        z = math.atanh(r)
        zc = stats.norm.isf(s.alpha / 2) / math.sqrt(n - 3)
        ci = [math.tanh(z - zc), math.tanh(z + zc)]
    result = _result(
        "correlation",
        name,
        h0,
        h1,
        "r" if method == "pearson" else ("ρ_s" if method == "spearman" else "τ"),
        r,
        s,
        p=p,
        estimate_label="Coeficiente de correlación",
        estimate=r,
        confidence_interval=ci,
        confidence=(1 - s.alpha) if ci else None,
        effect_size_label="r²" if method == "pearson" else None,
        effect_size=r**2 if method == "pearson" else None,
        sample_info={"n": n, "x": inp.get("x_name", "x"), "y": inp.get("y_name", "y")},
        **extra,
    )
    if t_stat is not None:
        result.plot_data["t_statistic"] = t_stat
    return result


def _prepare_two_raw(frame, roles, params):
    a, b, la, lb = _two_samples(frame, roles, minimum=1)
    return {"a": a, "b": b, "label1": la, "label2": lb}


def _compute_mann_whitney(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    a, b = np.asarray(inp["a"], dtype=float), np.asarray(inp["b"], dtype=float)
    alternative = {"two-sided": "two-sided", "left": "less", "right": "greater"}[s.tail]
    outcome = stats.mannwhitneyu(a, b, alternative=alternative)
    u1 = float(outcome.statistic)
    h0 = "H₀: las distribuciones de ambos grupos son iguales"
    h1 = {
        "two-sided": "H₁: las distribuciones difieren",
        "left": "H₁: el grupo 1 tiende a tomar valores menores",
        "right": "H₁: el grupo 1 tiende a tomar valores mayores",
    }[s.tail]
    n1, n2 = len(a), len(b)
    return _result(
        "mann_whitney",
        "U de Mann-Whitney",
        h0,
        h1,
        "U",
        u1,
        s,
        p=float(outcome.pvalue),
        effect_size_label="Correlación biserial de rangos",
        effect_size=2 * u1 / (n1 * n2) - 1,
        sample_info={"n₁": n1, "mediana₁": float(np.median(a)), "n₂": n2, "mediana₂": float(np.median(b))},
    )


def _prepare_wilcoxon(frame, roles, params):
    x_name, y_name = roles.get("x"), roles.get("y")
    if not x_name:
        raise InvalidInputError("Seleccione la columna numérica «x».")
    if y_name:
        pair = pd.DataFrame({"x": numeric_series(frame, x_name), "y": numeric_series(frame, y_name)}).dropna()
        diffs = (pair["x"] - pair["y"]).to_numpy()
        return {"values": diffs, "paired": True}
    return {"values": _clean(frame, x_name, 1), "paired": False}


def _compute_wilcoxon(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    values = np.asarray(inp["values"], dtype=float)
    paired = bool(inp["paired"])
    centre = 0.0 if paired else _number(s.params, "mu0", "mediana hipotética", 0.0)
    diffs = values - centre
    if np.count_nonzero(diffs) < 1:
        raise ComputationError("Todas las diferencias son cero.")
    alternative = {"two-sided": "two-sided", "left": "less", "right": "greater"}[s.tail]
    outcome = stats.wilcoxon(diffs, alternative=alternative)
    symbol = "mediana de las diferencias" if paired else "mediana"
    h0, h1 = _hypothesis_texts(symbol, _fmt(centre), s.tail)
    return _result(
        "wilcoxon",
        "Rango con signo de Wilcoxon",
        h0,
        h1,
        "W",
        float(outcome.statistic),
        s,
        p=float(outcome.pvalue),
        sample_info={"n": len(values), "mediana": float(np.median(values))},
        notes=["Las diferencias iguales a cero se descartan (método de Wilcoxon)."],
    )


def _compute_shapiro(inp: dict[str, Any], s: Settings) -> HypothesisTestResult:
    values = np.asarray(inp["values"], dtype=float)
    if len(values) < 3 or len(values) > 5000:
        raise InvalidInputError("Shapiro-Wilk requiere entre 3 y 5000 observaciones.")
    if np.ptp(values) == 0:
        raise ComputationError("Todos los valores son iguales.")
    outcome = stats.shapiro(values)
    return _result(
        "shapiro",
        "Normalidad de Shapiro-Wilk",
        "H₀: los datos provienen de una distribución normal",
        "H₁: los datos no provienen de una distribución normal",
        "W",
        float(outcome.statistic),
        s,
        p=float(outcome.pvalue),
        tail="left",
        sample_info={"n": len(values)},
    )


def _prepare_single_values(frame, roles, params):
    name = roles.get("x")
    if not name:
        raise InvalidInputError("Seleccione la columna numérica.")
    return {"values": _clean(frame, name, 3)}


# ============================================================================ catalogue
def _field(name, label, kind="number", default=None, options=(), help="", optional=False):
    return FieldSpec(name, label, kind, default, tuple(options), help, optional)


_X = RoleSpec("x", "Variable numérica")
_X_OPT_Y_GROUP = (
    RoleSpec("x", "Variable numérica (x)"),
    RoleSpec("y", "Segunda columna numérica (formato ancho)", "numeric", True),
    RoleSpec("group", "Columna de grupos (formato largo, 2 niveles)", "any", True),
)
_MU0 = _field("mu0", "Valor hipotético (μ₀)", default=0)
_SUCCESS = _field(
    "success_value",
    "Valor de éxito",
    "text",
    "",
    help="Categoría que cuenta como éxito (opcional si la columna es 0/1).",
    optional=True,
)

SPECS: tuple[TestSpec, ...] = (
    TestSpec(
        "z_mean",
        "Prueba z para una media",
        "Medias",
        "Compara la media muestral con μ₀ cuando σ es conocida.",
        (_X,),
        (_MU0, _field("sigma", "σ conocida", default=1)),
        (_field("n", "n"), _field("mean", "Media muestral")),
        ALL_TAILS,
        _prepare_one_mean,
        _compute_z_mean,
    ),
    TestSpec(
        "t_one_sample",
        "Prueba t para una media",
        "Medias",
        "Compara la media muestral con μ₀ cuando σ es desconocida.",
        (_X,),
        (_MU0,),
        (_field("n", "n"), _field("mean", "Media muestral"), _field("sd", "Desv. estándar muestral")),
        ALL_TAILS,
        _prepare_one_mean,
        _compute_t_one,
    ),
    TestSpec(
        "t_two_sample",
        "Prueba t para dos medias independientes",
        "Medias",
        "Compara las medias de dos grupos independientes (Welch o varianzas iguales).",
        _X_OPT_Y_GROUP,
        (
            _field("diff0", "Diferencia hipotética (μ₁ − μ₂)", default=0),
            _field(
                "variance_assumption",
                "Varianzas",
                "select",
                "welch",
                (("welch", "Desiguales (Welch)"), ("pooled", "Iguales (agrupada)")),
            ),
        ),
        (
            _field("n1", "n₁"),
            _field("mean1", "Media 1"),
            _field("sd1", "Desv. 1"),
            _field("n2", "n₂"),
            _field("mean2", "Media 2"),
            _field("sd2", "Desv. 2"),
        ),
        ALL_TAILS,
        _prepare_two_means,
        _compute_t_two,
    ),
    TestSpec(
        "t_paired",
        "Prueba t para muestras pareadas",
        "Medias",
        "Compara dos mediciones sobre los mismos sujetos.",
        (RoleSpec("x", "Medición 1 (antes)"), RoleSpec("y", "Medición 2 (después)")),
        (_field("diff0", "Diferencia hipotética", default=0),),
        (_field("n", "n (pares)"), _field("mean_diff", "Diferencia media"), _field("sd_diff", "Desv. de diferencias")),
        ALL_TAILS,
        _prepare_paired,
        _compute_t_paired,
    ),
    TestSpec(
        "z_prop_one",
        "Prueba z para una proporción",
        "Proporciones",
        "Compara una proporción con p₀ (aproximación normal).",
        (RoleSpec("x", "Variable binaria/categórica", "any"),),
        (_field("p0", "Proporción hipotética (p₀)", default=0.5), _SUCCESS),
        (_field("successes", "Éxitos (x)"), _field("n", "n")),
        ALL_TAILS,
        _prepare_one_prop,
        _compute_z_prop_one,
    ),
    TestSpec(
        "z_prop_two",
        "Prueba z para dos proporciones",
        "Proporciones",
        "Compara las proporciones de dos grupos.",
        (
            RoleSpec("x", "Variable binaria/categórica", "any"),
            RoleSpec("group", "Columna de grupos (2 niveles)", "any"),
        ),
        (_SUCCESS,),
        (_field("successes1", "Éxitos 1"), _field("n1", "n₁"), _field("successes2", "Éxitos 2"), _field("n2", "n₂")),
        ALL_TAILS,
        _prepare_two_props,
        _compute_z_prop_two,
    ),
    TestSpec(
        "binomial_exact",
        "Prueba binomial exacta",
        "Proporciones",
        "Prueba exacta de una proporción (recomendada con n pequeño).",
        (RoleSpec("x", "Variable binaria/categórica", "any"),),
        (_field("p0", "Proporción hipotética (p₀)", default=0.5), _SUCCESS),
        (_field("successes", "Éxitos (x)"), _field("n", "n")),
        ALL_TAILS,
        _prepare_one_prop,
        _compute_binomial,
    ),
    TestSpec(
        "chi2_variance",
        "Prueba χ² para una varianza",
        "Varianzas",
        "Compara la varianza muestral con σ₀².",
        (_X,),
        (_field("variance0", "Varianza hipotética (σ₀²)", default=1),),
        (_field("n", "n"), _field("sd", "Desv. estándar muestral")),
        ALL_TAILS,
        _prepare_one_mean,
        _compute_chi2_var,
    ),
    TestSpec(
        "f_variance",
        "Prueba F para dos varianzas",
        "Varianzas",
        "Compara las varianzas de dos grupos independientes.",
        _X_OPT_Y_GROUP,
        (),
        (_field("n1", "n₁"), _field("sd1", "Desv. 1"), _field("n2", "n₂"), _field("sd2", "Desv. 2")),
        ALL_TAILS,
        _prepare_two_sds,
        _compute_f_var,
    ),
    TestSpec(
        "chi2_independence",
        "χ² de independencia",
        "Variables categóricas",
        "Evalúa la asociación entre dos variables categóricas.",
        (RoleSpec("row", "Variable de filas", "any"), RoleSpec("col", "Variable de columnas", "any")),
        (_field("yates", "Corrección de Yates (2×2)", "select", "no", (("no", "No"), ("yes", "Sí"))),),
        (
            _field(
                "table",
                "Tabla de contingencia",
                "matrix",
                help="Una fila por línea, valores separados por espacio o coma.",
            ),
        ),
        RIGHT_ONLY,
        _prepare_contingency,
        _compute_chi2_independence,
    ),
    TestSpec(
        "chi2_goodness",
        "χ² de bondad de ajuste",
        "Variables categóricas",
        "Compara frecuencias observadas con proporciones esperadas.",
        (RoleSpec("x", "Variable categórica", "any"),),
        (
            _field(
                "expected",
                "Proporciones esperadas",
                "text",
                "",
                help="Opcional, en el orden alfabético de las categorías. Vacío = iguales.",
                optional=True,
            ),
        ),
        (_field("observed", "Frecuencias observadas", "text", help="Valores separados por coma."),),
        RIGHT_ONLY,
        _prepare_goodness,
        _compute_goodness,
    ),
    TestSpec(
        "fisher_exact",
        "Prueba exacta de Fisher (2×2)",
        "Variables categóricas",
        "Asociación en tablas 2×2 con muestras pequeñas.",
        (
            RoleSpec("row", "Variable de filas (2 niveles)", "any"),
            RoleSpec("col", "Variable de columnas (2 niveles)", "any"),
        ),
        (),
        (_field("table", "Tabla 2×2", "matrix", help="Dos filas, dos columnas."),),
        ALL_TAILS,
        _prepare_contingency,
        _compute_fisher,
    ),
    TestSpec(
        "correlation",
        "Prueba de correlación",
        "Asociación",
        "Contrasta si la correlación entre dos variables es distinta de cero.",
        (RoleSpec("x", "Variable x"), RoleSpec("y", "Variable y")),
        (
            _field(
                "method",
                "Método",
                "select",
                "pearson",
                (("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall τ")),
            ),
        ),
        (),
        ALL_TAILS,
        _prepare_xy_pairs,
        _compute_correlation,
    ),
    TestSpec(
        "mann_whitney",
        "U de Mann-Whitney",
        "No paramétricas",
        "Alternativa no paramétrica a la t de dos muestras.",
        _X_OPT_Y_GROUP,
        (),
        (),
        ALL_TAILS,
        _prepare_two_raw,
        _compute_mann_whitney,
    ),
    TestSpec(
        "wilcoxon",
        "Rango con signo de Wilcoxon",
        "No paramétricas",
        "Alternativa no paramétrica a la t pareada / de una muestra.",
        (RoleSpec("x", "Variable x"), RoleSpec("y", "Variable y (pareada, opcional)", "numeric", True)),
        (_field("mu0", "Mediana hipotética (una muestra)", default=0),),
        (),
        ALL_TAILS,
        _prepare_wilcoxon,
        _compute_wilcoxon,
    ),
    TestSpec(
        "shapiro",
        "Normalidad de Shapiro-Wilk",
        "Normalidad",
        "Contrasta si los datos provienen de una distribución normal.",
        (_X,),
        (),
        (),
        ("left",),
        _prepare_single_values,
        _compute_shapiro,
    ),
)

_BY_ID: dict[str, TestSpec] = {spec.id: spec for spec in SPECS}


def list_tests() -> list[TestSpec]:
    return list(SPECS)


def get_test(test_id: str) -> TestSpec:
    try:
        return _BY_ID[test_id]
    except KeyError:
        raise InvalidInputError(f"Prueba desconocida: «{test_id}».") from None


def _parse_summary(spec: TestSpec, summary: dict[str, Any]) -> dict[str, Any]:
    if not spec.summary_fields:
        raise InvalidInputError("Esta prueba no admite estadísticos resumidos; use columnas del dataset.")
    inputs: dict[str, Any] = {}
    for fld in spec.summary_fields:
        if fld.kind in ("matrix", "text"):
            if summary.get(fld.name) in (None, ""):
                raise InvalidInputError(f"Falta «{fld.label}».")
            inputs[fld.name] = summary[fld.name]
        else:
            inputs.update(_numeric_inputs(summary, {fld.name: fld.label}))
    return inputs


def run_hypothesis_test(
    test_id: str,
    *,
    frame: pd.DataFrame | None,
    roles: dict[str, str],
    params: dict[str, Any],
    summary: dict[str, Any] | None,
    alpha: float,
    tail: str,
) -> HypothesisTestResult:
    spec = get_test(test_id)
    check_alpha(alpha)
    if tail not in spec.tails:
        if len(spec.tails) != 1:
            raise InvalidInputError(f"La prueba «{spec.label}» no admite la cola «{tail}».")
        tail = spec.tails[0]  # tests with a single natural rejection region ignore the user's choice
    if summary:
        inputs = _parse_summary(spec, summary)
    else:
        if frame is None:
            raise InvalidInputError("Seleccione un dataset o ingrese estadísticos resumidos.")
        missing = [r.label for r in spec.roles if not r.optional and not roles.get(r.name)]
        if missing:
            raise InvalidInputError(f"Falta seleccionar: {', '.join(missing)}.")
        inputs = spec.prepare(frame, roles, params)
    return spec.compute(inputs, Settings(alpha=alpha, tail=tail, params=params))
