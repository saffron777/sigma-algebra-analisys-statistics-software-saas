"""Descriptive statistics: summaries, frequencies, normality, distribution fitting, correlation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.distributions import DistributionRef, get_distribution
from sigma.domain.errors import InvalidInputError
from sigma.domain.results import (
    CorrelationResult,
    DescriptiveResult,
    DistributionFit,
    FrequencyRow,
    FrequencyTable,
    NormalityTest,
    NumericSummary,
)
from sigma.domain.statistics.common import (
    cap_for_plot,
    check_confidence,
    finite,
    label_of,
    numeric_series,
    require_columns,
)

PERCENTILE_POINTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
MAX_GROUPS = 30
MAX_CATEGORIES = 50
CORRELATION_METHODS = ("pearson", "spearman", "kendall")


def summarize_numeric(
    values: pd.Series, name: str, confidence: float = 0.95, group: str | None = None
) -> NumericSummary:
    n_total = len(values)
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        raise InvalidInputError(f"La columna «{name}» no tiene valores válidos.")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else float("nan")
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    if n > 1 and sem > 0:
        ci_low, ci_high = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    elif n > 1:  # constant column: the interval collapses to the mean
        ci_low = ci_high = mean
    else:
        ci_low = ci_high = float("nan")
    q1, q3 = (float(q) for q in clean.quantile([0.25, 0.75]))
    modes = clean.mode()
    mode = float(modes.iloc[0]) if len(modes) and n != clean.nunique() else None
    percentiles = {f"{p}": float(np.percentile(clean, p)) for p in PERCENTILE_POINTS}
    return NumericSummary(
        name=name,
        group=group,
        n=n,
        n_missing=n_total - n,
        mean=mean,
        median=float(clean.median()),
        mode=mode,
        std=std,
        variance=std**2 if n > 1 else float("nan"),
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        range=float(clean.max() - clean.min()),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        cv=(std / abs(mean)) if n > 1 and mean != 0 else None,
        skewness=float(clean.skew()) if n > 2 else float("nan"),
        kurtosis=float(clean.kurt()) if n > 3 else float("nan"),
        sem=sem,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        percentiles=percentiles,
    )


def frequency_table(series: pd.Series, name: str) -> FrequencyTable:
    n_total = len(series)
    clean = series.dropna()
    n = len(clean)
    counts = clean.map(label_of).value_counts()
    if len(counts) > MAX_CATEGORIES:
        head = counts.iloc[:MAX_CATEGORIES]
        counts = pd.concat([head, pd.Series({"(otros)": int(counts.iloc[MAX_CATEGORIES:].sum())})])
    rows: list[FrequencyRow] = []
    cumulative = 0
    for value, count in counts.items():
        cumulative += int(count)
        rows.append(
            FrequencyRow(
                value=str(value),
                count=int(count),
                percent=100.0 * int(count) / n if n else 0.0,
                cumulative_count=cumulative,
                cumulative_percent=100.0 * cumulative / n if n else 0.0,
            )
        )
    return FrequencyTable(name=name, n=n, n_missing=n_total - n, rows=rows)


def normality_tests(values: pd.Series, name: str, group: str | None = None) -> list[NormalityTest]:
    clean = values.dropna().astype(float).to_numpy()
    n = len(clean)
    results: list[NormalityTest] = []
    if n >= 3 and np.ptp(clean) > 0:
        shapiro = stats.shapiro(clean if n <= 5000 else clean[:5000])
        results.append(NormalityTest(name, group, "Shapiro-Wilk", float(shapiro.statistic), float(shapiro.pvalue), n))
    if n >= 8 and np.ptp(clean) > 0:
        k2 = stats.normaltest(clean)
        results.append(
            NormalityTest(
                name,
                group,
                "D'Agostino-Pearson (K²)",
                float(k2.statistic),
                float(k2.pvalue),
                n,
                DistributionRef("chi2", {"df": 2.0}),
            )
        )
    return results


# ------------------------------------------------------------------ distribution fitting
@dataclass(frozen=True)
class _FitSpec:
    key: str
    n_params: int
    valid: Callable[[np.ndarray], bool]
    fit: Callable[[np.ndarray], dict[str, float]]


def _fit_normal(x: np.ndarray) -> dict[str, float]:
    return {"mu": float(np.mean(x)), "sigma": float(np.std(x, ddof=0))}


def _fit_lognormal(x: np.ndarray) -> dict[str, float]:
    logs = np.log(x)
    return {"mu": float(np.mean(logs)), "sigma": float(np.std(logs))}


def _fit_exponential(x: np.ndarray) -> dict[str, float]:
    return {"rate": float(1.0 / np.mean(x))}


def _fit_gamma(x: np.ndarray) -> dict[str, float]:
    shape, _, scale = stats.gamma.fit(x, floc=0)
    return {"shape": float(shape), "scale": float(scale)}


def _fit_weibull(x: np.ndarray) -> dict[str, float]:
    shape, _, scale = stats.weibull_min.fit(x, floc=0)
    return {"shape": float(shape), "scale": float(scale)}


def _fit_uniform(x: np.ndarray) -> dict[str, float]:
    return {"a": float(np.min(x)), "b": float(np.max(x))}


def _fit_beta(x: np.ndarray) -> dict[str, float]:
    alpha, beta, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    return {"alpha": float(alpha), "beta": float(beta)}


def _all_positive(x: np.ndarray) -> bool:
    return bool(np.all(x > 0))


_FIT_SPECS: dict[str, _FitSpec] = {
    "normal": _FitSpec("normal", 2, lambda x: True, _fit_normal),
    "lognormal": _FitSpec("lognormal", 2, _all_positive, _fit_lognormal),
    "exponential": _FitSpec("exponential", 1, lambda x: bool(np.all(x >= 0) and np.mean(x) > 0), _fit_exponential),
    "gamma": _FitSpec("gamma", 2, _all_positive, _fit_gamma),
    "weibull": _FitSpec("weibull", 2, _all_positive, _fit_weibull),
    "uniform": _FitSpec("uniform", 2, lambda x: bool(np.ptp(x) > 0), _fit_uniform),
    "beta": _FitSpec("beta", 2, lambda x: bool(np.all((x > 0) & (x < 1))), _fit_beta),
}

FITTABLE = tuple(_FIT_SPECS)


def fit_distributions(values: pd.Series, name: str, candidates: list[str]) -> list[DistributionFit]:
    x = values.dropna().astype(float).to_numpy()
    if len(x) < 5:
        raise InvalidInputError(f"«{name}» necesita al menos 5 observaciones para ajustar una distribución.")
    fits: list[DistributionFit] = []
    for key in candidates:
        spec = _FIT_SPECS.get(key)
        if spec is None:
            raise InvalidInputError(f"No se puede ajustar la distribución «{key}».")
        if not spec.valid(x):
            continue  # data outside the distribution's support: silently not a candidate
        try:
            params = spec.fit(x)
            frozen = get_distribution(key).build(params)
            ks = stats.kstest(x, frozen.cdf)
            log_likelihood = float(np.sum(frozen.logpdf(x)))
        except (ValueError, FloatingPointError, InvalidInputError):
            continue
        fits.append(
            DistributionFit(
                column=name,
                distribution=key,
                label=get_distribution(key).label,
                params=params,
                ks_statistic=float(ks.statistic),
                ks_p_value=float(ks.pvalue),
                aic=2 * spec.n_params - 2 * log_likelihood,
            )
        )
    return sorted(fits, key=lambda fit: fit.aic)


# ------------------------------------------------------------------ correlation
def correlation_matrix(frame: pd.DataFrame, columns: list[str], method: str = "pearson") -> CorrelationResult:
    if method not in CORRELATION_METHODS:
        raise InvalidInputError(f"Método de correlación inválido: {method}.")
    if len(columns) < 2:
        raise InvalidInputError("La correlación necesita al menos dos columnas numéricas.")
    data = pd.DataFrame({c: numeric_series(frame, c) for c in columns})
    size = len(columns)
    matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    p_values: list[list[float | None]] = [[None] * size for _ in range(size)]
    n_matrix = [[0] * size for _ in range(size)]
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for i in range(size):
        for j in range(size):
            pair = data.iloc[:, [i, j]].dropna() if i != j else data.iloc[:, [i]].dropna()
            n_matrix[i][j] = len(pair)
            if i == j:
                matrix[i][j] = 1.0
                continue
            if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            r, p = func(pair.iloc[:, 0], pair.iloc[:, 1])
            matrix[i][j], p_values[i][j] = finite(r), finite(p)
    return CorrelationResult(method, list(columns), matrix, p_values, n_matrix)


# ------------------------------------------------------------------ orchestration
def describe(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    group_by: str | None = None,
    confidence: float = 0.95,
    run_normality: bool = True,
    fit_candidates: list[str] | None = None,
    correlation_method: str | None = None,
) -> DescriptiveResult:
    check_confidence(confidence)
    if not numeric_columns and not categorical_columns:
        raise InvalidInputError("Seleccione al menos una columna para describir.")
    require_columns(frame, [*numeric_columns, *categorical_columns, *([group_by] if group_by else [])])

    groups: list[tuple[str | None, pd.DataFrame]] = [(None, frame)]
    if group_by:
        levels = frame[group_by].dropna().map(label_of)
        if levels.nunique() > MAX_GROUPS:
            raise InvalidInputError(f"La variable de agrupación «{group_by}» tiene más de {MAX_GROUPS} niveles.")
        key = frame[group_by].map(lambda v: label_of(v) if pd.notna(v) else None)
        groups += [(str(level), frame[key == level]) for level in sorted(levels.unique())]

    numeric_results: list[NumericSummary] = []
    normality: list[NormalityTest] = []
    plot_numeric: dict[str, dict[str, np.ndarray]] = {}
    for column in numeric_columns:
        series = numeric_series(frame, column)
        plot_numeric[column] = {}
        for group_name, subset in groups:
            values = series.loc[subset.index]
            if values.notna().sum() == 0:
                continue
            numeric_results.append(summarize_numeric(values, column, confidence, group_name))
            if run_normality:
                normality.extend(normality_tests(values, column, group_name))
            plot_numeric[column][group_name or "Todos"] = cap_for_plot(values.dropna().to_numpy())

    fits: list[DistributionFit] = []
    for column in numeric_columns if fit_candidates else []:
        fits.extend(fit_distributions(numeric_series(frame, column), column, fit_candidates or []))

    categorical = [frequency_table(frame[column], column) for column in categorical_columns]
    correlation = (
        correlation_matrix(frame, numeric_columns, correlation_method)
        if correlation_method and len(numeric_columns) >= 2
        else None
    )
    return DescriptiveResult(
        confidence=confidence,
        group_by=group_by,
        numeric=numeric_results,
        categorical=categorical,
        normality=normality,
        fits=fits,
        correlation=correlation,
        plot_data={"numeric": plot_numeric},
    )
