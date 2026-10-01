"""Survey estimation: proportions and means with margins of error for probability samples.

Two estimation paths:

* **Simple random sample** (no weights/strata/clusters): textbook formulas, with the finite
  population correction when ``population_size`` is given.
* **Complex design** (weights, strata and/or clusters): ratio estimator with Taylor linearisation
  of the variance (ultimate-cluster approximation).

Confidence intervals for proportions use the *effective sample size* ``n_eff = p·q / Var(p)``, so
every interval method (Wald, Wilson, Agresti-Coull, Clopper-Pearson) accounts for the design.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.errors import InvalidInputError
from sigma.domain.results import EstimateRow, SurveyEstimateResult
from sigma.domain.statistics.common import check_confidence, label_of, numeric_series, require_columns

CI_METHODS = ("wilson", "wald", "agresti_coull", "clopper_pearson")
CI_LABELS = {
    "wilson": "Wilson",
    "wald": "Wald (normal)",
    "agresti_coull": "Agresti-Coull",
    "clopper_pearson": "Clopper-Pearson (exacto)",
}
MAX_CATEGORIES = 30


def proportion_interval(p: float, n_eff: float, confidence: float, method: str) -> tuple[float, float]:
    """Confidence interval for a proportion ``p`` observed with effective sample size ``n_eff``."""
    if method not in CI_METHODS:
        raise InvalidInputError(f"Método de intervalo inválido «{method}».")
    alpha = 1 - confidence
    z = float(stats.norm.isf(alpha / 2))
    x = p * n_eff
    if method == "wald":
        half = z * math.sqrt(p * (1 - p) / n_eff)
        return max(0.0, p - half), min(1.0, p + half)
    if method == "wilson":
        centre = (p + z**2 / (2 * n_eff)) / (1 + z**2 / n_eff)
        half = z * math.sqrt(p * (1 - p) / n_eff + z**2 / (4 * n_eff**2)) / (1 + z**2 / n_eff)
        return max(0.0, centre - half), min(1.0, centre + half)
    if method == "agresti_coull":
        n_tilde = n_eff + z**2
        p_tilde = (x + z**2 / 2) / n_tilde
        half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        return max(0.0, p_tilde - half), min(1.0, p_tilde + half)
    low = 0.0 if x <= 0 else float(stats.beta.ppf(alpha / 2, x, n_eff - x + 1))
    high = 1.0 if x >= n_eff else float(stats.beta.ppf(1 - alpha / 2, x + 1, n_eff - x))
    return low, high


def _fpc(n: int, population: int | None) -> float:
    if population is None:
        return 1.0
    if population < n:
        raise InvalidInputError("La población N no puede ser menor que el tamaño de la muestra.")
    return (population - n) / (population - 1) if population > 1 else 0.0


# --------------------------------------------------------------------------- linearisation
def _linearized_variance(
    y: np.ndarray, w: np.ndarray, strata: np.ndarray, psu: np.ndarray, use_fpc: bool, clustered: bool
) -> tuple[float, float, int]:
    """Return (ratio estimate, variance, degrees of freedom) of sum(w*y)/sum(w)."""
    total_w = w.sum()
    ratio = float((w * y).sum() / total_w)
    z = w * (y - ratio) / total_w
    frame = pd.DataFrame({"z": z, "w": w, "stratum": strata, "psu": psu})
    variance, n_psu, n_strata = 0.0, 0, 0
    for _, stratum in frame.groupby("stratum"):
        totals = stratum.groupby("psu")["z"].sum().to_numpy()
        n_h = len(totals)
        if n_h < 2:
            raise InvalidInputError(
                "Cada estrato necesita al menos 2 conglomerados (o unidades) para estimar la varianza."
            )
        f_h = 0.0
        if use_fpc and not clustered:
            f_h = min(n_h / float(stratum["w"].sum()), 1.0)
        variance += (1 - f_h) * n_h / (n_h - 1) * float(((totals - totals.mean()) ** 2).sum())
        n_psu, n_strata = n_psu + n_h, n_strata + 1
    return ratio, variance, max(n_psu - n_strata, 1)


def _complex_columns(
    frame: pd.DataFrame, weight_column: str | None, strata_column: str | None, cluster_column: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    weights = numeric_series(frame, weight_column).to_numpy() if weight_column else np.ones(n)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise InvalidInputError("Los pesos deben ser números positivos y sin valores vacíos.")
    strata = frame[strata_column].map(label_of).to_numpy() if strata_column else np.zeros(n, dtype=object)
    psu = frame[cluster_column].map(label_of).to_numpy() if cluster_column else np.arange(n)
    return weights, strata, psu


def estimate_survey(
    frame: pd.DataFrame,
    variable: str,
    kind: str = "proportion",
    confidence: float = 0.95,
    methods: list[str] | None = None,
    population_size: int | None = None,
    weight_column: str | None = None,
    strata_column: str | None = None,
    cluster_column: str | None = None,
    apply_fpc: bool = True,
) -> SurveyEstimateResult:
    check_confidence(confidence)
    if kind not in ("proportion", "mean"):
        raise InvalidInputError("El tipo de estimación debe ser «proportion» o «mean».")
    require_columns(frame, [c for c in (variable, weight_column, strata_column, cluster_column) if c])
    methods = methods or ["wilson"]
    complex_design = any((weight_column, strata_column, cluster_column))

    needed = [c for c in (variable, weight_column, strata_column, cluster_column) if c]
    data = frame.dropna(subset=needed).reset_index(drop=True)
    n = len(data)
    if n < 2:
        raise InvalidInputError("Se necesitan al menos 2 observaciones válidas.")
    notes: list[str] = []
    if n < len(frame):
        notes.append(f"Se excluyeron {len(frame) - n} fila(s) con valores vacíos.")

    if complex_design:
        weights, strata, psu = _complex_columns(data, weight_column, strata_column, cluster_column)
        if population_size:
            notes.append(
                "Con pesos/estratos/conglomerados la población N no se usa: "
                "la corrección finita se deriva de los pesos."
            )
        design_label = "Diseño complejo (linealización de Taylor)"
    else:
        weights = np.full(n, (population_size / n) if population_size else 1.0)
        strata, psu = np.zeros(n, dtype=object), np.arange(n)
        design_label = "Muestreo aleatorio simple" + (" con población finita" if population_size else "")
    fpc_active = bool(population_size) if not complex_design else apply_fpc
    clustered = bool(cluster_column)

    rows: list[EstimateRow] = []
    deff_values: list[float] = []
    if kind == "proportion":
        labels = data[variable].map(label_of)
        counts = labels.value_counts()
        if len(counts) > MAX_CATEGORIES:
            raise InvalidInputError(
                f"«{variable}» tiene más de {MAX_CATEGORIES} categorías; no es una pregunta categórica."
            )
        for category in sorted(counts.index):
            indicator = (labels == category).to_numpy(dtype=float)
            p, var, n_eff = _proportion_estimate(
                indicator, weights, strata, psu, complex_design, clustered, fpc_active, population_size, n, deff_values
            )
            for method in methods:
                low, high = proportion_interval(p, n_eff, confidence, method)
                rows.append(
                    EstimateRow(
                        label=str(category),
                        n=n,
                        estimate=p,
                        std_error=math.sqrt(var),
                        margin_of_error=(high - low) / 2,
                        ci_low=low,
                        ci_high=high,
                        method=CI_LABELS[method],
                        count=float(counts[category]),
                    )
                )
    else:
        values = numeric_series(data, variable).to_numpy()
        if complex_design:
            mean, var, df = _linearized_variance(values, weights, strata, psu, fpc_active, clustered)
            sample_var = _weighted_variance(values, weights, mean)
            srs_var = sample_var / n * (1 - n / weights.sum() if fpc_active and not clustered else 1.0)
            deff_values.append(var / srs_var if srs_var > 0 else float("nan"))
        else:
            mean = float(values.mean())
            var = float(values.var(ddof=1)) / n * _fpc(n, population_size if fpc_active else None)
            df = n - 1
        t = float(stats.t.isf((1 - confidence) / 2, df))
        se = math.sqrt(var)
        rows.append(EstimateRow(variable, n, mean, se, t * se, mean - t * se, mean + t * se, "t de Student"))

    deff = float(np.nanmean(deff_values)) if deff_values else None
    if complex_design and weight_column:
        kish = n * float((weights**2).sum()) / float(weights.sum()) ** 2
        notes.append(f"Efecto de los pesos (Kish) = {kish:.3f}; tamaño efectivo por pesos = {n / kish:.1f}.")
    return SurveyEstimateResult(
        kind=kind,
        variable=variable,
        design=design_label,
        confidence=confidence,
        n=n,
        population_size=population_size,
        design_effect=deff,
        effective_n=(n / deff) if deff and deff > 0 else None,
        rows=rows,
        notes=notes,
    )


def _weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    total = weights.sum()
    return float((weights * (values - mean) ** 2).sum() / total * len(values) / (len(values) - 1))


def _proportion_estimate(
    indicator: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
    complex_design: bool,
    clustered: bool,
    use_fpc: bool,
    population: int | None,
    n: int,
    deff_values: list[float],
) -> tuple[float, float, float]:
    """Return (p, variance of p, effective n) for one category."""
    if complex_design:
        p, var, _ = _linearized_variance(indicator, weights, strata, psu, use_fpc, clustered)
        srs_var = p * (1 - p) / n
        if srs_var > 0 and var > 0:
            deff_values.append(var / srs_var)
            return p, var, p * (1 - p) / var
        return p, var, float(n)
    p = float(indicator.mean())
    fpc = _fpc(n, population if use_fpc else None)
    var = p * (1 - p) / n * fpc
    return p, var, n / max(fpc, 1e-12)
