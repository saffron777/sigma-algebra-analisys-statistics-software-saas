"""Survey tabulation: cross-tabs with association tests and Likert-scale summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.distributions import DistributionRef
from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.domain.results import CrosstabResult, LikertResult
from sigma.domain.statistics.common import label_of, require_columns

MAX_LEVELS = 40


def _level_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def crosstab(frame: pd.DataFrame, row_variable: str, col_variable: str) -> CrosstabResult:
    if row_variable == col_variable:
        raise InvalidInputError("Las variables de filas y columnas deben ser distintas.")
    require_columns(frame, [row_variable, col_variable])
    pair = frame[[row_variable, col_variable]].dropna()
    if pair.empty:
        raise InvalidInputError("No hay filas completas en las dos variables elegidas.")
    table = pd.crosstab(pair[row_variable].map(label_of), pair[col_variable].map(label_of))
    table = table.reindex(index=sorted(table.index, key=_level_key), columns=sorted(table.columns, key=_level_key))
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise InvalidInputError("Cada variable necesita al menos 2 categorías.")
    if max(table.shape) > MAX_LEVELS:
        raise InvalidInputError(f"Las variables no pueden tener más de {MAX_LEVELS} categorías.")
    observed = table.to_numpy(dtype=float)
    chi2, p, df, expected = stats.chi2_contingency(observed, correction=False)
    n = float(observed.sum())
    row_totals, col_totals = observed.sum(axis=1, keepdims=True), observed.sum(axis=0, keepdims=True)
    residuals = (observed - expected) / np.sqrt(expected * (1 - row_totals / n) * (1 - col_totals / n))
    k = min(observed.shape) - 1
    notes = []
    low = float((expected < 5).mean())
    if low > 0.2:
        notes.append(f"Aviso: {low:.0%} de las frecuencias esperadas son menores que 5; interprete χ² con cautela.")
    return CrosstabResult(
        row_variable=row_variable,
        col_variable=col_variable,
        row_labels=[str(v) for v in table.index],
        col_labels=[str(v) for v in table.columns],
        observed=observed.tolist(),
        expected=expected.tolist(),
        row_percent=(100 * observed / row_totals).tolist(),
        col_percent=(100 * observed / col_totals).tolist(),
        total_percent=(100 * observed / n).tolist(),
        standardized_residuals=residuals.tolist(),
        chi2=float(chi2),
        df=int(df),
        p_value=float(p),
        cramers_v=math.sqrt(chi2 / (n * k)),
        n=n,
        notes=notes,
        distribution=DistributionRef("chi2", {"df": float(df)}),
    )


def likert_summary(frame: pd.DataFrame, items: list[str]) -> LikertResult:
    items = list(dict.fromkeys(items))
    if len(items) < 1:
        raise InvalidInputError("Seleccione al menos un ítem de la escala.")
    require_columns(frame, items)
    data = frame[items]
    labels = data.apply(lambda column: column.map(lambda v: label_of(v) if pd.notna(v) else None))
    levels = sorted({v for v in labels.to_numpy().ravel() if v is not None}, key=_level_key)
    if len(levels) < 2:
        raise InvalidInputError("La escala necesita al menos 2 niveles de respuesta.")
    if len(levels) > 15:
        raise InvalidInputError("Hay demasiados niveles de respuesta para una escala Likert.")
    counts = [[int((labels[item] == level).sum()) for level in levels] for item in items]
    percents = [[100 * c / sum(row) if sum(row) else 0.0 for c in row] for row in counts]
    numeric_levels = [_level_key(level)[0] == 0 for level in levels]
    scores = np.array([_level_key(level)[1] for level in levels])
    means: list[float] = []
    top2: list[float] = []
    for row in counts:
        total = sum(row)
        means.append(float(np.dot(row, scores) / total) if all(numeric_levels) and total else float("nan"))
        top2.append(100 * (row[-1] + row[-2]) / total if total else float("nan"))
    numeric = data.apply(pd.to_numeric, errors="coerce")
    complete = numeric.dropna()
    alpha = _cronbach_alpha(complete) if len(items) >= 2 and all(numeric_levels) and len(complete) >= 3 else None
    return LikertResult(
        items=list(items),
        levels=levels,
        counts=counts,
        percents=percents,
        means=means,
        top2box=top2,
        cronbach_alpha=alpha,
        n_items=len(items),
        n_complete=len(complete),
    )


def _cronbach_alpha(complete: pd.DataFrame) -> float | None:
    k = complete.shape[1]
    item_variances = complete.var(ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        raise ComputationError("La varianza total de la escala es cero.")
    return float(k / (k - 1) * (1 - item_variances / total_variance))
