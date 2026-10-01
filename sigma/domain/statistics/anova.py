"""One-way and two-way ANOVA with assumption checks, effect sizes and post-hoc tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.domain.results import (
    AnovaResult,
    AnovaRow,
    AssumptionCheck,
    FTestRef,
    GroupStats,
    PostHocRow,
)
from sigma.domain.statistics.common import (
    cap_for_plot,
    check_alpha,
    label_of,
    numeric_series,
    require_columns,
)

MAX_LEVELS = 30
SS_TYPES = (1, 2, 3)


def _level_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def _group_stats(label: str, values: np.ndarray, alpha: float) -> GroupStats:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    margin = stats.t.isf(alpha / 2, n - 1) * sem if n > 1 else float("nan")
    return GroupStats(label, n, mean, std, sem, mean - margin, mean + margin)


def _prepare(frame: pd.DataFrame, response: str, factors: list[str]) -> pd.DataFrame:
    require_columns(frame, [response, *factors])
    data = pd.DataFrame({"y": numeric_series(frame, response)})
    for i, factor in enumerate(factors):
        data[f"f{i}"] = frame[factor].map(lambda v: label_of(v) if pd.notna(v) else None)
    clean = data.dropna()
    for i, factor in enumerate(factors):
        if clean[f"f{i}"].nunique() < 2:
            raise InvalidInputError(f"El factor «{factor}» necesita al menos 2 niveles.")
        if clean[f"f{i}"].nunique() > MAX_LEVELS:
            raise InvalidInputError(f"El factor «{factor}» tiene más de {MAX_LEVELS} niveles.")
    return clean


def _tukey(values: np.ndarray, labels: np.ndarray, alpha: float, prefix: str = "") -> list[PostHocRow]:
    """Tukey-Kramer pairwise comparisons; ``mean_diff`` is mean(group2) − mean(group1)."""
    levels = sorted(set(labels), key=_level_key)
    samples = [values[labels == level] for level in levels]
    outcome = stats.tukey_hsd(*samples)
    interval = outcome.confidence_interval(confidence_level=1 - alpha)
    rows: list[PostHocRow] = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            p = float(outcome.pvalue[i, j])
            rows.append(
                PostHocRow(
                    f"{prefix}{levels[i]}",
                    f"{prefix}{levels[j]}",
                    -float(outcome.statistic[i, j]),
                    p,
                    -float(interval.high[i, j]),
                    -float(interval.low[i, j]),
                    p < alpha,
                )
            )
    return rows


def _assumptions(residuals: np.ndarray, samples: list[np.ndarray], alpha: float) -> list[AssumptionCheck]:
    checks: list[AssumptionCheck] = []
    if 3 <= len(residuals) <= 5000 and np.ptp(residuals) > 0:
        sw = stats.shapiro(residuals)
        checks.append(
            AssumptionCheck(
                "Normalidad de los residuos",
                "Shapiro-Wilk",
                float(sw.statistic),
                float(sw.pvalue),
                bool(sw.pvalue >= alpha),
                "Los residuos son compatibles con una distribución normal."
                if sw.pvalue >= alpha
                else "Se rechaza la normalidad de los residuos; considere Kruskal-Wallis.",
            )
        )
    if all(len(s) >= 2 for s in samples) and any(np.ptp(s) > 0 for s in samples):
        lev = stats.levene(*samples, center="median")
        checks.append(
            AssumptionCheck(
                "Homogeneidad de varianzas",
                "Levene (Brown-Forsythe)",
                float(lev.statistic),
                float(lev.pvalue),
                bool(lev.pvalue >= alpha),
                "Las varianzas son homogéneas."
                if lev.pvalue >= alpha
                else "Se rechaza la igualdad de varianzas; considere la prueba de Welch.",
            )
        )
        bar = stats.bartlett(*samples)
        checks.append(
            AssumptionCheck(
                "Homogeneidad de varianzas",
                "Bartlett",
                float(bar.statistic),
                float(bar.pvalue),
                bool(bar.pvalue >= alpha),
                "Las varianzas son homogéneas (sensible a la no normalidad)."
                if bar.pvalue >= alpha
                else "Se rechaza la igualdad de varianzas (sensible a la no normalidad).",
            )
        )
    return checks


# ------------------------------------------------------------------------- one way
def _one_way(
    data: pd.DataFrame, response: str, factor: str, alpha: float, post_hoc: bool, n_dropped: int
) -> AnovaResult:
    levels = sorted(data["f0"].unique(), key=_level_key)
    samples = [data.loc[data["f0"] == level, "y"].to_numpy() for level in levels]
    if any(len(s) < 2 for s in samples):
        raise InvalidInputError("Cada grupo necesita al menos 2 observaciones.")
    k, n_total = len(samples), len(data)
    grand = float(data["y"].mean())
    ss_between = float(sum(len(s) * (s.mean() - grand) ** 2 for s in samples))
    ss_within = float(sum(((s - s.mean()) ** 2).sum() for s in samples))
    ss_total = ss_between + ss_within
    df_b, df_w = k - 1, n_total - k
    if ss_within == 0:
        raise ComputationError("La variabilidad dentro de los grupos es cero: no se puede calcular F.")
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    f_value = ms_b / ms_w
    p = float(stats.f.sf(f_value, df_b, df_w))
    omega = max(0.0, (ss_between - df_b * ms_w) / (ss_total + ms_w))
    table = [
        AnovaRow(
            f"Entre grupos ({factor})",
            ss_between,
            df_b,
            ms_b,
            f_value,
            p,
            ss_between / ss_total,
            ss_between / (ss_between + ss_within),
            omega,
        ),
        AnovaRow("Dentro de grupos (error)", ss_within, df_w, ms_w, None, None),
        AnovaRow("Total", ss_total, n_total - 1, None, None, None),
    ]
    residuals = np.concatenate([s - s.mean() for s in samples])
    welch = anova_oneway(samples, use_var="unequal", welch_correction=True)
    welch_result = {
        "statistic": float(welch.statistic),
        "df_num": float(welch.df_num),
        "df_den": float(welch.df_denom),
        "p_value": float(welch.pvalue),
    }
    kruskal = stats.kruskal(*samples)
    return AnovaResult(
        design="one_way",
        response=response,
        factors=[factor],
        alpha=alpha,
        n_obs=n_total,
        n_dropped=n_dropped,
        table=table,
        groups=[_group_stats(lv, s, alpha) for lv, s in zip(levels, samples, strict=True)],
        assumptions=_assumptions(residuals, samples, alpha),
        post_hoc=_tukey(data["y"].to_numpy(), data["f0"].to_numpy(), alpha) if post_hoc else [],
        post_hoc_method="Tukey HSD" if post_hoc else None,
        f_tests=[FTestRef(factor, f_value, df_b, df_w, p)],
        welch=welch_result,
        kruskal={"statistic": float(kruskal.statistic), "df": float(k - 1), "p_value": float(kruskal.pvalue)},
        plot_data={
            "groups": {lv: cap_for_plot(s) for lv, s in zip(levels, samples, strict=True)},
            "residuals": residuals,
            "fitted": np.concatenate([np.full(len(s), s.mean()) for s in samples]),
        },
    )


# ------------------------------------------------------------------------- two way
def _two_way(
    data: pd.DataFrame,
    response: str,
    factors: list[str],
    alpha: float,
    interaction: bool,
    ss_type: int,
    post_hoc: bool,
    n_dropped: int,
) -> AnovaResult:
    terms = "C(f0, Sum) * C(f1, Sum)" if interaction else "C(f0, Sum) + C(f1, Sum)"
    model = smf.ols(f"y ~ {terms}", data=data).fit()
    if model.df_resid <= 0:
        raise InvalidInputError("No hay observaciones suficientes para estimar el error del modelo.")
    anova = sm.stats.anova_lm(model, typ=ss_type)
    names = {
        "C(f0, Sum)": factors[0],
        "C(f1, Sum)": factors[1],
        "C(f0, Sum):C(f1, Sum)": f"{factors[0]} × {factors[1]}",
        "Residual": "Error (residual)",
    }
    anova = anova.drop(index="Intercept", errors="ignore")
    ss_error = float(anova.loc["Residual", "sum_sq"])
    df_error = float(anova.loc["Residual", "df"])
    ss_total = float(anova["sum_sq"].sum())
    ms_error = ss_error / df_error
    rows: list[AnovaRow] = []
    f_tests: list[FTestRef] = []
    for term, record in anova.iterrows():
        label = names.get(str(term), str(term))
        ss, df = float(record["sum_sq"]), float(record["df"])
        if term == "Residual":
            rows.append(AnovaRow(label, ss, df, ss / df, None, None))
            continue
        f_value, p = float(record["F"]), float(record["PR(>F)"])
        rows.append(
            AnovaRow(
                label,
                ss,
                df,
                ss / df,
                f_value,
                p,
                ss / ss_total,
                ss / (ss + ss_error),
                max(0.0, (ss - df * ms_error) / (ss_total + ms_error)),
            )
        )
        f_tests.append(FTestRef(label, f_value, df, df_error, p))
    rows.append(AnovaRow("Total", ss_total, float(anova["df"].sum()), None, None, None))

    cells = data.groupby(["f0", "f1"])["y"]
    cell_stats: list[GroupStats] = []
    cell_plot: list[dict[str, Any]] = []
    samples: list[np.ndarray] = []
    for (a, b), values in cells:
        arr = values.to_numpy()
        samples.append(arr)
        cell_stats.append(_group_stats(f"{a} × {b}", arr, alpha))
        cell_plot.append(
            {
                "a": a,
                "b": b,
                "mean": float(arr.mean()),
                "sem": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            }
        )
    notes_order = sorted(data["f0"].unique(), key=_level_key)
    hoc: list[PostHocRow] = []
    if post_hoc:
        for i, factor in enumerate(factors):
            hoc.extend(_tukey(data["y"].to_numpy(), data[f"f{i}"].to_numpy(), alpha, prefix=f"{factor}: "))
    return AnovaResult(
        design="two_way",
        response=response,
        factors=factors,
        alpha=alpha,
        n_obs=len(data),
        n_dropped=n_dropped,
        table=rows,
        groups=cell_stats,
        assumptions=_assumptions(np.asarray(model.resid), samples, alpha),
        post_hoc=hoc,
        post_hoc_method="Tukey HSD (comparaciones marginales por factor)" if post_hoc else None,
        f_tests=f_tests,
        ss_type=ss_type,
        plot_data={
            "cells": cell_plot,
            "residuals": np.asarray(model.resid),
            "fitted": np.asarray(model.fittedvalues),
            "levels_a": notes_order,
            "groups": {f"{c.group}": cap_for_plot(s) for c, s in zip(cell_stats, samples, strict=True)},
        },
    )


def run_anova(
    frame: pd.DataFrame,
    response: str,
    factors: list[str],
    alpha: float = 0.05,
    interaction: bool = True,
    ss_type: int = 2,
    post_hoc: bool = True,
) -> AnovaResult:
    check_alpha(alpha)
    factors = list(dict.fromkeys(factors))
    if len(factors) not in (1, 2):
        raise InvalidInputError("El ANOVA admite uno o dos factores.")
    if response in factors:
        raise InvalidInputError("La variable respuesta no puede ser también un factor.")
    if ss_type not in SS_TYPES:
        raise InvalidInputError("El tipo de suma de cuadrados debe ser 1, 2 o 3.")
    data = _prepare(frame, response, factors)
    n_dropped = len(frame) - len(data)
    if len(factors) == 1:
        return _one_way(data, response, factors[0], alpha, post_hoc, n_dropped)
    return _two_way(data, response, factors, alpha, interaction, ss_type, post_hoc, n_dropped)
