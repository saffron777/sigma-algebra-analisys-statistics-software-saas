"""Simple and multiple linear regression (OLS) with diagnostics and prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api import types as pdt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from sigma.domain.distributions import DistributionRef
from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.domain.results import (
    AnovaRow,
    Coefficient,
    DiagnosticTest,
    InfluentialPoint,
    PredictionRow,
    RegressionResult,
)
from sigma.domain.statistics.common import (
    check_confidence,
    finite,
    label_of,
    numeric_series,
    require_columns,
)

INTERCEPT = "Intercepto"
TOP_INFLUENTIAL = 10
GRID_POINTS = 100


def _level_sort_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


@dataclass
class _Design:
    """How predictors are encoded; reused to encode new observations for prediction."""

    numeric: list[str]
    categorical: dict[str, list[str]]  # column -> levels (first = reference)
    intercept: bool

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for name in self.numeric:
            columns[name] = numeric_series(frame, name)
        for name, levels in self.categorical.items():
            labels = frame[name].map(label_of)
            unknown = set(labels.unique()) - set(levels)
            if unknown:
                raise InvalidInputError(
                    f"La categoría «{', '.join(sorted(unknown))}» de «{name}» no existe en los datos del modelo."
                )
            for level in levels[1:]:
                columns[f"{name}[{level}]"] = (labels == level).astype(float)
        matrix = pd.DataFrame(columns, index=frame.index)
        if self.intercept:
            matrix.insert(0, INTERCEPT, 1.0)
        return matrix


def _build_design(frame: pd.DataFrame, predictors: list[str], categorical: set[str], intercept: bool) -> _Design:
    numeric: list[str] = []
    levels: dict[str, list[str]] = {}
    for name in predictors:
        series = frame[name]
        if name in categorical or not (pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series)):
            unique = sorted(series.map(label_of).unique(), key=_level_sort_key)
            if len(unique) < 2:
                raise InvalidInputError(f"La variable categórica «{name}» necesita al menos 2 niveles.")
            levels[name] = unique
        else:
            numeric.append(name)
    return _Design(numeric, levels, intercept)


def _equation(response: str, coefficients: list[Coefficient]) -> str:
    text = ""
    for c in coefficients:
        magnitude = f"{abs(c.estimate):.4g}"
        term = magnitude if c.term == INTERCEPT else f"{magnitude}·{c.term}"
        if not text:
            text = f"{'−' if c.estimate < 0 else ''}{term}"
        else:
            text += f" {'−' if c.estimate < 0 else '+'} {term}"
    return f"{response} = {text}"


def _diagnostics(model: Any, exog: pd.DataFrame, has_intercept: bool) -> list[DiagnosticTest]:
    resid = np.asarray(model.resid)
    out: list[DiagnosticTest] = []
    dw = float(durbin_watson(resid))
    dw_ok = 1.5 <= dw <= 2.5
    out.append(
        DiagnosticTest(
            "Durbin-Watson",
            dw,
            None,
            "sin autocorrelación evidente" if dw_ok else "posible autocorrelación de los residuos",
            dw_ok,
        )
    )
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    out.append(_normality_check("Jarque-Bera (normalidad)", float(jb_stat), float(jb_p)))
    if 3 <= len(resid) <= 5000 and np.ptp(resid) > 0:
        sw = stats.shapiro(resid)
        out.append(_normality_check("Shapiro-Wilk (normalidad)", float(sw.statistic), float(sw.pvalue)))
    if has_intercept and exog.shape[1] > 1:
        try:
            lm, lm_p, _, _ = het_breuschpagan(resid, exog.to_numpy())
            out.append(
                DiagnosticTest(
                    "Breusch-Pagan (homocedasticidad)",
                    float(lm),
                    float(lm_p),
                    "varianza constante" if lm_p >= 0.05 else "indicios de heterocedasticidad",
                    bool(lm_p >= 0.05),
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            pass
    return out


def _normality_check(name: str, statistic: float, p_value: float) -> DiagnosticTest:
    passed = p_value >= 0.05
    return DiagnosticTest(
        name,
        statistic,
        p_value,
        "residuos compatibles con normalidad" if passed else "los residuos se alejan de la normalidad",
        passed,
    )


def _vif(exog: pd.DataFrame, has_intercept: bool) -> dict[str, float | None]:
    names = [c for c in exog.columns if c != INTERCEPT]
    if len(names) < 2:
        return {name: 1.0 for name in names}
    matrix = exog.to_numpy() if has_intercept else np.column_stack([np.ones(len(exog)), exog.to_numpy()])
    offset = 0 if has_intercept else 1
    result: dict[str, float | None] = {}
    for position, name in enumerate(exog.columns):
        if name == INTERCEPT:
            continue
        result[name] = finite(variance_inflation_factor(matrix, position + offset))
    return result


def _influence(model: Any, rows: pd.Index) -> tuple[list[InfluentialPoint], dict[str, np.ndarray]]:
    influence = model.get_influence()
    leverage = np.asarray(influence.hat_matrix_diag)
    cooks = np.asarray(influence.cooks_distance[0])
    standardized = np.asarray(influence.resid_studentized_internal)
    observed, fitted, resid = (np.asarray(v) for v in (model.model.endog, model.fittedvalues, model.resid))
    order = np.argsort(cooks)[::-1][:TOP_INFLUENTIAL]
    points = [
        InfluentialPoint(
            row=int(rows[i]) + 1 if isinstance(rows[i], (int, np.integer)) else i + 1,
            observed=float(observed[i]),
            fitted=float(fitted[i]),
            residual=float(resid[i]),
            leverage=float(leverage[i]),
            cooks_distance=float(cooks[i]),
        )
        for i in order
    ]
    return points, {"leverage": leverage, "cooks": cooks, "standardized": standardized}


def _predictions(
    model: Any, design: _Design, new_rows: list[dict[str, Any]], predictors: list[str], confidence: float
) -> list[PredictionRow]:
    if not new_rows:
        return []
    frame = pd.DataFrame(new_rows)
    missing = [p for p in predictors if p not in frame.columns]
    if missing:
        raise InvalidInputError(f"Faltan valores para predecir: {', '.join(missing)}.")
    if frame[predictors].isna().any().any():
        raise InvalidInputError("Los valores para predecir no pueden estar vacíos.")
    exog = design.encode(frame)
    table = model.get_prediction(exog.to_numpy()).summary_frame(alpha=1 - confidence)
    return [
        PredictionRow(
            inputs={p: new_rows[i].get(p) for p in predictors},
            prediction=float(row["mean"]),
            mean_ci_low=float(row["mean_ci_lower"]),
            mean_ci_high=float(row["mean_ci_upper"]),
            pred_ci_low=float(row["obs_ci_lower"]),
            pred_ci_high=float(row["obs_ci_upper"]),
        )
        for i, (_, row) in enumerate(table.iterrows())
    ]


def _simple_line(model: Any, design: _Design, x_name: str, x: np.ndarray, confidence: float) -> dict[str, np.ndarray]:
    grid = np.linspace(float(np.min(x)), float(np.max(x)), GRID_POINTS)
    exog = design.encode(pd.DataFrame({x_name: grid}))
    frame = model.get_prediction(exog.to_numpy()).summary_frame(alpha=1 - confidence)
    return {
        "grid": grid,
        "fit": frame["mean"].to_numpy(),
        "ci_low": frame["mean_ci_lower"].to_numpy(),
        "ci_high": frame["mean_ci_upper"].to_numpy(),
        "pi_low": frame["obs_ci_lower"].to_numpy(),
        "pi_high": frame["obs_ci_upper"].to_numpy(),
    }


def fit_regression(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    categorical: list[str] | None = None,
    intercept: bool = True,
    confidence: float = 0.95,
    predict: list[dict[str, Any]] | None = None,
) -> RegressionResult:
    check_confidence(confidence)
    predictors = list(dict.fromkeys(predictors))
    if not predictors:
        raise InvalidInputError("Seleccione al menos una variable predictora.")
    if response in predictors:
        raise InvalidInputError("La variable respuesta no puede ser también predictora.")
    require_columns(frame, [response, *predictors])

    y_all = numeric_series(frame, response)
    used = frame[predictors].copy()
    used[response] = y_all
    complete = used.dropna()
    n_dropped = len(frame) - len(complete)
    design = _build_design(complete, predictors, set(categorical or []), intercept)
    exog = design.encode(complete)
    y = complete[response].astype(float)

    n_params = exog.shape[1]
    if len(complete) <= n_params:
        raise InvalidInputError(
            f"Hay {len(complete)} observaciones completas y {n_params} parámetros: "
            "se necesitan más observaciones que parámetros."
        )
    for column in exog.columns:
        if column != INTERCEPT and exog[column].nunique() < 2:
            raise ComputationError(f"La variable «{column}» es constante en los datos usados.")
    if np.linalg.matrix_rank(exog.to_numpy()) < n_params:
        raise ComputationError("Multicolinealidad perfecta: algún predictor es combinación lineal de otros.")

    model = sm.OLS(y.to_numpy(), exog.to_numpy()).fit()
    interval = np.asarray(model.conf_int(alpha=1 - confidence))
    vif = _vif(exog, intercept)
    coefficients = [
        Coefficient(
            term=str(name),
            estimate=float(model.params[i]),
            std_error=float(model.bse[i]),
            t_value=float(model.tvalues[i]),
            p_value=float(model.pvalues[i]),
            ci_low=float(interval[i, 0]),
            ci_high=float(interval[i, 1]),
            vif=vif.get(str(name)),
        )
        for i, name in enumerate(exog.columns)
    ]

    total_ss = float(model.centered_tss if intercept else model.uncentered_tss)
    df_model, df_resid = float(model.df_model), float(model.df_resid)
    anova = [
        AnovaRow(
            "Regresión",
            float(model.ess),
            df_model,
            float(model.ess) / df_model,
            finite(model.fvalue),
            finite(model.f_pvalue),
        ),
        AnovaRow("Residual", float(model.ssr), df_resid, float(model.mse_resid), None, None),
        AnovaRow("Total", total_ss, df_model + df_resid, None, None, None),
    ]
    influential, influence_arrays = _influence(model, complete.index)
    is_simple = len(predictors) == 1 and not design.categorical
    notes = [
        f"Categoría de referencia de «{name}»: {levels[0]} (los coeficientes comparan contra ella)."
        for name, levels in design.categorical.items()
    ]
    if n_dropped:
        notes.append(f"Se excluyeron {n_dropped} fila(s) con valores faltantes (eliminación por lista).")

    plot_data: dict[str, Any] = {
        "observed": y.to_numpy(),
        "fitted": np.asarray(model.fittedvalues),
        "residuals": np.asarray(model.resid),
        **influence_arrays,
    }
    if is_simple:
        x = complete[predictors[0]].astype(float).to_numpy()
        plot_data.update(x=x, x_name=predictors[0], line=_simple_line(model, design, predictors[0], x, confidence))

    return RegressionResult(
        response=response,
        predictors=predictors,
        is_simple=is_simple,
        has_intercept=intercept,
        confidence=confidence,
        equation=_equation(response, coefficients),
        n_obs=int(model.nobs),
        n_dropped=n_dropped,
        df_model=df_model,
        df_resid=df_resid,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=finite(model.fvalue),
        f_p_value=finite(model.f_pvalue),
        residual_std_error=math.sqrt(float(model.mse_resid)),
        aic=float(model.aic),
        bic=float(model.bic),
        log_likelihood=float(model.llf),
        coefficients=coefficients,
        anova=anova,
        diagnostics=_diagnostics(model, exog, intercept),
        influential=influential,
        predictions=_predictions(model, design, predict or [], predictors, confidence),
        f_distribution=DistributionRef("f", {"dfn": df_model, "dfd": df_resid}),
        t_distribution=DistributionRef("t", {"df": df_resid}),
        notes=notes,
        plot_data=plot_data,
    )
