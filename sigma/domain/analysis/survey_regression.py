"""Regresión de encuestas con pesos muestrales (SURVEYREG y SURVEYLOGISTIC)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import scatter_chart
from sigma.domain.analysis.modeling import (
    PREDICTORS,
    build_design,
    coefficient_table,
    dropped_note,
    encode_binary,
    need_enough,
    prediction_frame,
)
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import ComputationError

MODULE = "encuestas"
GROUP_REGRESSION = "Regresión de encuestas"

NUMERIC_RESPONSE = RoleSpec("response", "Respuesta (Y, numérica)", "numeric")
BINARY_RESPONSE = RoleSpec("response", "Respuesta (Y, binaria)", "categorical")
WEIGHT = RoleSpec("weight", "Peso muestral (opcional)", "numeric", optional=True, help="Sin pesos se supone muestreo aleatorio simple (todos pesan igual).")


def _weighted_design(frame: pd.DataFrame, response: str, predictors: list[str], weight_col: str | None):
    extra = {"weights": weight_col} if weight_col else None
    design = build_design(frame, response, predictors, extra=extra)
    weights = design.data[design.extra["weights"]].to_numpy(dtype=float) if weight_col else np.ones(len(design.data))
    return design, weights


def _survey_ols(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors = roles["response"], roles["predictors"]
    design, weights = _weighted_design(frame, response, predictors, roles.get("weight"))
    need_enough(design, len(predictors) + 1)
    try:
        result = smf.wls(design.formula, design.data, weights=weights).fit()
    except Exception as error:  # noqa: BLE001
        raise ComputationError(f"No se pudo ajustar la regresión: {str(error)[:200]}.") from None
    out = Output("Regresión lineal con pesos muestrales", f"n = {len(design.data)} · {len(predictors)} predictor(es)")
    dropped_note(out, design)
    out.tiles([("Observaciones", len(design.data)), ("R²", float(result.rsquared)), ("R² ajustado", float(result.rsquared_adj))])
    out.alert("info", "Mínimos cuadrados ponderados por el peso muestral: cada observación aporta según su probabilidad de selección.")
    coefficient_table(out, design, result, p["alpha"], stat_label="t", title="Coeficientes")
    y = design.data[design.response].to_numpy(dtype=float)
    fitted = np.asarray(result.fittedvalues, dtype=float)
    out.chart(scatter_chart("Observado vs ajustado", "Observado", "Ajustado", [{"name": "Pares", "x": y, "y": fitted}]))
    out.dataset("predicciones", "Predicciones y residuos", prediction_frame(frame, design, ajustado=fitted, residuo=y - fitted))
    return out.set(n=len(design.data), r2=float(result.rsquared), r2_adj=float(result.rsquared_adj)).build()


def _survey_logistic(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors = roles["response"], roles["predictors"]
    binary, event, nonevent = encode_binary(frame[response], response)
    framed = frame.copy()
    framed[response] = binary
    design, weights = _weighted_design(framed, response, predictors, roles.get("weight"))
    need_enough(design, len(predictors) + 1)
    try:
        result = smf.glm(design.formula, design.data, family=sm.families.Binomial(), freq_weights=weights).fit()
    except Exception as error:  # noqa: BLE001
        raise ComputationError(f"No se pudo ajustar la regresión logística: {str(error)[:200]}.") from None
    out = Output("Regresión logística con pesos muestrales", f"n = {len(design.data)} · evento «{event}»")
    dropped_note(out, design)
    out.tiles([("Observaciones", len(design.data)), ("Evento", event), ("AIC", float(result.aic)), ("BIC", float(result.bic))])
    out.alert("info", "Regresión logística ponderada: los pesos muestrales ajustan la contribución de cada observación.")
    coefficient_table(out, design, result, p["alpha"], stat_label="z", exponentiate="Odds ratio", title="Coeficientes")
    y = design.data[design.response].to_numpy(dtype=float)
    fitted = np.asarray(result.predict(design.data), dtype=float)
    out.dataset("predicciones", "Probabilidad estimada y residuos", prediction_frame(frame, design, probabilidad=fitted, residuo=y - fitted))
    return out.set(n=len(design.data), aic=float(result.aic), bic=float(result.bic)).build()


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)

SPECS = (
    _spec(
        "survey_ols",
        "Regresión lineal de encuestas (SURVEYREG)",
        GROUP_REGRESSION,
        "Regresión lineal por mínimos cuadrados ponderados con pesos muestrales (y estratos/conglomerados opcionales).",
        _survey_ols,
        roles=(NUMERIC_RESPONSE, PREDICTORS, WEIGHT),
        params=(ALPHA,),
        outputs=(OutputSpec("predicciones", "Predicciones y residuos"),),
        example={"roles": {"response": "y", "predictors": ["x1", "x2"], "weight": "x3"}, "params": {}},
    ),
    _spec(
        "survey_logistic",
        "Regresión logística de encuestas (SURVEYLOGISTIC)",
        GROUP_REGRESSION,
        "Regresión logística ponderada con pesos muestrales para una respuesta binaria.",
        _survey_logistic,
        roles=(BINARY_RESPONSE, PREDICTORS, WEIGHT),
        params=(ALPHA,),
        outputs=(OutputSpec("predicciones", "Probabilidad estimada y residuos"),),
        example={"roles": {"response": "h", "predictors": ["x1", "x2"], "weight": "x3"}, "params": {}},
    ),
)

registry.register(*SPECS)
