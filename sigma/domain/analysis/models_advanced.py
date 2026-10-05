"""GAM, PLS y modelos loglineales: extensiones de la página de modelos."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import PLSRegression
from statsmodels.gam.api import BSplines, GLMGam

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import scatter_chart
from sigma.domain.analysis.modeling import RESPONSE
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "modelos"
GROUP_ADVANCED = "Modelos aditivos y reducción"
GROUP_CATEGORICAL = "Tablas de contingencia"

NUMERIC_PREDICTORS = RoleSpec("predictors", "Predictores numéricos", "numeric", multiple=True, min_count=1, max_count=30)
COUNT = RoleSpec("count", "Frecuencia (conteo)", "numeric", help="El número de casos de cada combinación de factores.")
FACTORS = RoleSpec("factors", "Factores (variables categóricas)", "categorical", multiple=True, min_count=1, max_count=6)


def _numeric_frame(frame: pd.DataFrame, response: str, predictors: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = [response, *predictors]
    data = frame[cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    if len(data) < 12:
        raise InvalidInputError(f"Se necesitan al menos 12 filas completas (hay {len(data)}).")
    return data, data[response].to_numpy(dtype=float), data[predictors].to_numpy(dtype=float)


# ============================================================================ GAM
def _gam(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors = roles["response"], roles["predictors"]
    data, y, X = _numeric_frame(frame, response, predictors)
    df = int(p["splines"])
    k = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bs = BSplines(X, df=[df] * k, degree=[3] * k)
        gam = GLMGam(y, X, smoother=bs).fit()
    fitted = np.asarray(gam.fittedvalues, dtype=float)
    r2 = float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    out = Output(
        "Modelo aditivo generalizado (GAM)",
        f"n = {len(data)} · {k} predictor(es) · splines con {df} grados de libertad",
    )
    out.tiles([("Observaciones", len(data)), ("AIC", float(gam.aic)), ("R²", r2), ("Parámetros", int(len(gam.params)))])
    out.alert("info", "El GAM suma funciones suaves (splines) de cada predictor: capta relaciones no lineales sin imponer una forma.")
    lo, hi = float(y.min()), float(y.max())
    out.chart(
        scatter_chart(
            "Observado vs ajustado",
            "Observado",
            "Ajustado",
            [{"name": "Pares", "x": y, "y": fitted}],
            lines=[{"name": "Identidad", "x": [lo, hi], "y": [lo, hi], "dash": "dash"}],
        )
    )
    out.dataset("predicciones", "Predicciones y residuos", pd.DataFrame({"observado": y, "ajustado": fitted, "residuo": y - fitted}, index=data.index))
    return out.set(n=len(data), aic=float(gam.aic), r2=r2).build()


# ============================================================================ PLS
def _pls(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors = roles["response"], roles["predictors"]
    data, y, X = _numeric_frame(frame, response, predictors)
    n_comp = min(int(p["components"]), X.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=bool(p["scale"])).fit(X, y)
    fitted = pls.predict(X).ravel()
    r2 = float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    out = Output(
        "Mínimos cuadrados parciales (PLS)",
        f"n = {len(data)} · {X.shape[1]} predictores · {n_comp} componente(s)",
    )
    out.tiles([("Observaciones", len(data)), ("Componentes", n_comp), ("R²", r2)])
    out.alert("info", "PLS reduce predictores correlacionados a pocas componentes latentes que explican la respuesta: útil con más columnas que filas.")
    # variance explained per component
    x_var = np.var(pls.x_scores_, axis=0) / np.sum(np.var(X, axis=0))
    y_var = np.var(pls.y_scores_, axis=0) / np.var(y)
    out.table(
        "Varianza explicada por componente",
        ["Componente", "Varianza de X explicada", "Varianza de Y explicada", "Acumulada (Y)"],
        [
            [i + 1, float(x_var[i]), float(y_var[i]), float(np.cumsum(y_var)[i])]
            for i in range(n_comp)
        ],
    )
    out.table(
        "Coeficientes de regresión",
        ["Predictor", "Coeficiente"],
        [[predictors[i], float(pls.coef_.ravel()[i])] for i in range(X.shape[1])],
    )
    lo, hi = float(y.min()), float(y.max())
    out.chart(
        scatter_chart(
            "Observado vs ajustado",
            "Observado",
            "Ajustado",
            [{"name": "Pares", "x": y, "y": fitted}],
            lines=[{"name": "Identidad", "x": [lo, hi], "y": [lo, hi], "dash": "dash"}],
        )
    )
    out.dataset("predicciones", "Predicciones y residuos", pd.DataFrame({"observado": y, "ajustado": fitted, "residuo": y - fitted}, index=data.index))
    return out.set(n=len(data), components=n_comp, r2=r2).build()


# ============================================================================ loglinear
def _loglinear(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    count, factors = roles["count"], list(roles["factors"])
    data = frame[[count, *factors]].copy()
    data[count] = pd.to_numeric(data[count], errors="coerce")
    for f in factors:
        data[f] = data[f].astype(str)
    data = data.dropna()
    if len(data) < 4:
        raise InvalidInputError("Se necesitan al menos 4 filas para el modelo loglineal.")
    agg = data.groupby(factors, as_index=False)[count].sum()
    names = {f"f{i}": factors[i] for i in range(len(factors))}
    renamed = agg.rename(columns={count: "count", **{factors[i]: f"f{i}" for i in range(len(factors))}})
    terms = [f"C(f{i})" for i in range(len(factors))]
    if p["interactions"]:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                terms.append(f"C(f{i}):C(f{j})")
    formula = "count ~ " + " + ".join(terms)
    try:
        result = smf.glm(formula, renamed, family=sm.families.Poisson()).fit()
    except Exception as error:  # noqa: BLE001
        raise ComputationError(f"No se pudo ajustar el modelo loglineal: {str(error)[:200]}.") from None

    def pretty(term: str) -> str:
        text = str(term)
        if text == "Intercept":
            return "Constante"
        text = re.sub(r"C\(f(\d+)\)\[T\.([^\]]*)\]", lambda m: f"{names[f'f{m.group(1)}']} = {m.group(2)}", text)
        return text.replace(":", " × ")

    out = Output("Modelo loglineal (tabla de contingencia)", f"{len(agg)} celdas · {len(factors)} factor(es)")
    out.tiles([("Celdas", len(agg)), ("Desvianza", float(result.deviance)), ("AIC", float(result.aic)), ("BIC", float(result.bic))])
    out.table(
        "Parámetros (efectos sobre el logaritmo del conteo)",
        ["Término", "Coeficiente", "Error est.", "z", "Valor p", "Exp(coef.)"],
        [
            [
                pretty(term),
                float(result.params[term]),
                float(result.bse[term]),
                float(result.tvalues[term]),
                P(float(result.pvalues[term])),
                float(np.exp(result.params[term])),
            ]
            for term in result.params.index
        ],
        note="Exp(coef.) es el factor multiplicativo del conteo por cada efecto.",
    )
    out.alert("info", "El modelo loglineal describe la estructura de asociación de la tabla; la desvianza residual mide el ajuste.")
    return out.set(cells=len(agg), deviance=float(result.deviance), aic=float(result.aic), bic=float(result.bic)).build()


# ============================================================================ catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "gam",
        "Modelo aditivo generalizado (GAM)",
        GROUP_ADVANCED,
        "Ajusta la respuesta como suma de funciones suaves (splines) de los predictores, sin imponer linealidad.",
        _gam,
        roles=(RESPONSE, NUMERIC_PREDICTORS),
        params=(FieldSpec("splines", "Grados de libertad por spline", "integer", 5, minimum=3, maximum=15),),
        outputs=(OutputSpec("predicciones", "Predicciones y residuos"),),
        example={"roles": {"response": "y", "predictors": ["x1", "x2"]}, "params": {"splines": 5}},
    ),
    _spec(
        "pls",
        "Mínimos cuadrados parciales (PLS)",
        GROUP_ADVANCED,
        "Reduce predictores correlacionados a componentes latentes que explican la respuesta; útil con muchas columnas.",
        _pls,
        roles=(RESPONSE, NUMERIC_PREDICTORS),
        params=(
            FieldSpec("components", "Número de componentes", "integer", 2, minimum=1, maximum=20),
            FieldSpec("scale", "Estandarizar las variables", "boolean", True),
        ),
        outputs=(OutputSpec("predicciones", "Predicciones y residuos"),),
        example={"roles": {"response": "y", "predictors": ["x1", "x2", "x3"]}, "params": {"components": 2}},
    ),
    _spec(
        "loglinear",
        "Modelo loglineal",
        GROUP_CATEGORICAL,
        "Modela la frecuencia de cada celda de una tabla de contingencia y examina la asociación entre los factores.",
        _loglinear,
        roles=(COUNT, FACTORS),
        params=(FieldSpec("interactions", "Incluir interacciones de dos factores", "boolean", False),),
        example={"roles": {"count": "count", "factors": ["g", "h"]}, "params": {"interactions": False}},
    ),
)

registry.register(*SPECS)
