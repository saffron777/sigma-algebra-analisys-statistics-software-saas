"""Diseño de experimentos (DOE): factoriales 2^k y superficie de respuesta, generando el diseño y analizando la respuesta."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import interval_chart
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MODULE = "doe"
GROUP_FACTORIAL = "Diseños factoriales"
GROUP_RSM = "Superficie de respuesta"

RESPONSE = RoleSpec("response", "Respuesta (opcional)", "numeric", optional=True, help="Si se elige, se analiza contra el diseño generado (debe tener una fila por corrida).")
FACTORS = FieldSpec("factors", "Factores (2 a 7)", "integer", 3, minimum=2, maximum=7)
CENTERS = FieldSpec("center_points", "Réplicas en el centro", "integer", 0, minimum=0, maximum=10)
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)


def _factor_names(k: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(k)]


def _full_factorial(k: int, centers: int) -> pd.DataFrame:
    """2^k full factorial in coded -1/+1, plus ``centers`` rows at 0."""
    matrix = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)
    if centers:
        matrix = np.vstack([matrix, np.zeros((centers, k))])
    return pd.DataFrame(matrix, columns=_factor_names(k))


def _fit_and_report(out: Output, design: pd.DataFrame, response: np.ndarray, alpha: float, quadratic: bool):
    names = list(design.columns)
    data = design.copy()
    data["y"] = response
    terms = names + [f"{a}:{b}" for a, b in itertools.combinations(names, 2)]
    if quadratic:
        terms += [f"I({n} ** 2)" for n in names]
    formula = "y ~ " + " + ".join(terms)
    result = smf.ols(formula, data).fit()
    out.tiles([("Corridas", len(design)), ("R²", float(result.rsquared)), ("R² ajustado", float(result.rsquared_adj))])
    out.table(
        "Efectos y coeficientes",
        ["Término", "Coeficiente", "Error est.", "t", "Valor p"],
        [
            [term, float(result.params[term]), float(result.bse[term]), float(result.tvalues[term]), P(float(result.pvalues[term]))]
            for term in result.params.index
        ],
        note="Factores codificados (−1 / +1): el coeficiente es la mitad del efecto. Los términos con p < α son significativos.",
    )
    significant = [term for term in names if float(result.pvalues[term]) < alpha]
    out.alert(
        "info" if significant else "warning",
        "Factores significativos: " + (", ".join(significant) if significant else "ninguno") + ".",
    )
    # effects plot (mean of the response at -1 vs +1 per factor)
    groups = []
    for n in names:
        low = float(response[design[n] == -1].mean()) if (design[n] == -1).any() else None
        high = float(response[design[n] == 1].mean()) if (design[n] == 1).any() else None
        if low is not None and high is not None:
            groups.append({"name": n, "mean": (low + high) / 2, "lower": low, "upper": high})
    if groups:
        out.chart(interval_chart("Media de la respuesta por nivel del factor", "Factor", "Respuesta", groups))
    out.set(r2=float(result.rsquared), r2_adj=float(result.rsquared_adj), coefficients=[float(v) for v in result.params])


def _factorial(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    design = _full_factorial(int(p["factors"]), int(p["center_points"]))
    out = Output("Diseño factorial 2^k", f"{len(design)} corridas · {int(p['factors'])} factores a dos niveles")
    out.tiles([("Factores", int(p["factors"])), ("Corridas", len(design)), ("Réplicas en el centro", int(p["center_points"]))])
    out.alert("info", "Diseño codificado en −1 (nivel bajo) y +1 (nivel alto); guárdelo como dataset y añada la columna de respuesta al ejecutar los experimentos.")
    if roles.get("response"):
        values = pd.to_numeric(frame[roles["response"]], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) != len(design):
            raise InvalidInputError(f"El diseño tiene {len(design)} corridas pero la respuesta tiene {len(values)} valores: deben coincidir.")
        _fit_and_report(out, design, values, p["alpha"], quadratic=False)
    out.dataset("diseño", "Diseño factorial", design)
    return out.set(factors=int(p["factors"]), runs=len(design)).build()


def _response_surface(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    k = int(p["factors"])
    centers = int(p["center_points"])
    base = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)
    alpha = float(p["axial"]) if p.get("axial") else float(2 ** (k / 4))
    axial = []
    for i in range(k):
        for sign in (-1.0, 1.0):
            point = np.zeros(k)
            point[i] = sign * alpha
            axial.append(point)
    matrix = np.vstack([base, np.array(axial), np.zeros((max(centers, 1), k))])
    design = pd.DataFrame(matrix, columns=_factor_names(k))
    out = Output(
        "Diseño de superficie de respuesta (compuesto central)",
        f"{len(design)} corridas · {k} factores · axial a ±{alpha:g}",
    )
    out.tiles([("Factores", k), ("Corridas", len(design)), ("Distancia axial α", alpha)])
    out.alert("info", "Diseño compuesto central (factorial + puntos axiales + centro) para ajustar un modelo cuadrático.")
    if roles.get("response"):
        values = pd.to_numeric(frame[roles["response"]], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) != len(design):
            raise InvalidInputError(f"El diseño tiene {len(design)} corridas pero la respuesta tiene {len(values)} valores: deben coincidir.")
        _fit_and_report(out, design, values, p["alpha"], quadratic=True)
    out.dataset("diseño", "Diseño compuesto central", design)
    return out.set(factors=k, runs=len(design), axial=alpha).build()


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "factorial",
        "Factorial completo 2^k",
        GROUP_FACTORIAL,
        "Genera un diseño factorial 2^k (con réplicas centrales opcionales) y, si hay respuesta, ajusta los efectos principales e interacciones.",
        _factorial,
        roles=(RESPONSE,),
        params=(FACTORS, CENTERS, ALPHA),
        outputs=(OutputSpec("diseño", "Diseño factorial"),),
        example={"roles": {}, "params": {"factors": 3, "center_points": 0}},
    ),
    _spec(
        "response_surface",
        "Superficie de respuesta (compuesto central)",
        GROUP_RSM,
        "Genera un diseño compuesto central y, si hay respuesta, ajusta el modelo cuadrático con interacciones.",
        _response_surface,
        roles=(RESPONSE,),
        params=(FACTORS, FieldSpec("axial", "Distancia axial α (vacío = rotable 2^(k/4))", "number", optional=True, minimum=0.5, maximum=3.0), CENTERS, ALPHA),
        outputs=(OutputSpec("diseño", "Diseño compuesto central"),),
        example={"roles": {}, "params": {"factors": 2, "center_points": 2}},
    ),
)

registry.register(*SPECS)
