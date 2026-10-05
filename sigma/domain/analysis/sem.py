"""Ecuaciones estructurales: análisis de rutas (path analysis) sobre variables observadas, sin librerías externas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MODULE = "sem"
GROUP_PATHS = "Análisis de rutas"

VARIABLES = RoleSpec("variables", "Variables observadas (numéricas)", "numeric", multiple=True, min_count=2, max_count=30)
MODEL = FieldSpec(
    "model",
    "Modelo de rutas (una ecuación por línea)",
    "text",
    "",
    help="Escriba «dependiente ~ predictor1 + predictor2» por línea, por ejemplo: y ~ x1 + x2.",
)


def _parse_model(model: str, variables: list[str]) -> list[tuple[str, list[str]]]:
    equations: list[tuple[str, list[str]]] = []
    for line in model.splitlines():
        line = line.strip()
        if not line:
            continue
        if "~" not in line:
            raise InvalidInputError(f"Línea sin «~»: «{line}». Use «dependiente ~ predictor1 + predictor2».")
        dep, rhs = line.split("~", 1)
        dep = dep.strip()
        preds = [p.strip() for p in rhs.split("+") if p.strip()]
        if dep not in variables:
            raise InvalidInputError(f"«{dep}» no está entre las variables elegidas.")
        for pred in preds:
            if pred not in variables:
                raise InvalidInputError(f"«{pred}» no está entre las variables elegidas.")
        if not preds:
            raise InvalidInputError(f"La ecuación «{line}» no tiene predictores.")
        equations.append((dep, preds))
    if not equations:
        raise InvalidInputError("Indique al menos una ecuación en el formato «y ~ x1 + x2».")
    return equations


def _path(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    variables = list(roles["variables"])
    data = frame[variables].copy()
    for v in variables:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if len(data) < 10:
        raise InvalidInputError(f"Se necesitan al menos 10 filas completas (hay {len(data)}).")
    equations = _parse_model(p["model"], variables)
    std = (data - data.mean()) / data.std(ddof=1)

    out = Output("Análisis de rutas", f"n = {len(data)} · {len(equations)} ecuación(es)")
    out.tiles([("Observaciones", len(data)), ("Ecuaciones", len(equations)), ("Variables", len(variables))])
    out.alert(
        "info",
        "Cada ecuación es una regresión por mínimos cuadrados; el coeficiente estandarizado es el coeficiente de ruta (β): el efecto en desviaciones estándar.",
    )
    for dep, preds in equations:
        y = data[dep].to_numpy(dtype=float)
        X = sm.add_constant(data[preds].to_numpy(dtype=float))
        fit = sm.OLS(y, X).fit()
        ys = std[dep].to_numpy(dtype=float)
        Xs = sm.add_constant(std[preds].to_numpy(dtype=float))
        fits = sm.OLS(ys, Xs).fit()
        names = ["Constante", *preds]
        out.heading(f"{dep} ~ {' + '.join(preds)} (R² = {fit.rsquared:.3f})")
        out.table(
            "Coeficientes",
            ["Término", "Coeficiente", "Estandarizado (ruta)", "Error est.", "t", "Valor p"],
            [
                [names[i], float(fit.params[i]), float(fits.params[i]), float(fit.bse[i]), float(fit.tvalues[i]), P(float(fit.pvalues[i]))]
                for i in range(len(names))
            ],
        )
    out.set(n=len(data), equations=len(equations), variables=len(variables))
    return out.build()


def _path_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    from sigma.domain.analysis.support import load_line

    return (
        "import statsmodels.api as sm\n\n"
        + load_line(dataset)
        + f"model = {p['model']!r}\nfor line in model.splitlines():\n"
        + "    dep, rhs = line.split('~')\n    preds = [x.strip() for x in rhs.split('+')]\n"
        + "    X = sm.add_constant(df[preds].to_numpy(float))\n    fit = sm.OLS(df[dep.strip()].to_numpy(float), X).fit()\n"
        + "    print(dep.strip(), fit.params, fit.rsquared)\n"
    )


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "path_analysis",
        "Análisis de rutas (path analysis)",
        GROUP_PATHS,
        "Estima un sistema de ecuaciones estructurales sobre variables observadas y reporta los coeficientes de ruta estandarizados.",
        _path,
        roles=(VARIABLES,),
        params=(MODEL,),
        code=_path_code,
        example={"roles": {"variables": ["y", "x1", "x2"]}, "params": {"model": "y ~ x1 + x2"}},
    ),
)

registry.register(*SPECS)
