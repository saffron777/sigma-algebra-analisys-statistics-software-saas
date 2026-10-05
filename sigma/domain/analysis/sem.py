"""Ecuaciones estructurales: análisis de rutas (variables observadas) y modelos con variables latentes (AFC y SEM) con semopy."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from semopy import Model, calc_stats

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MODULE = "sem"
GROUP_PATHS = "Análisis de rutas"
GROUP_LATENT = "Variables latentes (AFC y SEM)"

VARIABLES = RoleSpec("variables", "Variables observadas (numéricas)", "numeric", multiple=True, min_count=2, max_count=30)
MODEL = FieldSpec(
    "model",
    "Modelo de rutas (una ecuación por línea)",
    "text",
    "",
    help="Escriba «dependiente ~ predictor1 + predictor2» por línea, por ejemplo: y ~ x1 + x2.",
)
SEM_MODEL = FieldSpec(
    "model",
    "Modelo (sintaxis lavaan)",
    "text",
    "",
    help="Ejemplo CFA: «F =~ x1 + x2 + x3». Ejemplo SEM: «F =~ x1 + x2 + x3» y «y ~ F» (una instrucción por línea).",
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


# ============================================================================ latent-variable SEM (semopy)
_FIT_INDEXES = ("CFI", "TLI", "RMSEA")


def _fit_sem(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    variables = list(roles["variables"])
    data = frame[variables].copy()
    for v in variables:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if len(data) < 30:
        raise InvalidInputError(f"Se necesitan al menos 30 filas completas (hay {len(data)}).")
    description = str(p["model"]).strip()
    if not description:
        raise InvalidInputError("Escriba el modelo (sintaxis lavaan), por ejemplo «F =~ x1 + x2 + x3».")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(description)
            model.fit(data)
            params = model.inspect(std_est=True)
            stats = calc_stats(model)
    except Exception as error:  # noqa: BLE001
        raise InvalidInputError(f"El modelo no se pudo ajustar: {str(error)[:200]}.") from None

    def _stat(name: str) -> float | None:
        value = stats.get(name)
        if value is None:
            return None
        if isinstance(value, pd.Series):  # calc_stats devuelve un DataFrame de una fila
            value = value.iloc[0]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if not np.isfinite(value) else value

    cfi, tli, rmsea = _stat("CFI"), _stat("TLI"), _stat("RMSEA")
    chi2, dof = _stat("chi2"), _stat("DoF")
    out = Output("Modelo de ecuaciones estructurales", f"n = {len(data)} · {len(variables)} variables observadas")
    out.tiles(
        [
            ("CFI", cfi),
            ("TLI", tli),
            ("RMSEA", rmsea),
            ("χ² (g.l.)", f"{chi2:.2f} ({int(dof)})" if chi2 is not None and dof is not None else None),
        ]
    )
    good = cfi is not None and cfi >= 0.95 and rmsea is not None and rmsea <= 0.06
    out.alert(
        "success" if good else "warning",
        "Ajuste aceptable (CFI ≥ 0.95 y RMSEA ≤ 0.06)." if good else "Ajuste no óptimo: revise el modelo o considere más indicadores.",
    )
    def _num(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if not np.isfinite(number) else number

    rows = []
    for _, r in params.iterrows():
        term = f"{r['lval']} {r['op']} {r['rval']}".replace("~~", "↔").replace(" ~ ", " ← ")
        pvalue = _num(r["p-value"])
        rows.append([term, _num(r["Estimate"]), _num(r["Est. Std"]), _num(r["Std. Err"]), _num(r["z-value"]), P(pvalue) if pvalue is not None else None])

    out.table(
        "Estimaciones (estandarizadas)",
        ["Término", "Estimación", "Estandarizada", "Error est.", "z", "Valor p"],
        rows,
        note="«=~» define la medición (cargas), «~» las relaciones estructurales y «~~» varianzas/covarianzas. Est. Std: efecto en desviaciones estándar.",
    )
    out.set(n=len(data), cfi=cfi, tli=tli, rmsea=rmsea, chi2=chi2, dof=int(dof) if dof is not None else None)
    return out.build()


def _sem_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    from sigma.domain.analysis.support import load_line

    return (
        "from semopy import Model, calc_stats\n\n"
        + load_line(dataset)
        + f"m = Model({p['model']!r})\nm.fit(df)\nprint(m.inspect(std_est=True))\nprint(calc_stats(m))\n"
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
    _spec(
        "cfa",
        "Análisis factorial confirmatorio (AFC)",
        GROUP_LATENT,
        "Estima un modelo de medida con factores latentes (sintaxis lavaan «F =~ x1 + x2 + x3») y reporta cargas e índices de ajuste.",
        _fit_sem,
        roles=(VARIABLES,),
        params=(SEM_MODEL,),
        code=_sem_code,
        example={"roles": {"variables": ["x1", "x2", "x3"]}, "params": {"model": "F =~ x1 + x2 + x3"}},
    ),
    _spec(
        "sem",
        "Modelo de ecuaciones estructurales (SEM)",
        GROUP_LATENT,
        "Estima un SEM completo con variables latentes (medida «=~» y estructural «~») y reporta coeficientes e índices de ajuste.",
        _fit_sem,
        roles=(VARIABLES,),
        params=(SEM_MODEL,),
        code=_sem_code,
        example={"roles": {"variables": ["x1", "x2", "x3", "y"]}, "params": {"model": "F =~ x1 + x2 + x3\ny ~ F"}},
    ),
)

registry.register(*SPECS)
