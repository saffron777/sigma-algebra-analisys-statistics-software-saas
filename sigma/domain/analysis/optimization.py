"""Optimización (SAS/OR): programación lineal y entera para asignación de recursos con restricción de presupuesto."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MODULE = "optimizacion"
GROUP_LP = "Programación lineal"

VALUE = RoleSpec("value", "Coeficiente del objetivo (por unidad)", "numeric", help="Lo que aporta cada elemento por unidad (beneficio, rendimiento…).")
COST = RoleSpec("cost", "Uso de recurso por unidad (opcional)", "numeric", optional=True, help="Cuánto consume cada unidad del recurso limitado; por defecto 1.")
BOUND = RoleSpec("bound", "Límite superior por elemento (opcional)", "numeric", optional=True, help="Cantidad máxima de cada elemento; por defecto 1.")


def _optimize(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    n = len(frame)
    if n < 1:
        raise InvalidInputError("Se necesita al menos una fila (un elemento).")
    value = pd.to_numeric(frame[roles["value"]], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(value)):
        raise InvalidInputError("El coeficiente del objetivo tiene valores no numéricos o vacíos.")
    cost = pd.to_numeric(frame[roles["cost"]], errors="coerce").to_numpy(dtype=float) if roles.get("cost") else np.ones(n)
    if np.any(~np.isfinite(cost)) or np.any(cost < 0):
        raise InvalidInputError("El uso de recurso debe ser un número no negativo.")
    bound = pd.to_numeric(frame[roles["bound"]], errors="coerce").to_numpy(dtype=float) if roles.get("bound") else np.ones(n)
    if np.any(~np.isfinite(bound)) or np.any(bound < 0):
        raise InvalidInputError("El límite superior debe ser un número no negativo.")
    budget = float(p["budget"])
    maximize = p["sense"] == "maximize"
    integer = bool(p["integer"])

    c = -value if maximize else value  # linprog minimiza; para maximizar se niega
    bounds = [(0.0, float(b)) for b in bound]
    if integer:
        result = milp(
            c=c,
            integrality=np.ones(n),
            bounds=Bounds(lb=np.zeros(n), ub=bound),
            constraints=LinearConstraint(cost.reshape(1, -1), -np.inf, budget),
        )
        if not result.success:
            raise InvalidInputError(f"No se pudo resolver el problema: {result.message}.")
        x = result.x
    else:
        result = linprog(c, A_ub=cost.reshape(1, -1), b_ub=[budget], bounds=bounds, method="highs")
        if not result.success:
            raise InvalidInputError(f"No se pudo resolver el problema: {result.message}.")
        x = result.x

    objective = float(np.sum(value * x))
    used = float(np.sum(cost * x))
    out = Output(
        "Optimización de recursos" + (" (entera)" if integer else ""),
        f"{n} elementos · {'maximizar' if maximize else 'minimizar'} · presupuesto {budget:g}",
    )
    out.tiles(
        [
            ("Valor objetivo", objective),
            ("Recurso usado", used),
            ("Elementos con x > 0", int((x > 1e-9).sum())),
        ]
    )
    out.alert(
        "success" if np.isfinite(objective) else "warning",
        f"Solución {'entera' if integer else 'continua'} óptima: objetivo = {objective:.4g}, usando {used:.4g} de {budget:g} del recurso.",
    )
    out.table(
        "Solución",
        ["Elemento", "Valor por unidad", "Recurso por unidad", "Cantidad óptima (x)"],
        [[i + 1, float(value[i]), float(cost[i]), float(x[i])] for i in range(n)],
        note="x es la cantidad a asignar a cada elemento; se muestra la solución óptima.",
    )
    out.dataset("solucion", "Asignación óptima", pd.DataFrame({"elemento": np.arange(1, n + 1), "x": x}))
    return out.set(objective=objective, resource_used=used, n_selected=int((x > 1e-9).sum())).build()


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "lp",
        "Programación lineal / entera (asignación con presupuesto)",
        GROUP_LP,
        "Optimiza la asignación de recursos: maximiza o minimiza el valor sujeto a un presupuesto y límites por elemento.",
        _optimize,
        roles=(VALUE, COST, BOUND),
        params=(
            FieldSpec("sense", "Sentido", "select", "maximize", (("maximize", "Maximizar"), ("minimize", "Minimizar"))),
            FieldSpec("budget", "Presupuesto (recurso disponible)", "number", 10.0, minimum=0.0),
            FieldSpec("integer", "Variables enteras (0/1 con límite 1)", "boolean", False),
        ),
        outputs=(OutputSpec("solucion", "Asignación óptima"),),
        example={"roles": {"value": "x1", "cost": "x2", "bound": "x3"}, "params": {"sense": "maximize", "budget": 20, "integer": False}},
    ),
)

registry.register(*SPECS)
