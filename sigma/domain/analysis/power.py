"""Power and sample size (Minitab *Power and Sample Size*, SPSS *Power Analysis*): solve for n, power or effect."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower, TTestPower

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart
from sigma.domain.analysis.output import Output, strong
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec
from sigma.domain.errors import ComputationError, InvalidInputError

_ALTERNATIVES = (("two-sided", "Bilateral (≠)"), ("larger", "Cola derecha (>)"), ("smaller", "Cola izquierda (<)"))
_DESIGNS = (
    ("two_sample", "Dos muestras independientes"),
    ("one_sample", "Una muestra"),
    ("paired", "Muestras pareadas"),
)
_SOLVE = (("n", "Tamaño de muestra"), ("power", "Potencia"), ("effect", "Efecto detectable (d de Cohen)"))
_DESIGN_NAMES = dict(_DESIGNS)


def _solver(design: str):
    return TTestIndPower() if design == "two_sample" else TTestPower()


def _call(
    design: str,
    solver: Any,
    effect: float | None,
    n: float | None,
    alpha: float,
    power: float | None,
    ratio: float,
    alternative: str,
):
    """One call to statsmodels' ``solve_power``; the unknown is the argument left as ``None``."""
    if design == "two_sample":
        return solver.solve_power(
            effect_size=effect, nobs1=n, alpha=alpha, power=power, ratio=ratio, alternative=alternative
        )
    return solver.solve_power(effect_size=effect, nobs=n, alpha=alpha, power=power, alternative=alternative)


def power_at(design: str, effect: float, n: float, alpha: float, ratio: float, alternative: str) -> float:
    return float(_call(design, _solver(design), effect, n, alpha, None, ratio, alternative))


def _t_power(_: pd.DataFrame, __: dict[str, Any], p: dict[str, Any]):
    design, solve, alpha, alternative = p["design"], p["solve_for"], p["alpha"], p["alternative"]
    ratio = p["ratio"] if design == "two_sample" else 1.0
    effect, n, power = p.get("effect_size"), p.get("n"), p.get("power")
    solver = _solver(design)
    try:
        if solve == "n":
            n = float(_call(design, solver, effect, None, alpha, power, ratio, alternative))
        elif solve == "power":
            power = float(_call(design, solver, effect, n, alpha, None, ratio, alternative))
        else:
            effect = float(_call(design, solver, None, n, alpha, power, ratio, alternative))
    except (ValueError, RuntimeError, ZeroDivisionError) as error:
        raise ComputationError(f"No se pudo despejar el valor pedido con esos datos: {error}") from None
    if not all(v is not None and math.isfinite(v) for v in (n, power, effect)):
        raise ComputationError("No se pudo despejar el valor pedido: revise el efecto, α y la potencia.")

    n_up = math.ceil(n) if solve == "n" else n
    actual = power_at(design, effect, n_up, alpha, ratio, alternative) if solve == "n" else power
    per_group = f"{n_up:g}" if design != "two_sample" else f"{n_up:g} + {math.ceil(n_up * ratio):g}"
    out = Output(
        f"Potencia y tamaño de muestra: prueba t ({_DESIGN_NAMES[design].lower()})",
        f"α = {alpha:g} · {dict(_ALTERNATIVES)[alternative]}",
    )
    out.tiles(
        [
            (
                "Tamaño de muestra" + (" por grupo" if design == "two_sample" else ""),
                f"{n_up:g}" if solve == "n" else n,
            ),
            ("Potencia", actual),
            ("Efecto (d)", effect),
            ("α", alpha),
        ]
    )
    if solve == "n":
        out.alert(
            "info",
            f"Se necesitan {per_group} observaciones para detectar d = {effect:g} "
            f"con potencia ≥ {power:g} (potencia real: {actual:.4f}).",
        )
    out.table(
        "Parámetros",
        ["Parámetro", "Valor"],
        [
            ["Diseño", _DESIGN_NAMES[design]],
            ["Efecto estandarizado d", strong(effect) if solve == "effect" else effect],
            [
                "Tamaño de muestra" + (" (grupo 1)" if design == "two_sample" else ""),
                strong(n_up) if solve == "n" else n,
            ],
            ["Potencia", strong(actual) if solve != "n" else actual],
            ["α", alpha],
            *([["Razón n₂ / n₁", ratio]] if design == "two_sample" else []),
        ],
    )
    # power curves around the solution
    top = max(10, int(math.ceil(n_up * 2.2)))
    grid = np.unique(np.linspace(max(2.2, 3), top, 60))
    curves = []
    for factor, dash in ((0.5, "dot"), (1.0, "solid"), (1.5, "dash")):
        d = effect * factor
        curves.append(
            {
                "name": f"d = {d:.3g}",
                "x": grid,
                "y": [power_at(design, d, float(m), alpha, ratio, alternative) for m in grid],
                "dash": dash,
            }
        )
    out.chart(
        line_chart(
            "Curvas de potencia",
            "Tamaño de muestra" + (" por grupo" if design == "two_sample" else ""),
            "Potencia",
            curves,
            hlines=[{"value": 0.8, "label": "0.80"}],
            vlines=[{"value": n_up, "label": f"n = {n_up:g}"}],
        )
    )
    out.set(design=design, solve_for=solve, n=n_up, power=actual, effect_size=effect, alpha=alpha)
    return out.build()


def _t_power_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cls = "TTestIndPower" if p["design"] == "two_sample" else "TTestPower"
    n_arg = "nobs1" if p["design"] == "two_sample" else "nobs"
    extra = f", ratio={p['ratio']!r}" if p["design"] == "two_sample" else ""
    solved = {"n": n_arg, "power": "power", "effect": "effect_size"}[p["solve_for"]]
    args = {"effect_size": p.get("effect_size"), n_arg: p.get("n"), "power": p.get("power")}
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != solved and v is not None)
    return (
        f"from statsmodels.stats.power import {cls}\n\n"
        f"resultado = {cls}().solve_power({shown}, alpha={p['alpha']!r}{extra}, alternative={p['alternative']!r})\n"
        f"print(resultado)  # {solved} despejado\n"
    )


def _validate_design(p: dict[str, Any]) -> None:
    if p["solve_for"] != "effect" and p.get("effect_size") in (None, 0):
        raise InvalidInputError("El efecto d debe ser distinto de 0.")


def _compute_t_power(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    _validate_design(params)
    return _t_power(frame, roles, params)


SPECS = (
    AnalysisSpec(
        "t_power",
        "Prueba t: potencia y tamaño de muestra",
        "Pruebas t",
        "potencia",
        "Despeje el tamaño de muestra, la potencia o el efecto detectable de una prueba t y vea las curvas.",
        compute=_compute_t_power,
        needs_dataset=False,
        params=(
            FieldSpec("design", "Diseño", "select", "two_sample", _DESIGNS),
            FieldSpec("solve_for", "Calcular", "select", "n", _SOLVE),
            FieldSpec(
                "effect_size",
                "Efecto estandarizado d",
                "number",
                0.5,
                help="0.2 pequeño · 0.5 mediano · 0.8 grande.",
                visible_if=("solve_for", ("n", "power")),
            ),
            FieldSpec(
                "n",
                "Tamaño de muestra (por grupo)",
                "number",
                30,
                minimum=2,
                visible_if=("solve_for", ("power", "effect")),
            ),
            FieldSpec(
                "power",
                "Potencia deseada",
                "number",
                0.8,
                minimum=0.01,
                maximum=0.999,
                visible_if=("solve_for", ("n", "effect")),
            ),
            FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.0001, maximum=0.5),
            FieldSpec("alternative", "Hipótesis alternativa", "select", "two-sided", _ALTERNATIVES),
            FieldSpec(
                "ratio", "Razón n₂ / n₁", "number", 1.0, minimum=0.1, maximum=20, visible_if=("design", ("two_sample",))
            ),
        ),
        code=_t_power_code,
        example={"roles": {}, "params": {"design": "two_sample", "solve_for": "n", "effect_size": 0.5, "power": 0.8}},
    ),
)

registry.register(*SPECS)
