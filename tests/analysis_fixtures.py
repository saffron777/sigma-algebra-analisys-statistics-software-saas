"""Analyses that exist only for the tests of the framework (loaded with SIGMA_EXTRA_ANALYSIS_MODULES)."""

from __future__ import annotations

import time
import warnings
from typing import Any

import pandas as pd

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.errors import InvalidInputError


def _sleep(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    time.sleep(params["seconds"])
    return Output("Dormido").build()


def _boom(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    raise KeyError("columna_fantasma")


def _invalid(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    raise InvalidInputError("Parámetros imposibles.")


def _warns(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    warnings.warn("el optimizador no convergió", UserWarning, stacklevel=1)
    warnings.warn("divide by zero encountered in log", RuntimeWarning, stacklevel=1)
    return Output("Con advertencias").tiles([("n", len(frame))]).build()


def _copy(frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]):
    out = Output("Copia de columnas", f"{len(frame)} filas")
    out.dataset(
        "resultado", "Datos resultantes", frame[[roles["col"]]].assign(doble=frame[roles["col"]] * params["factor"])
    )
    out.table("Vista", ["n"], [[len(frame)]])
    return out.set(n=len(frame)).build()


_COMMON = {"heavy": True, "needs_dataset": False}
SPECS = (
    AnalysisSpec(
        "_sleep",
        "Dormir",
        "Pruebas",
        "series",
        "solo para pruebas",
        _sleep,
        params=(FieldSpec("seconds", "Segundos", "number", 0.1),),
        **_COMMON,
    ),
    AnalysisSpec("_boom", "Fallar", "Pruebas", "series", "solo para pruebas", _boom, **_COMMON),
    AnalysisSpec("_invalid", "Inválido", "Pruebas", "series", "solo para pruebas", _invalid, **_COMMON),
    AnalysisSpec("_warns", "Advertir", "Pruebas", "series", "solo para pruebas", _warns, **_COMMON),
    AnalysisSpec(
        "_copy",
        "Copiar",
        "Pruebas",
        "transformar",
        "solo para pruebas",
        _copy,
        roles=(RoleSpec("col", "Columna", "numeric"),),
        params=(FieldSpec("factor", "Factor", "number", 2.0),),
        outputs=(OutputSpec("resultado", "Datos resultantes", required=True),),
        can_replace=True,
    ),
)

registry.register(*SPECS)
