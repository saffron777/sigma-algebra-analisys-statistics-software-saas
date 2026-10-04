"""Every declared analysis, by module. Area files register their specs when imported (``load_all``)."""

from __future__ import annotations

import importlib
import os

from sigma.domain.analysis.modules import BY_KEY
from sigma.domain.analysis.spec import AnalysisSpec
from sigma.domain.errors import InvalidInputError

# the files that declare analyses (one per area); importing each one registers its specs
AREA_MODULES = (
    "sigma.domain.analysis.power",
    "sigma.domain.analysis.power_designs",
    "sigma.domain.analysis.series",
    "sigma.domain.analysis.series_explore",
    "sigma.domain.analysis.series_models",
    "sigma.domain.analysis.transform",
    "sigma.domain.analysis.datamanagement",
    "sigma.domain.analysis.cleaning",
    "sigma.domain.analysis.plots",
    "sigma.domain.analysis.pruebas",
    "sigma.domain.analysis.remuestreo",
    "sigma.domain.analysis.models",
    "sigma.domain.analysis.models_more",
    "sigma.domain.analysis.econ_multi",
    "sigma.domain.analysis.econ_models",
    "sigma.domain.analysis.multivariate",
    "sigma.domain.analysis.multivariate_groups",
    "sigma.domain.analysis.anova_advanced",
)

_SPECS: dict[str, dict[str, AnalysisSpec]] = {}
_loaded = False


def register(*specs: AnalysisSpec) -> None:
    for spec in specs:
        if spec.module not in BY_KEY:
            raise ValueError(f"analysis {spec.id!r}: unknown module {spec.module!r}")
        bucket = _SPECS.setdefault(spec.module, {})
        if spec.id in bucket:
            raise ValueError(f"analysis {spec.id!r} is declared twice in module {spec.module!r}")
        bucket[spec.id] = spec


def load_all() -> None:
    """Import every area file (and, for plug-ins and tests, the ones named in ``SIGMA_EXTRA_ANALYSIS_MODULES``)."""
    global _loaded
    if _loaded:
        return
    extra = [name.strip() for name in os.environ.get("SIGMA_EXTRA_ANALYSIS_MODULES", "").split(",") if name.strip()]
    for name in (*AREA_MODULES, *extra):
        importlib.import_module(name)
    _loaded = True


def catalog(module: str) -> list[AnalysisSpec]:
    load_all()
    if module not in BY_KEY:
        raise InvalidInputError(f"Módulo de análisis desconocido «{module}».")
    return [spec for spec in _SPECS.get(module, {}).values() if not spec.id.startswith("_")]  # "_x": internal


def get(module: str, analysis_id: str) -> AnalysisSpec:
    load_all()
    try:
        return _SPECS[module][analysis_id]
    except KeyError:
        raise InvalidInputError(f"El análisis «{analysis_id}» no existe en este módulo.") from None


def all_specs() -> list[AnalysisSpec]:
    """Every public analysis (ids starting with an underscore are internal and used by the tests)."""
    load_all()
    return [spec for bucket in _SPECS.values() for spec in bucket.values() if not spec.id.startswith("_")]
