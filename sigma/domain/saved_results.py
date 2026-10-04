"""Entity for results the user decided to keep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sigma.domain.analysis.modules import CATALOG_MODULES

MODULES = (
    "descriptive",
    "distributions",
    "hypothesis",
    "regression",
    "anova",
    "markets",
    *(m.key for m in CATALOG_MODULES),
)

MODULE_LABELS = {
    "descriptive": "Estadística descriptiva",
    "distributions": "Distribuciones",
    "hypothesis": "Pruebas de hipótesis",
    "regression": "Regresión lineal",
    "anova": "ANOVA",
    "markets": "Análisis de mercados",
    **{m.key: m.label for m in CATALOG_MODULES},
}


@dataclass
class SavedResultSummary:
    id: str
    module: str
    analysis_type: str
    title: str
    notes: str
    dataset_id: str | None
    dataset_name: str | None
    created_at: str
    updated_at: str


@dataclass
class SavedResult(SavedResultSummary):
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    interpretation: dict[str, Any] | None = None  # AI text kept with the result (see domain/interpretation.py)
