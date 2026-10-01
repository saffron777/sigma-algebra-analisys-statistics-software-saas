"""The per-module workspace: what a user was doing in a module, kept until they press «Limpiar»."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sigma.domain.errors import InvalidInputError

WORKSPACE_MODULES = (
    "datos",
    "descriptiva",
    "distribuciones",
    "hipotesis",
    "regresion",
    "anova",
    "mercados",
    "resultados",
)
MAX_WORKSPACE_BYTES = 16 * 1024 * 1024
IMPORT_MODULE = "datos"  # its workspace also holds the pending imports (uploaded file + preview)


class _Unset:
    """Marks a field the client did not send (a partial update keeps it as it is)."""

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _Unset()


@dataclass
class WorkspaceEntry:
    """``state`` is the form (small, saved on every edit); ``payload`` is the last result (large, saved per run)."""

    module: str
    state: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    revision: int = 0
    updated_at: str | None = None


def validate_module(module: str) -> str:
    if module not in WORKSPACE_MODULES:
        raise InvalidInputError(f"Módulo de trabajo inválido «{module}».")
    return module
