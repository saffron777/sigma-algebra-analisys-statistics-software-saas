"""Metadata describing a probability distribution and how to instantiate it."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sigma.domain.errors import InvalidInputError

CONTINUOUS = "continuous"
DISCRETE = "discrete"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    default: float
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    exclusive_min: bool = False


@dataclass(frozen=True)
class DistributionSpec:
    key: str
    label: str
    kind: str
    params: tuple[ParamSpec, ...]
    builder: Callable[..., Any]
    description: str = ""

    def build(self, values: dict[str, float]) -> Any:
        """Validate the parameters and return a frozen ``scipy.stats`` distribution."""
        clean: dict[str, float] = {}
        for spec in self.params:
            if spec.name not in values or values[spec.name] is None:
                raise InvalidInputError(f"Falta el parámetro «{spec.label}».")
            value = float(values[spec.name])
            if not math.isfinite(value):
                raise InvalidInputError(f"El parámetro «{spec.label}» debe ser finito.")
            if spec.integer and value != int(value):
                raise InvalidInputError(f"El parámetro «{spec.label}» debe ser entero.")
            if spec.minimum is not None and (value < spec.minimum or (spec.exclusive_min and value == spec.minimum)):
                relation = ">" if spec.exclusive_min else "≥"
                raise InvalidInputError(f"«{spec.label}» debe ser {relation} {spec.minimum:g}.")
            if spec.maximum is not None and value > spec.maximum:
                raise InvalidInputError(f"«{spec.label}» debe ser ≤ {spec.maximum:g}.")
            clean[spec.name] = int(value) if spec.integer else value
        return self.builder(**clean)

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


@dataclass(frozen=True)
class DistributionRef:
    """Reference to a distribution + parameters; results carry it so charts can draw it."""

    key: str
    params: dict[str, float] = field(default_factory=dict)
