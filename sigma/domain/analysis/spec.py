"""Declaration of an analysis and validation of what the user sends for it (columns and parameters)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sigma.domain.analysis.output import AnalysisOutput
from sigma.domain.datasets import classify_column
from sigma.domain.errors import InvalidInputError

# which dataset column kinds a role accepts (``None`` = any)
ROLE_KINDS: dict[str, frozenset[str] | None] = {
    "numeric": frozenset({"numeric", "boolean"}),
    "categorical": frozenset({"categorical", "boolean", "text"}),
    "datetime": frozenset({"datetime"}),
    "any": None,
}
_KIND_NAMES = {"numeric": "numérica", "categorical": "categórica", "datetime": "de fecha"}
FIELD_KINDS = ("number", "integer", "select", "text", "boolean", "multiselect", "code", "dataset")
MAX_TEXT = 500
MAX_CODE = 4000


@dataclass(frozen=True)
class RoleSpec:
    """A column (or several) the analysis asks for."""

    name: str
    label: str
    kind: str = "numeric"  # numeric | categorical | datetime | any
    multiple: bool = False
    optional: bool = False
    min_count: int = 1  # for ``multiple`` roles
    max_count: int | None = None
    help: str = ""
    dataset_param: str | None = (
        None  # the columns come from the dataset chosen in that "dataset" parameter, not the main one
    )


@dataclass(frozen=True)
class FieldSpec:
    """A parameter of the analysis. ``visible_if=(other_param, (values…))`` hides it unless the other one matches.

    The ``dataset`` kind holds the id of another dataset of the account (merge, append…); the use case loads it and
    hands the frame to ``compute`` in that parameter's place.
    """

    name: str
    label: str
    kind: str = "number"
    default: Any = None
    options: tuple[tuple[str, str], ...] = ()
    help: str = ""
    optional: bool = False
    minimum: float | None = None
    maximum: float | None = None
    visible_if: tuple[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class OutputSpec:
    """A table the analysis can leave behind as a new dataset (scores, predictions, residuals, the cleaned data…)."""

    key: str
    label: str
    required: bool = False  # the page insists on a name (data transformations)


Compute = Callable[[pd.DataFrame, dict[str, Any], dict[str, Any]], AnalysisOutput]
CodeBuilder = Callable[[dict[str, Any], dict[str, Any], str], str]


@dataclass(frozen=True)
class AnalysisSpec:
    id: str
    label: str
    group: str
    module: str
    description: str
    compute: Compute
    roles: tuple[RoleSpec, ...] = ()
    params: tuple[FieldSpec, ...] = ()
    outputs: tuple[OutputSpec, ...] = ()
    needs_dataset: bool = True
    heavy: bool = False  # runs in a separate process with a time limit
    can_replace: bool = False  # may overwrite the dataset it was run on (data transformations)
    code: CodeBuilder | None = None  # Python that reproduces the analysis: (roles, params, dataset name) -> text
    example: dict[str, Any] | None = None  # a valid call on the test dataset: {"roles": {...}, "params": {...}}
    reference: tuple[
        tuple[str, str], ...
    ] = ()  # (term, explanation) pairs the page lists, e.g. the functions of a formula


# ============================================================================ validation
def dataset_params(spec: AnalysisSpec) -> list[str]:
    """Names of the parameters that point to another dataset."""
    return [field.name for field in spec.params if field.kind == "dataset"]


def prepare_inputs(
    spec: AnalysisSpec,
    frame: pd.DataFrame | None,
    roles: dict[str, Any],
    params: dict[str, Any],
    frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the chosen columns and parameters against the declaration; returns them cleaned and with defaults.

    ``frames`` holds the data of the "dataset" parameters (by parameter name) so that roles taken from them can be checked.
    """
    return _clean_roles(spec, frame, roles or {}, frames or {}), _clean_params(spec, params or {})


def _clean_roles(
    spec: AnalysisSpec, frame: pd.DataFrame | None, raw: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for role in spec.roles:
        source = frame if role.dataset_param is None else frames.get(role.dataset_param)
        value = raw.get(role.name)
        names = (
            [value] if isinstance(value, str) and value else [v for v in (value or []) if v] if role.multiple else []
        )
        if not role.multiple and isinstance(value, list):
            names = [v for v in value if v][:1]
        if not names:
            if role.optional:
                continue
            raise InvalidInputError(
                f"Elija {'al menos una columna para' if role.multiple else 'la columna de'} «{role.label}»."
            )
        if role.multiple:
            if len(names) < role.min_count:
                raise InvalidInputError(f"«{role.label}» necesita al menos {role.min_count} columna(s).")
            if role.max_count and len(names) > role.max_count:
                raise InvalidInputError(f"«{role.label}» admite como máximo {role.max_count} columna(s).")
            if len(set(names)) != len(names):
                raise InvalidInputError(f"«{role.label}» repite una columna.")
        for name in names:
            _check_column(source, name, role)
        clean[role.name] = names if role.multiple else names[0]
    return clean


def _check_column(frame: pd.DataFrame | None, name: str, role: RoleSpec) -> None:
    if frame is None:
        raise InvalidInputError(
            "Elija el segundo dataset." if role.dataset_param else "Elija un dataset para tomar las columnas."
        )
    if name not in frame.columns:
        where = "en el segundo dataset" if role.dataset_param else "en el dataset"
        raise InvalidInputError(f"La columna «{name}» no existe {where}.")
    allowed = ROLE_KINDS.get(role.kind)
    if allowed is not None and classify_column(frame[name]) not in allowed:
        raise InvalidInputError(
            f"«{role.label}» necesita una columna {_KIND_NAMES.get(role.kind, role.kind)}: «{name}» no lo es."
        )


def _clean_params(spec: AnalysisSpec, raw: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for field in spec.params:
        if field.visible_if is not None:
            other, values = field.visible_if
            if str(clean.get(other)) not in values:
                clean[field.name] = _default(field)
                continue
        value = raw.get(field.name)
        if field.kind == "multiselect" and isinstance(value, list):
            clean[field.name] = (
                _coerce(field, value) if value else []
            )  # an empty choice is a choice: the analysis decides
            continue
        if value is None or value == "":
            if field.default is not None or field.optional or field.kind in ("boolean", "multiselect"):
                clean[field.name] = _default(field)
            else:
                raise InvalidInputError(f"Indique «{field.label}».")
            continue
        clean[field.name] = _coerce(field, value)
    return clean


def _default(field: FieldSpec) -> Any:
    if field.kind == "multiselect":
        return list(field.default or [])
    if field.kind == "boolean":
        return bool(field.default)
    return field.default


def _coerce(field: FieldSpec, value: Any) -> Any:
    label = field.label
    if field.kind in ("number", "integer"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InvalidInputError(f"«{label}» debe ser un número.") from None
        if not math.isfinite(number):
            raise InvalidInputError(f"«{label}» debe ser un número finito.")
        if field.kind == "integer":
            if number != int(number):
                raise InvalidInputError(f"«{label}» debe ser un número entero.")
            number = int(number)
        if field.minimum is not None and number < field.minimum:
            raise InvalidInputError(f"«{label}» debe ser ≥ {field.minimum:g}.")
        if field.maximum is not None and number > field.maximum:
            raise InvalidInputError(f"«{label}» debe ser ≤ {field.maximum:g}.")
        return number
    if field.kind == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "si", "sí", "on"}
    if field.kind == "select":
        valid = {option for option, _ in field.options}
        if str(value) not in valid:
            raise InvalidInputError(f"Valor no válido para «{label}».")
        return str(value)
    if field.kind == "multiselect":
        valid = {option for option, _ in field.options}
        items = [str(v) for v in (value if isinstance(value, list) else [value])]
        if any(item not in valid for item in items):
            raise InvalidInputError(f"Valor no válido para «{label}».")
        return items
    if field.kind == "dataset":
        identifier = str(value).strip()
        if not identifier or len(identifier) > 64:
            raise InvalidInputError(f"Elija un dataset para «{label}».")
        return identifier
    limit = MAX_CODE if field.kind == "code" else MAX_TEXT
    text = str(value).strip()
    if len(text) > limit:
        raise InvalidInputError(f"«{label}» admite como máximo {limit} caracteres.")
    return text
