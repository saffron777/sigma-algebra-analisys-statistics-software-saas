"""Dataset entities and helpers to describe/normalise tabular data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from sigma.domain.errors import InvalidInputError

MAX_COLUMNS = 2000  # SQLite default column limit

KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"
KIND_BOOLEAN = "boolean"
KIND_DATETIME = "datetime"
KIND_TEXT = "text"


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: str
    pandas_dtype: str
    n_missing: int
    n_unique: int


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    source_type: str
    n_rows: int
    n_cols: int
    columns: tuple[ColumnInfo, ...]
    created_at: str
    source_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    info: DatasetInfo
    data: pd.DataFrame


def classify_column(series: pd.Series) -> str:
    """Return the analytic kind of a column (numeric, categorical, boolean, datetime, text)."""
    if pdt.is_bool_dtype(series):
        return KIND_BOOLEAN
    if pdt.is_numeric_dtype(series):
        return KIND_NUMERIC
    if pdt.is_datetime64_any_dtype(series):
        return KIND_DATETIME
    if isinstance(series.dtype, pd.CategoricalDtype):
        return KIND_CATEGORICAL
    n_unique = series.nunique(dropna=True)
    limit = max(30, int(0.05 * max(len(series), 1)))
    return KIND_CATEGORICAL if n_unique <= limit else KIND_TEXT


def describe_columns(frame: pd.DataFrame) -> tuple[ColumnInfo, ...]:
    return tuple(
        ColumnInfo(
            name=str(column),
            kind=classify_column(frame[column]),
            pandas_dtype=str(frame[column].dtype),
            n_missing=int(frame[column].isna().sum()),
            n_unique=int(frame[column].nunique(dropna=True)),
        )
        for column in frame.columns
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Make column names strings, non-empty and unique (case-insensitive, as SQLite requires)."""
    if frame.shape[1] == 0:
        raise InvalidInputError("La fuente de datos no contiene columnas.")
    if frame.shape[1] > MAX_COLUMNS:
        raise InvalidInputError(f"El dataset tiene más de {MAX_COLUMNS} columnas.")
    seen: set[str] = set()
    names: list[str] = []
    for position, raw in enumerate(frame.columns, start=1):
        base = str(raw).strip() or f"columna_{position}"
        candidate, suffix = base, 1
        while candidate.lower() in seen:
            suffix += 1
            candidate = f"{base}_{suffix}"
        seen.add(candidate.lower())
        names.append(candidate)
    result = frame.copy()
    result.columns = names
    return result.reset_index(drop=True)


CONVERTIBLE_KINDS = (KIND_NUMERIC, KIND_TEXT, KIND_CATEGORICAL, KIND_DATETIME, KIND_BOOLEAN)
_TRUE_WORDS = {"true", "t", "yes", "y", "si", "sí", "s", "1", "verdadero", "v"}
_FALSE_WORDS = {"false", "f", "no", "n", "0", "falso"}
MAX_CATEGORY_CODE_LENGTH = 40
MAX_CATEGORY_LABEL_LENGTH = 200
MAX_CATEGORY_COUNT = 1000
_CATEGORY_FORBIDDEN = (";", "\r", "\n")


def convert_column(series: pd.Series, kind: str) -> tuple[pd.Series, int]:
    """Convert a column to an analytic kind; returns (new series, values that could not convert)."""
    if kind not in CONVERTIBLE_KINDS:
        raise InvalidInputError(f"Tipo de columna inválido «{kind}».")
    before = int(series.notna().sum())
    if kind == KIND_NUMERIC:
        text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
        converted = pd.to_numeric(text, errors="coerce")
    elif kind == KIND_DATETIME:
        converted = pd.to_datetime(series, errors="coerce", format="mixed")
    elif kind == KIND_BOOLEAN:
        lowered = series.astype("string").str.strip().str.lower()
        converted = lowered.map(lambda v: True if v in _TRUE_WORDS else (False if v in _FALSE_WORDS else pd.NA)).astype(
            "boolean"
        )
    elif kind == KIND_CATEGORICAL:
        converted = series.astype("string").astype("category")
    else:
        converted = series.astype("string").astype(object).where(series.notna(), None)
    failed = before - int(converted.notna().sum())
    return converted, max(failed, 0)


def clean_value_labels(value: Any, variable: str = "") -> list[dict[str, str]]:
    """Normalise the code → label definitions of a categorical variable.

    Empty rows are ignored; every other row needs both a code and a label. Codes and labels must be unique
    (case-insensitive) and must not collide with each other, so the mapping used by the data editor is unambiguous.
    A legacy comma-separated string (the survey design format) becomes ``1..n`` codes with those labels.
    """
    subject = f"«{variable}»" if variable else "la variable"
    if value is None:
        return []
    if isinstance(value, str):
        labels = [part.strip() for part in value.split(",") if part.strip()]
        entries: list[dict[str, str]] = [
            {"code": str(position), "label": label} for position, label in enumerate(labels, start=1)
        ]
    elif isinstance(value, (list, tuple)):
        entries = []
        for item in value:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if not isinstance(item, dict):
                raise InvalidInputError(f"Las categorías de {subject} deben tener un código y una etiqueta.")
            code = str(item.get("code") or "").strip()
            label = str(item.get("label") or "").strip()
            if not code and not label:
                continue
            entries.append({"code": code, "label": label})
    else:
        raise InvalidInputError(f"Las categorías de {subject} no tienen un formato válido.")

    cleaned: list[dict[str, str]] = []
    codes: set[str] = set()
    labels: set[str] = set()
    for entry in entries:
        if len(cleaned) >= MAX_CATEGORY_COUNT:
            raise InvalidInputError(f"Se admiten como máximo {MAX_CATEGORY_COUNT} categorías en {subject}.")
        code, label = entry["code"], entry["label"]
        if not code or not label:
            raise InvalidInputError(f"Cada categoría de {subject} debe tener un código y una etiqueta.")
        if len(code) > MAX_CATEGORY_CODE_LENGTH:
            raise InvalidInputError(
                f"El código «{code[:20]}…» de {subject} es demasiado largo (máximo {MAX_CATEGORY_CODE_LENGTH})."
            )
        if len(label) > MAX_CATEGORY_LABEL_LENGTH:
            raise InvalidInputError(
                f"La etiqueta «{label[:30]}…» de {subject} es demasiado larga (máximo {MAX_CATEGORY_LABEL_LENGTH})."
            )
        if any(char in code for char in _CATEGORY_FORBIDDEN) or any(char in label for char in _CATEGORY_FORBIDDEN):
            raise InvalidInputError(
                f"Los códigos y las etiquetas de {subject} no pueden contener «;» ni saltos de línea."
            )
        code_key, label_key = code.casefold(), label.casefold()
        if code_key in codes:
            raise InvalidInputError(f"El código «{code}» está repetido en {subject}.")
        if label_key in labels:
            raise InvalidInputError(f"La etiqueta «{label}» está repetida en {subject}.")
        codes.add(code_key)
        labels.add(label_key)
        cleaned.append({"code": code, "label": label})

    label_owner = {entry["label"].casefold(): position for position, entry in enumerate(cleaned)}
    for position, entry in enumerate(cleaned):
        owner = label_owner.get(entry["code"].casefold())
        if owner is not None and owner != position:
            raise InvalidInputError(
                f"El código «{entry['code']}» coincide con una etiqueta de {subject}; "
                "use códigos y etiquetas distintos."
            )
    return cleaned


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")
_DATE_SAMPLE = 200


def infer_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that clearly hold ISO dates (``2024-03-15``) into datetime columns."""
    result = frame.copy()
    for column in result.columns:
        series = result[column]
        if pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series) or pdt.is_datetime64_any_dtype(series):
            continue
        sample = series.dropna().astype(str).head(_DATE_SAMPLE)
        if len(sample) and sample.map(lambda v: bool(_ISO_DATE.match(v.strip()))).all():
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
            if converted.notna().sum() == series.notna().sum():
                result[column] = converted
    return result


def stringify_complex_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn lists/dicts inside object columns into text so the table can be stored and analysed."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda v: str(v) if isinstance(v, (list, dict, set, tuple, bytes)) else v
            )
    return result
