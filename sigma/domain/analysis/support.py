"""Small pieces shared by the analysis areas: new column names, table cells and the report of a data transformation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import OutputSpec
from sigma.domain.errors import InvalidInputError

MAX_NAME = 120
MAX_NEW_COLUMNS = 200
RESULT_KEY = "resultado"
RESULT = OutputSpec(RESULT_KEY, "Datos transformados", required=True)


def column_name(frame: pd.DataFrame, raw: Any, *, overwrite: bool = False, what: str = "la columna nueva") -> str:
    """A valid name for a column to create; when it exists it must be allowed to overwrite it."""
    name = str(raw or "").strip()
    if not name:
        raise InvalidInputError(f"Escriba el nombre de {what}.")
    if len(name) > MAX_NAME:
        raise InvalidInputError(f"El nombre de {what} admite como máximo {MAX_NAME} caracteres.")
    clash = next((c for c in frame.columns if str(c).lower() == name.lower()), None)
    if clash is not None and not overwrite:
        raise InvalidInputError(
            f"Ya existe una columna llamada «{clash}»: elija otro nombre o marque «Reemplazar si ya existe»."
        )
    return str(clash) if clash is not None else name


def derived_names(frame: pd.DataFrame, sources: list[str], suffix: str, *, replace: bool) -> dict[str, str]:
    """The name each source column's result gets: the same name (replace) or ``source + suffix``."""
    if replace:
        return {source: source for source in sources}
    taken = {str(c).lower() for c in frame.columns}
    names: dict[str, str] = {}
    for source in sources:
        name = f"{source}{suffix}"
        if name.lower() in taken:
            raise InvalidInputError(
                f"Ya existe la columna «{name}»: use «Reemplazar las columnas originales» o cambie lo que se añade al nombre."
            )
        taken.add(name.lower())
        names[source] = name
    return names


def check_new_columns(count: int) -> None:
    if count > MAX_NEW_COLUMNS:
        raise InvalidInputError(f"Se crearían {count} columnas nuevas; el máximo por operación es {MAX_NEW_COLUMNS}.")


def cell(value: Any) -> Any:
    """A value of a data table as something a report cell shows (missing values are ``None``)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float | np.floating) and not math.isfinite(float(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, np.generic):
        return value.item()
    return value


def preview(out: Output, frame: pd.DataFrame, columns: list[str], title: str = "Vista previa", rows: int = 8) -> None:
    """The first rows of some columns (the ones the operation created or changed)."""
    shown = list(dict.fromkeys(columns))[:12]
    if not shown or frame.empty:
        return
    head = frame[shown].head(rows)
    note = f"Primeras {len(head)} de {len(frame):,} filas.".replace(",", " ") if len(frame) > rows else None
    out.table(title, shown, [[cell(v) for v in row] for row in head.itertuples(index=False)], note)


def transformed(
    title: str, subtitle: str | None, before: pd.DataFrame, after: pd.DataFrame, *, created: list[str] | None = None
) -> Output:
    """The standard opening of a transformation report: sizes, the dataset on offer and nothing else yet."""
    out = Output(title, subtitle)
    items: list[tuple] = [("Filas", len(after)), ("Columnas", after.shape[1])]
    if len(after) != len(before):
        items.append(("Filas antes", len(before)))
    if created:
        items.append(("Columnas nuevas", len(created)))
    out.tiles(items)
    out.dataset(RESULT_KEY, RESULT.label, after)
    out.set(n_rows=len(after), n_cols=after.shape[1], n_rows_before=len(before), created=created or [])
    return out


def py_name(text: str) -> str:
    """A string literal for generated Python."""
    return repr(str(text))


def load_line(dataset: str) -> str:
    return f'df = sigma.load("{dataset}")\n'
