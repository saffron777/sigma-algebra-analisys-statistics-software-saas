"""The «ficha» of a dataset as report blocks (for the PDF/Word/Excel export of the Datos module)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sigma.domain.datasets import DatasetInfo
from sigma.domain.documents import DataTable, ReportSection

SOURCE_LABELS = {
    "csv": "CSV",
    "excel": "Excel",
    "text": "Texto",
    "sqlserver": "SQL Server",
    "json_api": "API JSON",
    "sample": "Muestra",
}
KIND_LABELS = {
    "numeric": "numérica",
    "categorical": "categórica",
    "boolean": "lógica",
    "datetime": "fecha",
    "text": "texto",
}
PREVIEW_ROWS = 20


def _number(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.6g}"


def _cell(value, native_dates: bool = False):
    """A plain Python value for a report cell (``native_dates`` keeps datetimes for Excel's date format)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime() if native_dates else value.isoformat(sep=" ")
    return value if isinstance(value, (int, float, str)) else str(value)


def dataset_section(info: DatasetInfo, frame: pd.DataFrame) -> ReportSection:
    missing_total = int(sum(c.n_missing for c in info.columns))
    cells = info.n_rows * max(info.n_cols, 1)
    blocks: list[dict] = [
        {"type": "heading", "text": "Resumen"},
        {
            "type": "tiles",
            "items": [
                {"label": "Filas", "value": f"{info.n_rows:,}".replace(",", " ")},
                {"label": "Columnas", "value": str(info.n_cols)},
                {"label": "Origen", "value": SOURCE_LABELS.get(info.source_type, info.source_type)},
                {
                    "label": "Valores vacíos",
                    "value": f"{missing_total:,}".replace(",", " "),
                    "hint": f"{100 * missing_total / cells:.1f} % de las celdas",
                },
                {"label": "Creado", "value": info.created_at[:10]},
            ],
        },
    ]
    heavy = [c.name for c in info.columns if info.n_rows and c.n_missing / info.n_rows > 0.2]
    if heavy:
        blocks.append(
            {
                "type": "alert",
                "kind": "warning",
                "message": "Más del 20 % de valores vacíos en: " + ", ".join(heavy[:10]) + ".",
            }
        )
    blocks.append(
        {
            "type": "table",
            "title": "Columnas",
            "columns": ["Columna", "Tipo", "Tipo de datos", "Vacíos", "Valores únicos"],
            "rows": [
                [
                    c.name,
                    KIND_LABELS.get(c.kind, c.kind),
                    c.pandas_dtype,
                    {"text": str(c.n_missing), "tone": "bad"} if c.n_missing else str(c.n_missing),
                    c.n_unique,
                ]
                for c in info.columns
            ],
        }
    )
    numeric = [c.name for c in info.columns if c.kind == "numeric" and c.name in frame.columns]
    if numeric:
        described = frame[numeric].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.25, 0.5, 0.75]).T
        blocks.append(
            {
                "type": "table",
                "title": "Resumen de las columnas numéricas",
                "columns": ["Columna", "n", "Media", "Desv. est.", "Mín.", "Q1", "Mediana", "Q3", "Máx."],
                "rows": [
                    [
                        name,
                        int(row["count"]),
                        _number(row["mean"]),
                        _number(row["std"]),
                        _number(row["min"]),
                        _number(row["25%"]),
                        _number(row["50%"]),
                        _number(row["75%"]),
                        _number(row["max"]),
                    ]
                    for name, row in described.iterrows()
                ],
            }
        )
    categorical = [c.name for c in info.columns if c.kind in ("categorical", "boolean") and c.name in frame.columns]
    for name in categorical[:6]:
        counts = frame[name].value_counts(dropna=True).head(8)
        total = max(int(frame[name].notna().sum()), 1)
        blocks.append(
            {
                "type": "table",
                "title": f"Frecuencias de «{name}»",
                "columns": ["Valor", "Frecuencia", "%"],
                "rows": [[str(value), int(n), f"{100 * n / total:.1f}"] for value, n in counts.items()],
                "note": "Se muestran los 8 valores más frecuentes." if frame[name].nunique() > 8 else None,
            }
        )
    head = frame.head(PREVIEW_ROWS)
    blocks.append(
        {
            "type": "table",
            "title": f"Primeras {len(head)} filas",
            "columns": [str(c) for c in head.columns],
            "rows": [[_cell(v) for v in row] for row in head.to_numpy(dtype=object).tolist()],
        }
    )
    return ReportSection(
        title=f"Ficha del dataset «{info.name}»",
        subtitle=f"{info.n_rows:,} filas × {info.n_cols} columnas".replace(",", " "),
        meta=[f"Origen: {SOURCE_LABELS.get(info.source_type, info.source_type)}", f"Creado: {info.created_at[:10]}"],
        blocks=blocks,
    )


def dataset_data_table(info: DatasetInfo, frame: pd.DataFrame) -> DataTable:
    """Every row of the dataset (only the Excel export carries it)."""
    return DataTable(
        name="Datos",
        columns=[str(c) for c in frame.columns],
        rows=[[_cell(v, native_dates=True) for v in row] for row in frame.to_numpy(dtype=object).tolist()],
    )
