"""Small validation helpers shared by the statistical services."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from sigma.domain.errors import InvalidInputError

PLOT_SAMPLE_LIMIT = 20000


def require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise InvalidInputError(f"Columnas inexistentes en el dataset: {', '.join(missing)}.")


def numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return the column as float, failing clearly if it is not numeric."""
    require_columns(frame, [name])
    series = frame[name]
    if pdt.is_bool_dtype(series):
        return series.astype(float)
    if pdt.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0 or converted.notna().sum() < non_missing:
        raise InvalidInputError(f"La columna «{name}» no es numérica.")
    return converted.astype(float)


def check_confidence(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise InvalidInputError("El nivel de confianza debe estar entre 0 y 1 (p. ej. 0.95).")
    return float(confidence)


def check_alpha(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise InvalidInputError("El nivel de significancia α debe estar entre 0 y 1 (p. ej. 0.05).")
    return float(alpha)


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def cap_for_plot(values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Down-sample a numeric array so charts stay light."""
    if len(values) <= PLOT_SAMPLE_LIMIT:
        return values
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=PLOT_SAMPLE_LIMIT, replace=False)


def label_of(value: object) -> str:
    """Stable text label for a category value (ints that came as floats lose the '.0')."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
