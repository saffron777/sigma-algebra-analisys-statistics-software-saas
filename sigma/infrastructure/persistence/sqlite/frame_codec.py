"""Round-trips a DataFrame through SQLite while preserving its pandas dtypes.

SQLite has only a handful of storage classes, so datetimes, categoricals, timedeltas and
nullable types are encoded to plain values on write and restored from a dtype map on read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def encode_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (frame safe to write with ``to_sql``, original dtype per column)."""
    dtypes = {str(column): str(frame[column].dtype) for column in frame.columns}
    encoded = {column: _encode_series(frame[column]) for column in frame.columns}
    return pd.DataFrame(encoded, index=frame.index), dtypes


def _encode_series(series: pd.Series) -> pd.Series:
    if pdt.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return _with_nulls(series.dt.strftime(DATETIME_FORMAT), series.notna())
    if pdt.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    if isinstance(series.dtype, pd.CategoricalDtype) or pdt.is_string_dtype(series) or series.dtype == object:
        return _with_nulls(series.map(_plain_scalar), series.notna())
    if pdt.is_unsigned_integer_dtype(series) and not series.isna().any() and series.max() > np.iinfo("int64").max:
        return series.astype("float64")
    return series


def _with_nulls(values: pd.Series, present: pd.Series) -> pd.Series:
    return values.astype(object).where(present, None)


def _plain_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None or value is pd.NA:
        return value
    return str(value)


def decode_frame(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Restore the dtypes recorded by :func:`encode_frame`."""
    result = frame.copy()
    for column in result.columns:
        tag = dtypes.get(str(column))
        if tag:
            result[column] = _decode_series(result[column], tag)
    return result


def _decode_series(series: pd.Series, tag: str) -> pd.Series:
    if tag.startswith("datetime64"):
        parsed = pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce")
        if "," in tag:  # timezone-aware, e.g. "datetime64[ns, UTC]"
            zone = tag.split(",", 1)[1].strip().rstrip("]")
            parsed = parsed.dt.tz_localize("UTC").dt.tz_convert(zone)
        return parsed
    if tag.startswith("timedelta"):
        return pd.to_timedelta(series, unit="s")
    if tag == "object":
        return series
    if tag == "bool" and series.isna().any():
        return series.astype("boolean")
    try:
        return series.astype(tag)
    except (ValueError, TypeError):
        return series  # e.g. an integer column that now contains NULLs: keep it as float
