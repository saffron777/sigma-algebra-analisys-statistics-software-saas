"""Converts domain results (dataclasses, numpy, pandas) into plain JSON-compatible structures."""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from sigma.domain.results import SERIALIZE_KEY


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value``; NaN → null, ±inf → "inf"/"-inf" strings."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.metadata.get(SERIALIZE_KEY, True)
        }
    if isinstance(value, (float, np.floating)):
        return _float(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if value is pd.NaT else value.isoformat()
    if isinstance(value, Decimal):
        return _float(float(value))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def _float(number: float) -> float | str | None:
    if math.isnan(number):
        return None
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number
