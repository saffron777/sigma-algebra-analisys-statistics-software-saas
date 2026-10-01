"""Number formatting for report tables (strings, so the browser renders them verbatim)."""

from __future__ import annotations

import math
from typing import Any

_EMPTY = "—"


def num(value: Any, digits: int = 4) -> str:
    """Format with ``digits`` significant figures, using scientific notation for extreme magnitudes."""
    if value is None:
        return _EMPTY
    if isinstance(value, str):
        return {"inf": "∞", "-inf": "−∞"}.get(value, value)
    if isinstance(value, bool):
        return "Sí" if value else "No"
    value = float(value)
    if math.isnan(value):
        return _EMPTY
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e7 or magnitude < 1e-4:
        return f"{value:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(magnitude)))
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "−")


def integer(value: Any) -> str:
    return _EMPTY if value is None else f"{int(round(float(value))):d}"


def pval(p: Any) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return _EMPTY
    return "< 0.0001" if float(p) < 1e-4 else f"{float(p):.4f}"


def pct(value: Any, digits: int = 1) -> str:
    return _EMPTY if value is None else f"{float(value):.{digits}f} %"


def ratio_pct(value: Any, digits: int = 1) -> str:
    """0.123 → '12.3 %'."""
    return _EMPTY if value is None else f"{100 * float(value):.{digits}f} %"


def interval(low: Any, high: Any, digits: int = 4) -> str:
    return f"[{num(low, digits)}, {num(high, digits)}]"
