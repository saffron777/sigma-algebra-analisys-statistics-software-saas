"""Value recoding by rules written one per line (SPSS *Recode*, Minitab *Code*).

1, 2 -> bajo          several values for one result
3..5 -> medio         inclusive range (``..5`` and ``10..`` are open ends)
>=6  -> alto          comparisons: < <= > >= !=
missing -> 0          the missing values
"N/A" -> missing      quotes keep commas and spaces; the result ``missing`` leaves the cell empty
else -> otros         everything not matched (without it the original values are kept)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from sigma.domain.errors import InvalidInputError

MAX_RULES = 200
_ARROW = re.compile(r"\s*(?:->|=>|→)\s*")
_MISSING = {"missing", "faltante", "vacio", "vacío", "nulo", "nan"}
_ELSE = {"else", "resto", "otros", "otro"}


@dataclass(frozen=True)
class Condition:
    kind: str  # eq | range | cmp | missing
    value: str | None = None
    low: float | None = None
    high: float | None = None
    op: str | None = None


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]
    result: str | None  # None → leave missing
    text: str
    is_else: bool = False


def _split(text: str, separator: str) -> list[str]:
    """Split on a separator that is not inside quotes."""
    parts, current, quote = [], [], ""
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == separator:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts]


def _unquote(token: str) -> tuple[str, bool]:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1], True
    return token, False


def _number(token: str) -> float | None:
    try:
        value = float(token.replace(",", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _condition(token: str, line: int) -> Condition:
    raw, quoted = _unquote(token)
    if not quoted:
        lowered = raw.lower()
        if lowered in _MISSING:
            return Condition("missing")
        if ".." in raw:
            low, high = (part.strip() for part in raw.split("..", 1))
            lo, hi = (_number(low) if low else -math.inf), (_number(high) if high else math.inf)
            if lo is None or hi is None:
                raise InvalidInputError(f"Regla {line}: el rango «{raw}» debe ser dos números separados por «..».")
            return Condition("range", low=lo, high=hi)
        match = re.fullmatch(r"(<=|>=|!=|<|>)\s*(.+)", raw)
        if match:
            number = _number(match.group(2))
            if number is None and match.group(1) != "!=":
                raise InvalidInputError(f"Regla {line}: «{raw}» compara con un número.")
            return Condition("cmp", value=match.group(2), low=number, op=match.group(1))
    if not raw:
        raise InvalidInputError(f"Regla {line}: hay un valor vacío antes de «->».")
    return Condition("eq", value=raw)


def parse_rules(text: str) -> list[Rule]:
    rules: list[Rule] = []
    for number, line in enumerate((text or "").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        halves = _ARROW.split(line, maxsplit=1)
        if len(halves) != 2 or not halves[1].strip():
            raise InvalidInputError(f"Regla {number}: escriba «valor -> resultado».")
        left, right = halves
        result, quoted = _unquote(right.strip())
        target = None if (not quoted and result.lower() in _MISSING) else result
        if left.strip().lower() in _ELSE:
            rules.append(Rule((), target, line, is_else=True))
            continue
        rules.append(Rule(tuple(_condition(t, number) for t in _split(left, ",") if t), target, line))
    if len(rules) > MAX_RULES:
        raise InvalidInputError(f"Se admiten como máximo {MAX_RULES} reglas.")
    if not rules:
        raise InvalidInputError("Escriba al menos una regla, por ejemplo: 1, 2 -> bajo")
    if sum(r.is_else for r in rules) > 1:
        raise InvalidInputError("Solo puede haber una regla «else».")
    return rules


def _mask(series: pd.Series, numeric: bool, condition: Condition) -> np.ndarray:
    missing = series.isna().to_numpy()
    if condition.kind == "missing":
        return missing
    if numeric:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        values = series.astype("string").fillna("").to_numpy(dtype=object)
    if condition.kind == "eq":
        if numeric:
            target = _number(condition.value or "")
            return ~missing & (np.isclose(values, target) if target is not None else np.zeros(len(series), bool))
        return ~missing & (values == condition.value)
    if not numeric:
        raise InvalidInputError("Los rangos y las comparaciones con «<», «>» solo se aplican a columnas numéricas.")
    if condition.kind == "range":
        return ~missing & (values >= condition.low) & (values <= condition.high)
    if condition.op == "!=":
        target = _number(condition.value or "")
        return ~missing & (~np.isclose(values, target) if target is not None else np.ones(len(series), bool))
    compare = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[condition.op or "<"]
    return ~missing & compare(values, condition.low)


def _label(value: object) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def recode_series(series: pd.Series, rules_text: str) -> tuple[pd.Series, list[tuple[str, int]]]:
    """The recoded column and, for each rule, how many rows it changed."""
    rules = parse_rules(rules_text)
    numeric = pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)
    n = len(series)
    assigned = np.zeros(n, dtype=bool)
    chosen = np.empty(n, dtype=object)
    counts: list[tuple[str, int]] = []
    fallback = next((r for r in rules if r.is_else), None)
    for rule in rules:
        if rule.is_else:
            continue
        hit = np.zeros(n, dtype=bool)
        for condition in rule.conditions:
            hit |= _mask(series, numeric, condition)
        hit &= ~assigned
        chosen[hit] = rule.result
        assigned |= hit
        counts.append((rule.text, int(hit.sum())))
    rest = ~assigned
    if fallback is not None:
        chosen[rest] = fallback.result
        counts.append((fallback.text, int(rest.sum())))
        assigned[:] = True
    results = [r.result for r in rules]
    numeric_results = all(r is None or _number(r) is not None for r in results)
    keeps_original = fallback is None and rest.any()
    if numeric_results and (not keeps_original or numeric):
        out = np.full(n, np.nan)
        for index in np.flatnonzero(assigned):
            value = chosen[index]
            out[index] = np.nan if value is None else _number(value)
        if keeps_original:
            out[rest] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)[rest]
        result = pd.Series(out, index=series.index)
        if result.notna().all() and (result == result.round()).all():
            result = result.astype("int64")
        return result, counts
    text = np.full(n, None, dtype=object)
    for index in np.flatnonzero(assigned):
        text[index] = chosen[index]
    if keeps_original:
        originals = series.to_numpy(dtype=object)
        for index in np.flatnonzero(rest):
            value = originals[index]
            text[index] = None if pd.isna(value) else _label(value)
    return pd.Series(text, index=series.index).astype("string"), counts
