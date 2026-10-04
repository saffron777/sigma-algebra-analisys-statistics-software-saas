"""Shared pieces of the model pages: a design from the chosen columns, readable term names, coefficient tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sigma.domain.analysis.output import P
from sigma.domain.analysis.spec import FieldSpec, OutputSpec, RoleSpec
from sigma.domain.datasets import KIND_BOOLEAN, KIND_NUMERIC, classify_column
from sigma.domain.errors import ComputationError, InvalidInputError

MIN_ROWS = 8
MAX_LEVELS = 30
RESPONSE = RoleSpec("response", "Respuesta (Y)", "numeric")
PREDICTORS = RoleSpec(
    "predictors",
    "Predictores (números y categorías)",
    "any",
    multiple=True,
    help="Las columnas de texto o con pocas categorías se tratan como factores (variables ficticias automáticas).",
)
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
PREDICTIONS = OutputSpec("predicciones", "Predicciones y residuos")


@dataclass
class Design:
    """The analysis table of a model: renamed columns, the formula over them and how to show each term."""

    data: pd.DataFrame
    response: str
    numeric: list[str]
    factors: list[str]
    formula: str
    names: dict[str, str]  # internal token -> the user's column name
    used: pd.Index  # rows of the original frame that made it into the fit
    dropped: int
    extra: dict[str, str] = field(default_factory=dict)  # role -> token for additional columns (weights, offset…)

    def pretty(self, term: str) -> str:
        """``C(v3)[T.B]`` → ``g = B``, ``I(v1 ** 2)`` → ``x1²``, ``v1:v2`` → ``x1 × x2``."""
        text = term
        text = re.sub(r"C\((v\d+)\)\[T\.([^\]]*)\]", lambda m: f"{self.names[m.group(1)]} = {m.group(2)}", text)
        text = re.sub(r"I\((v\d+) \*\* (\d+)\)", lambda m: self.names[m.group(1)] + _superscript(m.group(2)), text)
        text = re.sub(r"\bv\d+\b", lambda m: self.names[m.group(0)], text)
        return text.replace(":", " × ")

    def level_info(self) -> dict[str, list[str]]:
        return {self.names[f]: sorted(self.data[f].astype(str).unique()) for f in self.factors}


def _superscript(digits: str) -> str:
    return digits.translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _factor_like(series: pd.Series) -> bool:
    return classify_column(series) not in (KIND_NUMERIC, KIND_BOOLEAN)


def build_design(
    frame: pd.DataFrame,
    response: str | None,
    predictors: list[str],
    *,
    degree: int = 1,
    interactions: str = "none",
    extra: dict[str, str] | None = None,
    constant: bool = True,
    response_kind: str = "numeric",
    force_factors: tuple[str, ...] = (),
) -> Design:
    """Select the columns, drop incomplete rows and write the patsy formula.

    ``interactions``: none | two (all pairs of predictors) | full (every product). ``degree`` > 1 adds powers of the
    numeric predictors. ``extra`` maps a role (``weights``, ``offset``…) to a column that must be complete as well.
    """
    columns = [c for c in dict.fromkeys([*([response] if response else []), *predictors, *(extra or {}).values()])]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InvalidInputError(f"La columna «{missing[0]}» no existe en el dataset.")
    subset = frame[columns]
    complete = subset.dropna()
    tokens = {c: f"v{i}" for i, c in enumerate(columns)}
    data = pd.DataFrame(index=complete.index)
    numeric: list[str] = []
    factors: list[str] = []
    for column in columns:
        token, values = tokens[column], complete[column]
        if column == response:
            data[token] = values if response_kind == "raw" else pd.to_numeric(values, errors="coerce").astype(float)
        elif column in (extra or {}).values():
            data[token] = pd.to_numeric(values, errors="coerce").astype(float)
        elif column in force_factors or _factor_like(values):
            levels = values.astype(str).nunique()
            if levels > MAX_LEVELS:
                raise InvalidInputError(
                    f"«{column}» tiene {levels} categorías; un factor admite como máximo {MAX_LEVELS}."
                )
            if levels < 2:
                raise InvalidInputError(f"«{column}» tiene una sola categoría: no aporta al modelo.")
            data[token] = values.astype(str)
            factors.append(token)
        else:
            data[token] = pd.to_numeric(values.astype(float), errors="coerce")
            numeric.append(token)
    if response_kind != "raw":
        data = data.dropna()
    if len(data) < MIN_ROWS:
        raise InvalidInputError(f"Se necesitan al menos {MIN_ROWS} filas completas (hay {len(data)}).")
    for token in numeric:
        if data[token].nunique() < 2:
            original = next(c for c, t in tokens.items() if t == token)
            raise InvalidInputError(f"«{original}» es constante: no aporta al modelo.")
    terms: list[str] = []
    for token in numeric:
        terms.append(token)
        terms += [f"I({token} ** {d})" for d in range(2, degree + 1)]
    terms += [f"C({token})" for token in factors]
    base = [t for t in [*numeric, *[f"C({f})" for f in factors]]]
    if interactions != "none" and len(base) > 1:
        from itertools import combinations

        depth = 2 if interactions == "two" else len(base)
        for size in range(2, depth + 1):
            terms += [":".join(combo) for combo in combinations(base, size)]
    rhs = " + ".join(terms) if terms else "1"
    if not constant:
        rhs += " - 1"
    formula = f"{tokens[response]} ~ {rhs}" if response else rhs
    return Design(
        data=data,
        response=tokens[response] if response else "",
        numeric=numeric,
        factors=factors,
        formula=formula,
        names={t: c for c, t in tokens.items()},
        used=data.index,
        dropped=len(frame) - len(data),
        extra={role: tokens[column] for role, column in (extra or {}).items()},
    )


def dropped_note(out, design: Design) -> None:
    if design.dropped:
        out.alert("info", f"Se omitieron {design.dropped} fila(s) con datos faltantes en las columnas del modelo.")


def ci_columns(alpha: float) -> list[str]:
    level = f"{100 * (1 - alpha):g} %"
    return [f"IC {level} inf.", "IC sup."]


def coefficient_rows(
    design: Design,
    params: pd.Series,
    se: pd.Series,
    stat: pd.Series,
    p: pd.Series,
    ci: pd.DataFrame,
    *,
    exponentiate: bool = False,
) -> list[list[Any]]:
    rows = []
    for term in params.index:
        name = "Constante" if term == "Intercept" else design.pretty(str(term))
        row: list[Any] = [
            name,
            float(params[term]),
            float(se[term]),
            float(stat[term]),
            P(float(p[term])),
            float(ci.loc[term].iloc[0]),
            float(ci.loc[term].iloc[1]),
        ]
        if exponentiate:
            row += [
                float(np.exp(params[term])),
                float(np.exp(ci.loc[term].iloc[0])),
                float(np.exp(ci.loc[term].iloc[1])),
            ]
        rows.append(row)
    return rows


def coefficient_table(
    out,
    design: Design,
    result,
    alpha: float,
    *,
    stat_label: str = "z",
    exponentiate: str | None = None,
    title: str = "Coeficientes",
    note: str | None = None,
) -> None:
    """A coefficients table from a statsmodels results object (``params``, ``bse``, ``pvalues``, ``conf_int``)."""
    params, se, p = result.params, result.bse, result.pvalues
    stat = result.tvalues if hasattr(result, "tvalues") else params / se
    ci = result.conf_int(alpha=alpha)
    if not isinstance(ci, pd.DataFrame):
        ci = pd.DataFrame(ci, index=params.index)
    ci = ci.set_axis(["low", "high"], axis=1)
    head = ["Término", "Coeficiente", "Error est.", stat_label, "Valor p", *ci_columns(alpha)]
    if exponentiate:
        head += [exponentiate, *[f"{c} ({exponentiate})" for c in ci_columns(alpha)]]
    rows = coefficient_rows(design, params, se, stat, p, ci, exponentiate=bool(exponentiate))
    out.table(title, head, rows, note=note)


def need_enough(design: Design, parameters: int, *, per_parameter: int = 1) -> None:
    n = len(design.data)
    if n <= parameters * per_parameter:
        raise InvalidInputError(
            f"El modelo tiene {parameters} parámetros y solo hay {n} filas completas: faltan datos."
        )


def encode_binary(series: pd.Series, name: str, event: str | None = None) -> tuple[pd.Series, str, str]:
    """0/1 version of a two-valued column: the larger value in sort order (or ``event``) is the event."""
    values = series.dropna()
    levels = sorted(values.astype(str).unique(), key=lambda v: (not _is_number(v), float(v) if _is_number(v) else 0, v))
    if len(levels) != 2:
        raise InvalidInputError(f"«{name}» debe tener exactamente 2 valores distintos (tiene {len(levels)}).")
    chosen = event if event in levels else levels[1]
    other = levels[0] if chosen == levels[1] else levels[1]
    return (series.astype(str) == chosen).astype(float).where(series.notna()), chosen, other


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def fit_failure(error: Exception) -> ComputationError:
    text = str(error).strip()
    if "singular" in text.lower() or "perfect" in text.lower():
        text = "hay colinealidad perfecta o separación perfecta entre los predictores y la respuesta"
    return ComputationError(f"No se pudo ajustar el modelo: {text or type(error).__name__}.")


def aic_bic_rows(result) -> list[list[Any]]:
    rows = []
    for label, attr in (("Log-verosimilitud", "llf"), ("AIC", "aic"), ("BIC", "bic")):
        value = getattr(result, attr, None)
        if value is not None and np.isfinite(float(value)):
            rows.append([label, float(value)])
    return rows


def prediction_frame(frame: pd.DataFrame, design: Design, **columns: np.ndarray) -> pd.DataFrame:
    """The original data with the model's columns added (rows left out of the fit stay empty)."""
    result = frame.copy()
    for name, values in columns.items():
        series = pd.Series(np.nan, index=frame.index, dtype=float)
        series.loc[design.used] = values
        result[name] = series
    return result


def formula_code(response: str, predictors: list[str], extra: str = "") -> str:
    """Python that builds the same formula from the real column names (categories are detected at run time)."""
    return (
        "import pandas as pd\nimport statsmodels.formula.api as smf\n\n"
        f"predictors = {predictors!r}\n"
        "terms = [f'Q(\"{c}\")' if pd.api.types.is_numeric_dtype(df[c]) else f'C(Q(\"{c}\"))' for c in predictors]\n"
        f"formula = 'Q(\"{response}\") ~ ' + ' + '.join(terms){extra}\n"
    )
