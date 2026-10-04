"""Safe evaluator of column expressions (Minitab *Calculator*, SPSS *Compute Variable*, EViews *genr*).

``precio * cantidad``, ``where(edad >= 18, "adulto", "menor")``, ``log(ingreso) - lag(log(ingreso), 12)``…

The text is **never** run with ``eval``: it is parsed with :mod:`ast` and only a whitelist of node types and functions
is interpreted, over pandas columns. There are no attributes, subscripts, lambdas, comprehensions, f-strings or calls
to anything that is not in :data:`REFERENCE`; names are the dataset's columns (or ``col("nombre con espacios")``) and a
few constants. Size, depth, text length and window sizes are capped so that an expression cannot exhaust memory.
Comparisons with a missing value are false (use ``isna(x)``); divisions by zero and overflows give missing values.
"""

from __future__ import annotations

import ast
import difflib
import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats

from sigma.domain.errors import InvalidInputError

MAX_LENGTH = 1000
MAX_NODES = 300
MAX_DEPTH = 40
MAX_ARGS = 12
MAX_TEXT = 200
MAX_LIST = 200
_MAX_INT = 2**53

_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "nan": math.nan}


# ============================================================================ the public entry point
def evaluate(expression: str, frame: pd.DataFrame) -> pd.Series:
    """The expression evaluated over ``frame``, as a column aligned with its rows."""
    return _Evaluator(expression, frame).run()


def referenced_names(expression: str) -> list[str]:
    """Plain names and ``col("…")`` arguments the expression mentions (for messages; nothing is evaluated)."""
    tree = _parse(expression)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
            found.append(node.id)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "col" and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append(argument.value)
    return list(dict.fromkeys(found))


def _parse(expression: str) -> ast.Expression:
    text = (expression or "").strip()
    if not text:
        raise InvalidInputError("Escriba una expresión.")
    if len(text) > MAX_LENGTH:
        raise InvalidInputError(f"La expresión admite como máximo {MAX_LENGTH} caracteres.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        near = (
            f" cerca de «{error.text.strip()[max((error.offset or 1) - 1, 0) : (error.offset or 1) + 11]}»"
            if error.text
            else ""
        )
        raise InvalidInputError(f"La expresión no es válida (error de sintaxis{near}).") from None
    except (ValueError, RecursionError, MemoryError):
        raise InvalidInputError("La expresión no es válida o tiene demasiados niveles.") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise InvalidInputError("La expresión es demasiado larga o compleja.")
    return tree


# ============================================================================ typing helpers
def _kind(value: Any) -> str:
    """num | bool | text | date | list | other."""
    if isinstance(value, pd.Series):
        dtype = value.dtype
        if pdt.is_bool_dtype(dtype):
            return "bool"
        if pdt.is_numeric_dtype(dtype):
            return "num"
        if pdt.is_datetime64_any_dtype(dtype):
            return "date"
        if pdt.is_string_dtype(dtype) or pdt.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            return "text"
        return "other"
    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, int | float | np.integer | np.floating):
        return "num"
    if isinstance(value, str):
        return "text"
    if isinstance(value, pd.Timestamp):
        return "date"
    if isinstance(value, list):
        return "list"
    return "other"


_KIND_NAMES = {"num": "números", "bool": "condiciones", "text": "texto", "date": "fechas", "list": "una lista"}


def _num(value: Any, what: str = "Esta operación") -> Any:
    """A numeric value or column (booleans become 0/1); anything else is an error."""
    kind = _kind(value)
    if kind == "num":
        return value
    if kind == "bool":
        if isinstance(value, pd.Series):
            return value.astype("float64") if value.hasnans else value.astype("int64")
        return int(value)
    raise InvalidInputError(f"{what} necesita números y recibió {_KIND_NAMES.get(kind, 'otro tipo de dato')}.")


def _text(value: Any, what: str = "Esta función") -> pd.Series | str:
    if _kind(value) != "text":
        raise InvalidInputError(f"{what} necesita texto; use to_text(columna) para convertir.")
    return value


def _bool(value: Any) -> Any:
    kind = _kind(value)
    if kind != "bool":
        raise InvalidInputError("«and», «or», «not», &, | y ~ combinan condiciones (comparaciones, isna(), isin()…).")
    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)
    return bool(value)


def _plain(value: Any) -> Any:
    """Numpy scalars become Python ones."""
    return value.item() if isinstance(value, np.generic) else value


def _whole(value: Any, what: str, low: int, high: int) -> int:
    if isinstance(value, pd.Series) or _kind(value) != "num" or value != int(value):
        raise InvalidInputError(f"{what} debe ser un número entero fijo.")
    number = int(value)
    if not low <= number <= high:
        raise InvalidInputError(f"{what} debe estar entre {low} y {high}.")
    return number


def _date(value: Any, what: str) -> pd.Series:
    if not (isinstance(value, pd.Series) and _kind(value) == "date"):
        raise InvalidInputError(f"{what} necesita una columna de fecha; use to_date(columna) si está en texto.")
    return value


# ============================================================================ the evaluator
class _Evaluator:
    def __init__(self, expression: str, frame: pd.DataFrame) -> None:
        self.expression = expression.strip()
        self.frame = frame
        self.index = frame.index
        self.n = len(frame)
        self.tree = _parse(expression)

    # ------------------------------------------------------------------ driver
    def run(self) -> pd.Series:
        try:
            with np.errstate(all="ignore"):
                value = self.visit(self.tree.body, 0)
        except RecursionError:
            raise InvalidInputError("La expresión tiene demasiados niveles.") from None
        if isinstance(value, list):
            raise InvalidInputError("El resultado de una expresión no puede ser una lista.")
        result = value if isinstance(value, pd.Series) else self.series(value)
        return _tidy(result.rename(None))

    def series(self, value: Any) -> pd.Series:
        if isinstance(value, pd.Series):
            return value
        if isinstance(value, list):
            raise InvalidInputError("Una lista solo sirve dentro de isin() o después de «in».")
        return pd.Series(_plain(value), index=self.index)

    def column(self, name: str) -> pd.Series:
        if name not in self.frame.columns:
            raise InvalidInputError(self._unknown(name))
        column = self.frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            column = column.astype(column.cat.categories.dtype)
        return column

    def _unknown(self, name: str) -> str:
        near = difflib.get_close_matches(name, [str(c) for c in self.frame.columns] + list(FUNCTIONS), n=2)
        hint = f" ¿Quiso decir {' o '.join(f'«{n}»' for n in near)}?" if near else ""
        return f"«{name}» no es una columna del dataset ni una función.{hint}"

    # ------------------------------------------------------------------ dispatch
    def visit(self, node: ast.AST, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise InvalidInputError("La expresión tiene demasiados niveles.")
        handler = _HANDLERS.get(type(node))
        if handler is None:
            what = _FORBIDDEN.get(type(node), type(node).__name__)
            raise InvalidInputError(
                f"La expresión usa algo que no está permitido ({what}). Se admiten operaciones aritméticas, "
                "comparaciones, columnas y las funciones de la lista."
            )
        return handler(self, node, depth + 1)

    # ------------------------------------------------------------------ leaves
    def constant(self, node: ast.Constant, depth: int) -> Any:
        value = node.value
        if isinstance(value, bool) or value is None:
            return value if value is not None else math.nan
        if isinstance(value, int):
            return value if abs(value) <= _MAX_INT else float(value)
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            if len(value) > MAX_TEXT:
                raise InvalidInputError(f"Un texto de la expresión admite como máximo {MAX_TEXT} caracteres.")
            return value
        raise InvalidInputError("Solo se admiten números, textos, verdadero/falso y None.")

    def name(self, node: ast.Name, depth: int) -> Any:
        if node.id in self.frame.columns:
            return self.column(node.id)
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise InvalidInputError(self._unknown(node.id))

    def sequence(self, node: ast.List | ast.Tuple, depth: int) -> list:
        if len(node.elts) > MAX_LIST:
            raise InvalidInputError(f"Una lista admite como máximo {MAX_LIST} elementos.")
        items = [self.visit(item, depth) for item in node.elts]
        if any(isinstance(item, pd.Series | list) for item in items):
            raise InvalidInputError("Las listas solo pueden contener números y textos escritos en la expresión.")
        return items

    # ------------------------------------------------------------------ operators
    def unary(self, node: ast.UnaryOp, depth: int) -> Any:
        value = self.visit(node.operand, depth)
        if isinstance(node.op, ast.Not | ast.Invert):
            result = _bool(value)
            return (~result) if isinstance(result, pd.Series) else (not result)
        number = _num(value, "El signo")
        return -number if isinstance(node.op, ast.USub) else number

    def binary(self, node: ast.BinOp, depth: int) -> Any:
        left, right = self.visit(node.left, depth), self.visit(node.right, depth)
        op = type(node.op)
        if op in (ast.BitAnd, ast.BitOr, ast.BitXor):
            a, b = _bool(left), _bool(right)
            return {ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}[op](a, b)
        if op not in _ARITHMETIC:
            raise InvalidInputError("Esa operación no está permitida.")
        kinds = {_kind(left), _kind(right)}
        if op is ast.Add and "text" in kinds:
            if kinds != {"text"}:
                raise InvalidInputError("Solo se puede sumar texto con texto; use to_text(columna) para un número.")
            return self._concat(left, right)
        a, b = _num(left), _num(right)
        if op in (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow):
            a, b = _float(a), _float(b)
        elif not isinstance(a, pd.Series) and not isinstance(b, pd.Series):
            a, b = _float(a) if isinstance(a, float) else a, _float(b) if isinstance(b, float) else b
        try:
            result = _ARITHMETIC[op](a, b)
        except (ZeroDivisionError, OverflowError):
            return math.nan
        return _plain(result)

    def _concat(self, left: Any, right: Any) -> Any:
        if isinstance(left, str) and isinstance(right, str):
            if len(left) + len(right) > MAX_TEXT * 4:
                raise InvalidInputError("El texto resultante es demasiado largo.")
            return left + right
        return self.series(left).astype("string") + (
            right if isinstance(right, str) else self.series(right).astype("string")
        )

    def boolean(self, node: ast.BoolOp, depth: int) -> Any:
        values = [_bool(self.visit(item, depth)) for item in node.values]
        combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        result = values[0]
        for other in values[1:]:
            result = combine(result, other)
        return result

    def compare(self, node: ast.Compare, depth: int) -> Any:
        left = self.visit(node.left, depth)
        result: Any = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator, depth)
            step = self._compare_pair(type(op), left, right)
            result = step if result is None else operator.and_(result, step)
            left = right
        return result

    def _compare_pair(self, op: type, left: Any, right: Any) -> Any:
        if op in (ast.In, ast.NotIn):
            if not isinstance(right, list):
                raise InvalidInputError("Después de «in» escriba una lista: x in [1, 2, 3].")
            found = _isin(self, left, right)
            return found if op is ast.In else ~found if isinstance(found, pd.Series) else not found
        if op not in _COMPARISONS:
            raise InvalidInputError("Esa comparación no está permitida; use ==, !=, <, <=, > o >=.")
        left, right = _align_dates(left, right), _align_dates(right, left)
        ka, kb = _kind(left), _kind(right)
        comparable = ({ka, kb} <= {"num", "bool"}) or (ka == kb and ka in ("text", "date"))
        if not comparable:
            raise InvalidInputError(
                f"No se pueden comparar {_KIND_NAMES.get(ka, 'estos datos')} con {_KIND_NAMES.get(kb, 'estos datos')}; "
                "use to_text(), to_number() o to_date()."
            )
        if not isinstance(left, pd.Series) and not isinstance(right, pd.Series):
            return bool(_COMPARISONS[op](left, right)) if not (_missing(left) or _missing(right)) else False
        try:
            result = _COMPARISONS[op](left, right)
        except TypeError:
            raise InvalidInputError("No se pueden comparar esos valores.") from None
        result = pd.Series(result, index=self.index) if not isinstance(result, pd.Series) else result
        result = result.astype("boolean").fillna(False).astype(bool)
        gaps = _gaps(left) | _gaps(right)
        return result & ~gaps if isinstance(gaps, pd.Series) else result

    # ------------------------------------------------------------------ calls
    def call(self, node: ast.Call, depth: int) -> Any:
        if type(node.func) is not ast.Name or node.keywords:
            raise InvalidInputError("Solo se pueden llamar las funciones de la lista, con argumentos sin nombre.")
        name = node.func.id
        if name == "col":
            if len(node.args) != 1 or not (
                isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
            ):
                raise InvalidInputError(
                    'col() necesita el nombre de la columna entre comillas: col("Nombre con espacios").'
                )
            return self.column(node.args[0].value)
        function = FUNCTIONS.get(name)
        if function is None:
            raise InvalidInputError(self._unknown(name))
        if len(node.args) > MAX_ARGS:
            raise InvalidInputError(f"{name}() admite como máximo {MAX_ARGS} argumentos.")
        arguments = [self.visit(argument, depth) for argument in node.args]
        low, high = function.limits
        if not low <= len(arguments) <= high:
            raise InvalidInputError(
                f"{name}() necesita {low if low == high else f'entre {low} y {high}'} argumento(s)."
            )
        try:
            return function.run(self, *arguments)
        except InvalidInputError:
            raise
        except (TypeError, ValueError, AttributeError, OverflowError, ZeroDivisionError, KeyError, IndexError):
            raise InvalidInputError(f"No se pudo aplicar {name}() a esos valores.") from None


def _float(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.astype("float64")
    return np.float64(value)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _gaps(value: Any) -> Any:
    """Which entries are missing: a boolean column for columns, a plain bool for scalars."""
    return value.isna() if isinstance(value, pd.Series) else _missing(value)


def _align_dates(value: Any, other: Any) -> Any:
    """A text constant written next to a date column is read as a date (``fecha >= "2024-01-01"``)."""
    if isinstance(value, str) and _kind(other) == "date":
        try:
            return pd.Timestamp(value)
        except (ValueError, TypeError):
            raise InvalidInputError(f"«{value}» no es una fecha válida (use AAAA-MM-DD).") from None
    return value


def _isin(ev: _Evaluator, value: Any, options: list) -> Any:
    if isinstance(value, pd.Series):
        return value.isin(options).astype(bool)
    return value in options


def _tidy(series: pd.Series) -> pd.Series:
    """Plain numpy dtypes where possible, no infinities."""
    dtype = series.dtype
    if pdt.is_bool_dtype(dtype):
        return series.fillna(False).astype(bool) if not series.hasnans else series.astype("boolean")
    if pdt.is_numeric_dtype(dtype):
        if not isinstance(dtype, np.dtype):  # Int64 / Float64
            series = series.astype("float64") if series.hasnans else series.astype(dtype.numpy_dtype)
        if series.dtype.kind == "f":
            series = series.replace([np.inf, -np.inf], np.nan)
        return series
    if isinstance(dtype, pd.CategoricalDtype):
        return series.astype(object)
    return series


_ARITHMETIC: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARISONS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_HANDLERS: dict[type, Callable[[_Evaluator, Any, int], Any]] = {
    ast.Constant: _Evaluator.constant,
    ast.Name: _Evaluator.name,
    ast.List: _Evaluator.sequence,
    ast.Tuple: _Evaluator.sequence,
    ast.UnaryOp: _Evaluator.unary,
    ast.BinOp: _Evaluator.binary,
    ast.BoolOp: _Evaluator.boolean,
    ast.Compare: _Evaluator.compare,
    ast.Call: _Evaluator.call,
}
# what is named in the message when something else is used
_FORBIDDEN = {
    ast.Attribute: "acceso a atributos con un punto",
    ast.Subscript: 'corchetes para elegir elementos; use col("nombre") para una columna',
    ast.Lambda: "funciones lambda",
    ast.ListComp: "comprensiones de listas",
    ast.GeneratorExp: "generadores",
    ast.DictComp: "comprensiones",
    ast.SetComp: "comprensiones",
    ast.Dict: "diccionarios",
    ast.Set: "conjuntos",
    ast.JoinedStr: "textos con f",
    ast.NamedExpr: "asignaciones",
    ast.IfExp: "«si… entonces… si no»; use where(condición, a, b)",
    ast.Starred: "argumentos con *",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.Slice: "rebanadas",
}


# ============================================================================ the functions
class _Function:
    def __init__(self, name: str, low: int, high: int, run: Callable[..., Any]) -> None:
        self.name, self.limits, self.run = name, (low, high), run


FUNCTIONS: dict[str, _Function] = {}


def _function(*names: str, args: tuple[int, int] = (1, 1)) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(run: Callable[..., Any]) -> Callable[..., Any]:
        for name in names:
            FUNCTIONS[name] = _Function(name, args[0], args[1], run)
        return run

    return register


def _math(*names: str, func: Callable[[Any], Any]) -> None:
    _function(*names)(lambda ev, x, _f=func: _plain(_f(_float(_num(x)))))


_math("abs", func=np.abs)
_math("sqrt", func=np.sqrt)
_math("exp", func=np.exp)
_math("log", "ln", func=np.log)
_math("log10", func=np.log10)
_math("log2", func=np.log2)
_math("log1p", func=np.log1p)
_math("sin", func=np.sin)
_math("cos", func=np.cos)
_math("tan", func=np.tan)
_math("floor", func=np.floor)
_math("ceil", func=np.ceil)
_math("sign", func=np.sign)
_math("pnorm", func=stats.norm.cdf)
_math("qnorm", func=stats.norm.ppf)


@_function("round", args=(1, 2))
def _round(ev: _Evaluator, x: Any, digits: Any = 0) -> Any:
    return _plain(np.round(_float(_num(x)), _whole(digits, "La cantidad de decimales", -15, 15)))


@_function("clip", args=(3, 3))
def _clip(ev: _Evaluator, x: Any, low: Any, high: Any) -> Any:
    return _plain(np.clip(_float(_num(x)), _num(low), _num(high)))


@_function("minimum", args=(2, 2))
def _minimum(ev: _Evaluator, a: Any, b: Any) -> Any:
    return _plain(np.fmin(_float(_num(a)), _float(_num(b))))


@_function("maximum", args=(2, 2))
def _maximum(ev: _Evaluator, a: Any, b: Any) -> Any:
    return _plain(np.fmax(_float(_num(a)), _float(_num(b))))


# ---- column-wide summaries (one argument): the result is a single number
def _aggregate(name: str, how: Callable[[pd.Series], Any]) -> None:
    def run(ev: _Evaluator, x: Any, *more: Any) -> Any:
        if more:
            raise InvalidInputError(
                f"{name}() resume una columna; para comparar dos valores fila a fila use {name}imum(a, b)."
            )
        if not isinstance(x, pd.Series):
            raise InvalidInputError(f"{name}() resume una columna: {name}(columna).")
        return float(how(_float(_num(x))))

    _function(name, args=(1, 2) if name in ("min", "max") else (1, 1))(run)


_aggregate("mean", lambda s: s.mean())
_aggregate("median", lambda s: s.median())
_aggregate("std", lambda s: s.std(ddof=1))
_aggregate("var", lambda s: s.var(ddof=1))
_aggregate("sum", lambda s: s.sum())
_aggregate("count", lambda s: s.count())
_aggregate("min", lambda s: s.min())
_aggregate("max", lambda s: s.max())


@_function("quantile", args=(2, 2))
def _quantile(ev: _Evaluator, x: Any, q: Any) -> Any:
    if not isinstance(x, pd.Series) or not 0 <= float(_num(q)) <= 1:
        raise InvalidInputError("quantile(columna, q) necesita una columna y q entre 0 y 1.")
    return float(_float(_num(x)).quantile(float(q)))


# ---- row-wise across several columns (SPSS MEAN(a, b, c))
def _across(name: str, how: Callable[[pd.DataFrame], pd.Series]) -> None:
    def run(ev: _Evaluator, *values: Any) -> Any:
        frame = pd.concat([_float(ev.series(_num(v))).reset_index(drop=True) for v in values], axis=1)
        result = how(frame)
        result.index = ev.index
        return result

    _function(name, args=(1, MAX_ARGS))(run)


_across("rowmean", lambda f: f.mean(axis=1))
_across("rowsum", lambda f: f.sum(axis=1, min_count=1))
_across("rowmin", lambda f: f.min(axis=1))
_across("rowmax", lambda f: f.max(axis=1))
_across("rowstd", lambda f: f.std(axis=1, ddof=1))
_across("rownvalid", lambda f: f.notna().sum(axis=1).astype("float64"))


# ---- transformations of a column
def _column_function(name: str, how: Callable[[pd.Series], pd.Series], *, numeric: bool = True) -> None:
    def run(ev: _Evaluator, x: Any) -> Any:
        if not isinstance(x, pd.Series):
            raise InvalidInputError(f"{name}() se aplica a una columna.")
        return how(_float(_num(x)) if numeric else x)

    _function(name)(run)


_column_function("zscore", lambda s: (s - s.mean()) / s.std(ddof=1))
_column_function("minmax", lambda s: (s - s.min()) / (s.max() - s.min()))
_column_function("rank", lambda s: s.rank(method="average"))
_column_function("cumsum", lambda s: s.cumsum())
_column_function("cumprod", lambda s: s.cumprod())
_column_function("cummax", lambda s: s.cummax())
_column_function("cummin", lambda s: s.cummin())


def _shifted(name: str, how: Callable[[pd.Series, int], pd.Series], *, numeric: bool) -> None:
    def run(ev: _Evaluator, x: Any, k: Any = 1) -> Any:
        if not isinstance(x, pd.Series):
            raise InvalidInputError(f"{name}() se aplica a una columna.")
        return how(_float(_num(x)) if numeric else x, _whole(k, "El desfase", 1, max(ev.n, 1)))

    _function(name, args=(1, 2))(run)


_shifted("lag", lambda s, k: s.shift(k), numeric=False)
_shifted("lead", lambda s, k: s.shift(-k), numeric=False)
_shifted("diff", lambda s, k: s.diff(k), numeric=True)
_shifted("pct_change", lambda s, k: s.pct_change(k, fill_method=None), numeric=True)


def _rolling(name: str, how: Callable[[Any], pd.Series]) -> None:
    def run(ev: _Evaluator, x: Any, window: Any) -> Any:
        if not isinstance(x, pd.Series):
            raise InvalidInputError(f"{name}() se aplica a una columna.")
        size = _whole(window, "La ventana", 2, max(ev.n, 2))
        return how(_float(_num(x)).rolling(size, min_periods=size))

    _function(name, args=(2, 2))(run)


_rolling("rolling_mean", lambda r: r.mean())
_rolling("rolling_sum", lambda r: r.sum())
_rolling("rolling_std", lambda r: r.std(ddof=1))
_rolling("rolling_min", lambda r: r.min())
_rolling("rolling_max", lambda r: r.max())


# ---- logic and missing values
@_function("where", args=(3, 3))
def _where(ev: _Evaluator, condition: Any, yes: Any, no: Any) -> Any:
    keep = _bool(condition)
    if not isinstance(keep, pd.Series):
        return yes if keep else no
    if {_kind(yes), _kind(no)} not in ({"num"}, {"bool"}, {"text"}, {"date"}, {"num", "bool"}):
        raise InvalidInputError(
            "where() necesita que los dos resultados sean del mismo tipo (ambos números, ambos texto…)."
        )
    return ev.series(yes).where(keep, ev.series(no)).rename(None)


@_function("isna")
def _isna(ev: _Evaluator, x: Any) -> Any:
    return x.isna() if isinstance(x, pd.Series) else _missing(x)


@_function("notna")
def _notna(ev: _Evaluator, x: Any) -> Any:
    return x.notna() if isinstance(x, pd.Series) else not _missing(x)


@_function("coalesce", args=(2, MAX_ARGS))
def _coalesce(ev: _Evaluator, *values: Any) -> Any:
    result = ev.series(values[0])
    for other in values[1:]:
        result = result.where(result.notna(), ev.series(other))
    return result


@_function("isin", args=(2, 2))
def _isin_function(ev: _Evaluator, x: Any, options: Any) -> Any:
    if not isinstance(options, list):
        raise InvalidInputError("isin(columna, [valores]) necesita una lista entre corchetes.")
    return _isin(ev, x, options)


@_function("between", args=(3, 3))
def _between(ev: _Evaluator, x: Any, low: Any, high: Any) -> Any:
    lower = ev._compare_pair(ast.GtE, x, low)
    upper = ev._compare_pair(ast.LtE, x, high)
    return lower & upper


# ---- text
def _string(x: Any, ev: _Evaluator, name: str) -> pd.Series:
    return ev.series(_text(x, f"{name}()")).astype("string")


for _name, _method in (
    ("lower", "lower"),
    ("upper", "upper"),
    ("strip", "strip"),
    ("title", "title"),
):
    _function(_name)(lambda ev, x, _n=_name, _m=_method: getattr(_string(x, ev, _n).str, _m)())


@_function("length")
def _length(ev: _Evaluator, x: Any) -> Any:
    return _string(x, ev, "length").str.len().astype("float64")


for _name in ("contains", "startswith", "endswith"):

    def _search(ev: _Evaluator, x: Any, needle: Any, _n: str = _name) -> Any:
        if not isinstance(needle, str):
            raise InvalidInputError(f'{_n}(columna, "texto") necesita el texto entre comillas.')
        text = _string(x, ev, _n).str
        found = text.contains(needle, regex=False) if _n == "contains" else getattr(text, _n)(needle)
        return found.fillna(False).astype(bool)

    _function(_name, args=(2, 2))(_search)


@_function("replace", args=(3, 3))
def _replace(ev: _Evaluator, x: Any, old: Any, new: Any) -> Any:
    if not (isinstance(old, str) and isinstance(new, str)) or not old:
        raise InvalidInputError('replace(columna, "buscar", "reemplazo") necesita textos entre comillas.')
    return _string(x, ev, "replace").str.replace(old, new, regex=False)


@_function("substr", args=(3, 3))
def _substr(ev: _Evaluator, x: Any, start: Any, size: Any) -> Any:
    first = _whole(start, "El inicio", 1, 10_000)
    return _string(x, ev, "substr").str.slice(first - 1, first - 1 + _whole(size, "La longitud", 0, 10_000))


@_function("concat", args=(2, MAX_ARGS))
def _concat_function(ev: _Evaluator, *values: Any) -> Any:
    result = ev.series(values[0]).astype("string")
    for other in values[1:]:
        result = result + ev.series(other).astype("string")
    return result


@_function("to_text")
def _to_text(ev: _Evaluator, x: Any) -> Any:
    return ev.series(x).astype("string")


@_function("to_number")
def _to_number(ev: _Evaluator, x: Any) -> Any:
    text = ev.series(x).astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


@_function("to_date")
def _to_date(ev: _Evaluator, x: Any) -> Any:
    if _kind(x) in ("num", "bool"):
        raise InvalidInputError("to_date() necesita texto con fechas.")
    return pd.to_datetime(ev.series(x), errors="coerce", format="mixed")


# ---- dates
def _part(name: str, how: Callable[[Any], Any]) -> None:
    def run(ev: _Evaluator, x: Any) -> Any:
        return how(_date(x, f"{name}()").dt).astype("float64")

    _function(name)(run)


_part("year", lambda d: d.year)
_part("month", lambda d: d.month)
_part("day", lambda d: d.day)
_part("hour", lambda d: d.hour)
_part("quarter", lambda d: d.quarter)
_part("weekday", lambda d: d.weekday)
_part("dayofyear", lambda d: d.dayofyear)
_part("week", lambda d: d.isocalendar().week)


@_function("days_between", args=(2, 2))
def _days_between(ev: _Evaluator, start: Any, end: Any) -> Any:
    return (_date(end, "days_between()") - _date(start, "days_between()")) / pd.Timedelta(days=1)


@_function("add_days", args=(2, 2))
def _add_days(ev: _Evaluator, x: Any, days: Any) -> Any:
    return _date(x, "add_days()") + pd.to_timedelta(_float(_num(days)).clip(-1e6, 1e6), unit="D")


# ============================================================================ documentation (shown on the page)
REFERENCE: tuple[tuple[str, str], ...] = (
    ('col("nombre")', "columna cuyo nombre tiene espacios o símbolos"),
    ("+ − * / // % **", "aritmética (la división por cero da dato faltante)"),
    ("== != < <= > >=   in [..]", "comparaciones (con un dato faltante son falsas)"),
    ("and  or  not   &  |  ~", "combinan condiciones"),
    ("abs(x) sqrt(x) exp(x) log(x) ln(x) log10(x) log2(x) log1p(x)", "matemáticas (ln es lo mismo que log)"),
    ("sin(x) cos(x) tan(x) floor(x) ceil(x) sign(x)", "trigonometría y redondeos"),
    ("round(x, d)  clip(x, a, b)", "redondea · limita entre a y b"),
    ("minimum(a, b)  maximum(a, b)", "menor/mayor fila a fila"),
    ("pnorm(x)  qnorm(p)", "distribución normal estándar: acumulada e inversa"),
    ("mean(x) median(x) std(x) var(x) sum(x) count(x) min(x) max(x)", "resumen de una columna (un solo número)"),
    ("quantile(x, q)", "cuantil de una columna, q entre 0 y 1"),
    ("rowmean(a, b, …) rowsum rowmin rowmax rowstd rownvalid", "por fila, entre varias columnas (ignoran faltantes)"),
    ("zscore(x)  minmax(x)  rank(x)", "tipificar · escalar a [0, 1] · rangos"),
    ("cumsum(x) cumprod(x) cummax(x) cummin(x)", "acumulados"),
    ("lag(x, k)  lead(x, k)  diff(x, k)  pct_change(x, k)", "rezago · adelanto · diferencia · variación relativa"),
    ("rolling_mean(x, w) rolling_sum rolling_std rolling_min rolling_max", "medidas móviles con ventana w"),
    ("where(condición, a, b)", "a si se cumple, b si no (anidable)"),
    ("isna(x)  notna(x)  coalesce(a, b, …)", "faltantes · primer valor presente"),
    ("isin(x, [..])  between(x, a, b)", "pertenencia · rango (inclusive)"),
    ("lower upper strip title length", "texto"),
    (
        'contains(x, "t") startswith endswith replace(x, "a", "b") substr(x, inicio, largo)',
        "búsqueda y recorte de texto",
    ),
    ("concat(a, b, …)  to_text(x)  to_number(x)  to_date(x)", "conversiones (concat convierte a texto)"),
    ("year month day hour quarter weekday dayofyear week", "partes de una fecha"),
    ("days_between(inicio, fin)  add_days(fecha, n)", "días entre fechas · sumar días"),
    ("pi  e  nan", "constantes"),
)
