"""Create and modify variables (SPSS *Transform*, Minitab *Calc*/*Data*, EViews *genr*)."""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import bar_chart
from sigma.domain.analysis.expressions import REFERENCE, evaluate
from sigma.domain.analysis.recoding import recode_series
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import (
    RESULT,
    cell,
    check_new_columns,
    column_name,
    derived_names,
    load_line,
    preview,
    py_name,
    transformed,
)
from sigma.domain.errors import InvalidInputError

NAME = FieldSpec("name", "Nombre de la columna nueva", "text")
OVERWRITE = FieldSpec("overwrite", "Reemplazar si ya existe", "boolean", False)
MODE = FieldSpec(
    "mode",
    "Resultado",
    "select",
    "new",
    (("new", "Añadir columnas nuevas"), ("replace", "Reemplazar las columnas originales")),
)
_REPLACE_HELP = "Con «Añadir», el nombre nuevo es el de la columna más el sufijo indicado."
_SAVE = "\n# sigma.save(df, 'nombre_del_dataset_nuevo')\n"
_COMMON = {"group": "Crear y modificar variables", "module": "transformar", "outputs": (RESULT,), "can_replace": True}


def _numbers(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return frame[names].apply(pd.to_numeric, errors="coerce")


# ============================================================================ compute
def _compute(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = column_name(frame, p["name"], overwrite=p["overwrite"])
    values = evaluate(p["expression"], frame)
    after = frame.copy()
    after[name] = values
    out = transformed("Calcular variable", f"{name} = {p['expression'].strip()}", frame, after, created=[name])
    missing = int(values.isna().sum())
    tiles: list[tuple] = [("Con valor", len(values) - missing), ("Faltantes", missing)]
    if pd.api.types.is_bool_dtype(values):
        tiles.append(("Verdaderos", int(values.sum())))
    elif pd.api.types.is_numeric_dtype(values) and missing < len(values):
        tiles += [("Media", float(values.mean())), ("Mínimo", float(values.min())), ("Máximo", float(values.max()))]
    else:
        tiles.append(("Valores distintos", int(values.nunique())))
    out.tiles(tiles)
    if missing and not frame.empty:
        out.alert(
            "warning",
            f"{missing} fila(s) quedaron sin valor: falta un dato en alguna columna usada o el cálculo es imposible "
            "(división por cero, logaritmo de un número no positivo…).",
        )
    used = [c for c in frame.columns if str(c) in p["expression"]][:5]
    preview(out, after, [*used, name])
    return out.set(column=name, n_missing=missing).build()


def _compute_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return load_line(dataset) + f"df[{py_name(p['name'])}] = sigma.calc(df, {p['expression'].strip()!r})\n" + _SAVE


# ============================================================================ recode
def _recode(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    source = roles["col"]
    name = column_name(frame, p["name"], overwrite=p["overwrite"])
    result, counts = recode_series(frame[source], p["rules"])
    after = frame.copy()
    after[name] = result
    out = transformed("Recodificar valores", f"{source} → {name}", frame, after, created=[name])
    changed = sum(n for _, n in counts)
    out.tiles(
        [
            ("Filas cambiadas", changed),
            ("Sin cambio", len(frame) - changed),
            ("Valores distintos", int(result.nunique())),
        ]
    )
    out.table("Reglas", ["Regla", "Filas"], [[text, n] for text, n in counts])
    counts_table = result.value_counts(dropna=False).head(15)
    out.table(
        f"Valores de «{name}»",
        ["Valor", "Filas", "%"],
        [[cell(v) if not pd.isna(v) else "(faltante)", int(n), 100 * n / len(result)] for v, n in counts_table.items()],
    )
    return out.set(column=name, rules=counts).build()


def _recode_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + f"df[{py_name(p['name'])}] = sigma.recode(df[{py_name(roles['col'])}], {p['rules']!r})\n"
        + _SAVE
    )


# ============================================================================ bin
def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _edges(x: pd.Series, p: dict[str, Any]) -> np.ndarray:
    values = x.dropna()
    if values.empty:
        raise InvalidInputError("La columna no tiene valores numéricos.")
    method = p["method"]
    if method == "custom":
        try:
            cuts = sorted({float(c.replace(",", ".")) for c in str(p["cuts"] or "").replace(";", " ").split() if c})
        except ValueError:
            raise InvalidInputError("Los puntos de corte deben ser números separados por espacios o «;».") from None
        if not cuts:
            raise InvalidInputError("Escriba al menos un punto de corte.")
        return np.array([-np.inf, *cuts, np.inf])
    k = int(p["bins"])
    if method == "width":
        low, high = float(values.min()), float(values.max())
        if low == high:
            raise InvalidInputError("La columna es constante: no se puede dividir en intervalos.")
        return np.linspace(low, high, k + 1)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, k + 1)))
    if len(edges) < 2:
        raise InvalidInputError("La columna es constante: no se puede dividir en intervalos.")
    return edges


def _interval_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        if np.isinf(low):
            labels.append(f"≤ {_fmt(high)}")
        elif np.isinf(high):
            labels.append(f"> {_fmt(low)}")
        else:
            labels.append(f"{'[' if i == 0 else '('}{_fmt(low)}, {_fmt(high)}]")
    return labels


def _bin(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    source = roles["col"]
    name = column_name(frame, p["name"], overwrite=p["overwrite"])
    x = pd.to_numeric(frame[source], errors="coerce")
    edges = _edges(x, p)
    codes = pd.cut(x, bins=edges, include_lowest=True, labels=False)
    labels = _interval_labels(edges) if p["labels"] == "interval" else [str(i + 1) for i in range(len(edges) - 1)]
    mapping = dict(enumerate(labels))
    result = codes.map(mapping).astype("string") if p["labels"] == "interval" else (codes + 1).astype("Int64")
    after = frame.copy()
    after[name] = result
    out = transformed("Agrupar en intervalos", f"{source} → {name}", frame, after, created=[name])
    counts = codes.value_counts().reindex(range(len(labels)), fill_value=0)
    valid = int(counts.sum())
    out.table(
        "Intervalos",
        ["Código", "Intervalo", "Filas", "%"],
        [
            [i + 1, _interval_labels(edges)[i], int(counts[i]), 100 * counts[i] / max(valid, 1)]
            for i in range(len(labels))
        ],
        note=f"{len(frame) - valid} fila(s) sin valor numérico quedan sin intervalo." if valid < len(frame) else None,
    )
    out.chart(
        bar_chart(
            "Filas por intervalo",
            "Intervalo",
            "Filas",
            _interval_labels(edges),
            [{"name": "Filas", "y": counts.to_numpy()}],
        )
    )
    return out.set(column=name, edges=[float(e) for e in edges if np.isfinite(e)], counts=counts.tolist()).build()


def _bin_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    col = py_name(roles["col"])
    if p["method"] == "custom":
        cuts = sorted({float(c.replace(",", ".")) for c in str(p["cuts"] or "").replace(";", " ").split() if c})
        call = f"pd.cut(df[{col}], bins=[-np.inf, {', '.join(repr(c) for c in cuts)}, np.inf], include_lowest=True)"
    elif p["method"] == "quantile":
        call = f"pd.qcut(df[{col}], q={p['bins']}, duplicates='drop')"
    else:
        call = f"pd.cut(df[{col}], bins={p['bins']}, include_lowest=True)"
    return (
        "import numpy as np\nimport pandas as pd\n\n"
        + load_line(dataset)
        + f"df[{py_name(p['name'])}] = {call}\n"
        + _SAVE
    )


# ============================================================================ dummies
def _dummies(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols = roles["cols"]
    pieces, created, rows = [], [], []
    taken = {str(c).lower() for c in frame.columns}
    for col in cols:
        levels = frame[col].dropna().astype(str)
        if levels.nunique() > 50:
            raise InvalidInputError(
                f"«{col}» tiene {levels.nunique()} categorías: demasiadas para variables ficticias (máximo 50)."
            )
        dummies = pd.get_dummies(
            frame[col].astype("string"),
            prefix=col,
            prefix_sep=p["separator"],
            drop_first=p["drop_first"],
            dtype="int64",
        )
        for name in dummies.columns:
            if str(name).lower() in taken:
                raise InvalidInputError(f"Ya existe la columna «{name}»; cambie el separador o renombre la existente.")
            taken.add(str(name).lower())
        pieces.append(dummies)
        created += list(dummies.columns)
        reference = sorted(levels.unique())[0] if p["drop_first"] and len(levels) else None
        rows.append([col, levels.nunique(), len(dummies.columns), reference or "—"])
    check_new_columns(len(created))
    base = frame if p["keep_original"] else frame.drop(columns=cols)
    after = pd.concat([base, *pieces], axis=1)
    out = transformed("Variables ficticias (dummies)", ", ".join(cols), frame, after, created=created)
    out.table(
        "Columnas creadas",
        ["Columna de origen", "Categorías", "Columnas ficticias", "Categoría de referencia"],
        rows,
        note="Las filas con la categoría faltante quedan con ceros." if frame[cols].isna().any().any() else None,
    )
    preview(out, after, created)
    return out.set(created=created).build()


def _dummies_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cols = roles["cols"]
    drop = f", drop_first={p['drop_first']}" if p["drop_first"] else ""
    return (
        "import pandas as pd\n\n"
        + load_line(dataset)
        + f"dummies = pd.get_dummies(df[{cols!r}].astype('string'), prefix_sep={p['separator']!r}{drop}, dtype=int)\n"
        + (
            "df = pd.concat([df, dummies], axis=1)\n"
            if p["keep_original"]
            else f"df = pd.concat([df.drop(columns={cols!r}), dummies], axis=1)\n"
        )
        + _SAVE
    )


# ============================================================================ standardise / rescale
_SCALES = (
    ("zscore", "Tipificar (z): (x − media) / desv. estándar", "_z"),
    ("minmax", "Escalar a [0, 1]: (x − mín) / (máx − mín)", "_mm"),
    ("robust", "Robusto: (x − mediana) / rango intercuartílico", "_rob"),
    ("maxabs", "Dividir por el máximo absoluto", "_abs"),
    ("center", "Centrar: x − media", "_c"),
)


def _scale_params(x: pd.Series, method: str) -> tuple[float, float, str]:
    """(shift, divisor, what they are) so that result = (x − shift) / divisor."""
    if method == "zscore":
        return float(x.mean()), float(x.std(ddof=1)), "media y desv. estándar"
    if method == "minmax":
        return float(x.min()), float(x.max() - x.min()), "mínimo y rango"
    if method == "robust":
        q1, q3 = x.quantile([0.25, 0.75])
        return float(x.median()), float(q3 - q1), "mediana y rango intercuartílico"
    if method == "maxabs":
        return 0.0, float(x.abs().max()), "máximo absoluto"
    return float(x.mean()), 1.0, "media"


def _standardize(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols, method = roles["cols"], p["method"]
    suffix = {m: s for m, _, s in _SCALES}[method]
    names = derived_names(frame, cols, suffix, replace=p["mode"] == "replace")
    after = frame.copy()
    rows = []
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce")
        shift, divisor, _ = _scale_params(x, method)
        if not np.isfinite(divisor) or divisor == 0:
            raise InvalidInputError(f"«{col}» es constante o tiene muy pocos datos: no se puede escalar.")
        after[names[col]] = (x - shift) / divisor
        rows.append([col, shift, divisor, int(x.isna().sum())])
    out = transformed(
        "Tipificar y normalizar",
        dict((m, label) for m, label, _ in _SCALES)[method],
        frame,
        after,
        created=[n for c, n in names.items() if n != c],
    )
    out.table(
        "Parámetros usados",
        ["Columna", "Se resta", "Se divide por", "Faltantes"],
        rows,
        note="Guarde estos valores para aplicar la misma transformación a datos nuevos.",
    )
    preview(out, after, list(names.values()))
    return out.set(parameters={r[0]: {"shift": r[1], "divisor": r[2]} for r in rows}).build()


def _standardize_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    method, suffix = p["method"], {m: s for m, _, s in _SCALES}[p["method"]]
    expression = {
        "zscore": "(x - x.mean()) / x.std(ddof=1)",
        "minmax": "(x - x.min()) / (x.max() - x.min())",
        "robust": "(x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))",
        "maxabs": "x / x.abs().max()",
        "center": "x - x.mean()",
    }[method]
    target = "col" if p["mode"] == "replace" else f"col + {suffix!r}"
    return (
        "import pandas as pd\n\n"
        + load_line(dataset)
        + f"for col in {roles['cols']!r}:\n    x = pd.to_numeric(df[col], errors='coerce')\n    df[{target}] = {expression}\n"
        + _SAVE
    )


# ============================================================================ mathematical transformations
_FUNCTIONS = (
    ("log", "Logaritmo natural", "_ln"),
    ("log10", "Logaritmo base 10", "_log10"),
    ("log1p", "log(1 + x)", "_log1p"),
    ("sqrt", "Raíz cuadrada", "_raiz"),
    ("cbrt", "Raíz cúbica", "_raiz3"),
    ("square", "Cuadrado", "_cuad"),
    ("reciprocal", "Inverso 1/x", "_inv"),
    ("exp", "Exponencial", "_exp"),
    ("abs", "Valor absoluto", "_abs"),
    ("boxcox", "Box-Cox (λ estimado, x > 0)", "_bc"),
    ("yeojohnson", "Yeo-Johnson (λ estimado)", "_yj"),
)
_VALID = {
    "log": lambda x: x > 0,
    "log10": lambda x: x > 0,
    "log1p": lambda x: x > -1,
    "sqrt": lambda x: x >= 0,
    "reciprocal": lambda x: x != 0,
}


def _apply_function(x: pd.Series, name: str) -> tuple[pd.Series, float | None]:
    with np.errstate(all="ignore"):
        if name in ("boxcox", "yeojohnson"):
            values = x.dropna()
            if len(values) < 3:
                raise InvalidInputError("Se necesitan al menos 3 datos para estimar λ.")
            if name == "boxcox":
                if (values <= 0).any():
                    raise InvalidInputError("Box-Cox necesita valores estrictamente positivos; pruebe con Yeo-Johnson.")
                transformed_values, lam = stats.boxcox(values.to_numpy())
            else:
                transformed_values, lam = stats.yeojohnson(values.to_numpy())
            result = pd.Series(np.nan, index=x.index)
            result[values.index] = transformed_values
            return result, float(lam)
        keep = _VALID.get(name, lambda s: s.notna())(x)
        function = {
            "log": np.log, "log10": np.log10, "log1p": np.log1p, "sqrt": np.sqrt, "cbrt": np.cbrt,
            "square": np.square, "reciprocal": np.reciprocal, "exp": np.exp, "abs": np.abs,
        }[name]  # fmt: skip
        result = pd.Series(function(x.to_numpy(dtype=float)), index=x.index).where(keep)
        return result.replace([np.inf, -np.inf], np.nan), None


def _math(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols, name = roles["cols"], p["function"]
    suffix = {n: s for n, _, s in _FUNCTIONS}[name]
    names = derived_names(frame, cols, suffix, replace=p["mode"] == "replace")
    after = frame.copy()
    rows, lost = [], []
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce")
        result, lam = _apply_function(x, name)
        new_missing = int(result.isna().sum() - x.isna().sum())
        if new_missing == int(x.notna().sum()):
            raise InvalidInputError(f"La transformación no es posible para ningún valor de «{col}».")
        after[names[col]] = result
        lost.append(new_missing)
        rows.append([col, float(x.skew()), float(result.skew()), lam, new_missing])
    label = dict((n, lbl) for n, lbl, _ in _FUNCTIONS)[name]
    out = transformed(
        "Transformaciones matemáticas", label, frame, after, created=[n for c, n in names.items() if n != c]
    )
    out.table(
        "Efecto sobre la asimetría",
        ["Columna", "Asimetría antes", "Asimetría después", "λ estimado", "Valores no transformables"],
        rows,
        note="λ solo aparece en Box-Cox y Yeo-Johnson." if name in ("boxcox", "yeojohnson") else None,
    )
    if any(lost):
        out.alert(
            "warning",
            "Algunos valores no se pudieron transformar (fuera del dominio de la función) y quedaron faltantes.",
        )
    preview(out, after, list(names.values()))
    return out.set(lambdas={r[0]: r[3] for r in rows if r[3] is not None}).build()


def _math_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    suffix = {n: s for n, _, s in _FUNCTIONS}[p["function"]]
    target = "col" if p["mode"] == "replace" else f"col + {suffix!r}"
    if p["function"] in ("boxcox", "yeojohnson"):
        call = f"stats.{p['function']}(x.dropna())[0]"
        body = f"    x = pd.to_numeric(df[col], errors='coerce')\n    df[{target}] = np.nan\n    df.loc[x.notna(), {target}] = {call}\n"
    else:
        body = f"    df[{target}] = np.{p['function']}(pd.to_numeric(df[col], errors='coerce'))\n"
    return (
        "import numpy as np\nimport pandas as pd\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"for col in {roles['cols']!r}:\n{body}"
        + _SAVE
    )


# ============================================================================ lags, differences, rolling
_TIME_OPS = (
    ("lag", "Rezago (valor de k filas atrás)", "_lag{k}"),
    ("lead", "Adelanto (valor de k filas adelante)", "_lead{k}"),
    ("diff", "Diferencia con k periodos atrás", "_d{k}"),
    ("pct_change", "Variación relativa con k periodos atrás", "_pct{k}"),
    ("log_diff", "Diferencia de logaritmos (tasa continua)", "_dlog{k}"),
    ("cumsum", "Suma acumulada", "_acum"),
    ("rolling_mean", "Media móvil (ventana k)", "_mm{k}"),
    ("rolling_sum", "Suma móvil (ventana k)", "_ms{k}"),
    ("rolling_std", "Desviación estándar móvil (ventana k)", "_ds{k}"),
    ("ewma", "Media móvil exponencial (α)", "_ewma"),
)


def _time_series(work: pd.Series, operation: str, k: int, alpha: float, by: pd.Series | None) -> pd.Series:
    def one(s: pd.Series) -> pd.Series:
        if operation == "lag":
            return s.shift(k)
        if operation == "lead":
            return s.shift(-k)
        if operation == "diff":
            return s.diff(k)
        if operation == "pct_change":
            return s.pct_change(k, fill_method=None).replace([np.inf, -np.inf], np.nan)
        if operation == "log_diff":
            with np.errstate(all="ignore"):
                return np.log(s.where(s > 0)).diff(k)
        if operation == "cumsum":
            return s.cumsum()
        if operation == "rolling_mean":
            return s.rolling(k, min_periods=k).mean()
        if operation == "rolling_sum":
            return s.rolling(k, min_periods=k).sum()
        if operation == "rolling_std":
            return s.rolling(k, min_periods=k).std()
        return s.ewm(alpha=alpha, adjust=False).mean()

    if by is None:
        return one(work)
    return work.groupby(by, sort=False, dropna=False).transform(one)


def _time_ops(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols, operation, k = roles["cols"], p["operation"], int(p["periods"])
    pattern = {o: s for o, _, s in _TIME_OPS}[operation]
    names = derived_names(frame, cols, pattern.format(k=k), replace=p["mode"] == "replace")
    order = frame.sort_values(roles["time"], kind="stable").index if roles.get("time") else frame.index
    by = frame.loc[order, roles["by"]] if roles.get("by") else None
    after = frame.copy()
    for col in cols:
        work = pd.to_numeric(frame.loc[order, col], errors="coerce")
        after[names[col]] = _time_series(work, operation, k, p["alpha"], by).reindex(frame.index)
    label = {o: lbl for o, lbl, _ in _TIME_OPS}[operation]
    detail = (
        f"{label}"
        + (f" · k = {k}" if "{k}" in pattern else "")
        + (f" · por «{roles['by']}»" if roles.get("by") else "")
    )
    out = transformed(
        "Rezagos, diferencias y acumulados", detail, frame, after, created=[n for c, n in names.items() if n != c]
    )
    new_missing = {names[c]: int(after[names[c]].isna().sum() - frame[c].isna().sum()) for c in cols}
    out.table(
        "Valores que quedan sin dato",
        ["Columna nueva", "Filas sin valor"],
        [[n, m] for n, m in new_missing.items()],
        note="Los primeros (o últimos) periodos no tienen con qué compararse."
        if operation not in ("cumsum", "ewma")
        else None,
    )
    preview(out, after, [*([roles["time"]] if roles.get("time") else []), *cols, *names.values()])
    return out.build()


def _time_ops_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    k, operation = int(p["periods"]), p["operation"]
    pattern = {o: s for o, _, s in _TIME_OPS}[operation]
    target = "col" if p["mode"] == "replace" else f"col + {pattern.format(k=k)!r}"
    expression = {
        "lag": f"x.shift({k})", "lead": f"x.shift(-{k})", "diff": f"x.diff({k})",
        "pct_change": f"x.pct_change({k})", "log_diff": f"np.log(x).diff({k})", "cumsum": "x.cumsum()",
        "rolling_mean": f"x.rolling({k}).mean()", "rolling_sum": f"x.rolling({k}).sum()",
        "rolling_std": f"x.rolling({k}).std()", "ewma": f"x.ewm(alpha={p['alpha']}, adjust=False).mean()",
    }[operation]  # fmt: skip
    sort = f"df = df.sort_values({roles['time']!r})\n" if roles.get("time") else ""
    if roles.get("by"):
        body = f"    df[{target}] = df.groupby({roles['by']!r})[col].transform(lambda x: {expression})\n"
    else:
        body = f"    x = df[col]\n    df[{target}] = {expression}\n"
    return "import numpy as np\n\n" + load_line(dataset) + sort + f"for col in {roles['cols']!r}:\n{body}" + _SAVE


# ============================================================================ date parts
_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
_DATE_PARTS = (
    ("year", "Año"), ("quarter", "Trimestre"), ("month", "Mes (1-12)"), ("month_name", "Nombre del mes"),
    ("week", "Semana ISO"), ("day", "Día del mes"), ("dayofyear", "Día del año"),
    ("weekday", "Día de la semana (1 = lunes)"), ("is_weekend", "Fin de semana (1/0)"), ("hour", "Hora"),
    ("year_month", "Año-mes (texto AAAA-MM)"),
)  # fmt: skip


def _part(dates: pd.Series, part: str) -> pd.Series:
    dt = dates.dt
    if part == "month_name":
        return dt.month.map(lambda m: None if pd.isna(m) else _MONTHS[int(m) - 1]).astype("string")
    if part == "weekday":
        return (dt.weekday + 1).astype("Int64")
    if part == "is_weekend":
        return (dt.weekday >= 5).astype("int64").where(dates.notna())
    if part == "week":
        return dt.isocalendar().week.astype("Int64")
    if part == "year_month":
        return dt.strftime("%Y-%m").astype("string")
    return getattr(dt, part).astype("Int64")


def _date_parts(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    col, parts = roles["col"], p["parts"]
    if not parts:
        raise InvalidInputError("Elija al menos una parte de la fecha.")
    taken = {str(c).lower() for c in frame.columns}
    after, created = frame.copy(), []
    for part in parts:
        name = f"{col}_{part}"
        if name.lower() in taken:
            raise InvalidInputError(f"Ya existe la columna «{name}».")
        taken.add(name.lower())
        after[name] = _part(frame[col], part)
        created.append(name)
    out = transformed("Partes de una fecha", col, frame, after, created=created)
    preview(out, after, [col, *created])
    return out.build()


_PART_CODE = {
    "weekday": "{d}.dt.dayofweek + 1",
    "is_weekend": "({d}.dt.dayofweek >= 5).astype(int)",
    "week": "{d}.dt.isocalendar().week",
    "year_month": "{d}.dt.strftime('%Y-%m')",
    "month_name": "{d}.dt.month.map(dict(enumerate(MONTHS, 1)))",
}


def _date_parts_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    column = f"df[{py_name(roles['col'])}]"
    lines = [
        f"df[{py_name(roles['col'] + '_' + part)}] = " + _PART_CODE.get(part, "{d}.dt." + part).format(d=column)
        for part in p["parts"]
    ]
    months = f"MONTHS = {list(_MONTHS)!r}\n\n" if "month_name" in p["parts"] else ""
    return months + load_line(dataset) + "\n".join(lines) + "\n" + _SAVE


# ============================================================================ text clean-up
_TEXT_OPS = (
    ("trim", "Quitar espacios al inicio y al final"),
    ("collapse", "Reducir espacios repetidos a uno"),
    ("lower", "Minúsculas"),
    ("upper", "Mayúsculas"),
    ("title", "Primera letra de cada palabra en mayúscula"),
    ("no_accents", "Quitar tildes y diéresis"),
)


def _strip_accents(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _clean_text(series: pd.Series, operations: list[str]) -> pd.Series:
    text = series.astype("string")
    for op in operations:
        if op == "trim":
            text = text.str.strip()
        elif op == "collapse":
            text = text.str.replace(r"\s+", " ", regex=True)
        elif op == "lower":
            text = text.str.lower()
        elif op == "upper":
            text = text.str.upper()
        elif op == "title":
            text = text.str.title()
        elif op == "no_accents":
            text = text.map(_strip_accents).astype("string")
    return text


def _text_ops(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols = roles["cols"]
    if not p["operations"]:
        raise InvalidInputError("Elija al menos una operación.")
    order = [name for name, _ in _TEXT_OPS if name in p["operations"]]
    names = derived_names(frame, cols, "_limpio", replace=p["mode"] == "replace")
    after = frame.copy()
    rows = []
    for col in cols:
        cleaned = _clean_text(frame[col], order)
        changed = int((cleaned.fillna("") != frame[col].astype("string").fillna("")).sum())
        after[names[col]] = cleaned
        rows.append([col, changed, int(frame[col].nunique()), int(cleaned.nunique())])
    out = transformed(
        "Limpieza de texto",
        ", ".join(dict(_TEXT_OPS)[o].lower() for o in order),
        frame,
        after,
        created=[n for c, n in names.items() if n != c],
    )
    out.table("Efecto", ["Columna", "Valores modificados", "Categorías antes", "Categorías después"], rows)
    preview(out, after, [*cols, *(n for c, n in names.items() if n != c)])
    return out.build()


def _text_ops_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    call = {
        "trim": ".str.strip()", "collapse": ".str.replace(r'\\s+', ' ', regex=True)", "lower": ".str.lower()",
        "upper": ".str.upper()", "title": ".str.title()",
        "no_accents": ".str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')",
    }  # fmt: skip
    chain = "".join(call[o] for o, _ in _TEXT_OPS if o in p["operations"])
    target = "col" if p["mode"] == "replace" else "col + '_limpio'"
    return (
        load_line(dataset)
        + f"for col in {roles['cols']!r}:\n    df[{target}] = df[col].astype('string'){chain}\n"
        + _SAVE
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, compute=compute, description=description, **{**_COMMON, **kw})


SPECS = (
    _spec(
        "compute",
        "Calcular variable (fórmula)",
        "Crea una columna con una fórmula: aritmética, condiciones, texto, fechas, rezagos y funciones estadísticas.",
        _compute,
        params=(
            NAME,
            FieldSpec(
                "expression",
                "Fórmula",
                "code",
                help='Ejemplos: x1 * 2 + log(x3)   ·   where(x1 > 10, "alto", "bajo")   ·   zscore(y)',
            ),
            OVERWRITE,
        ),
        reference=REFERENCE,
        code=_compute_code,
        example={"params": {"name": "razon", "expression": "where(x2 > 0, x1 / x2, nan)"}, "roles": {}},
    ),
    _spec(
        "recode",
        "Recodificar valores",
        "Cambia valores, rangos o faltantes de una columna por otros según reglas escritas una por línea.",
        _recode,
        roles=(RoleSpec("col", "Columna a recodificar", "any"),),
        params=(
            NAME,
            FieldSpec(
                "rules",
                "Reglas",
                "code",
                help="Una por línea: «1, 2 -> bajo»  ·  «3..5 -> medio»  ·  «>=6 -> alto»  ·  «missing -> 0»  ·  «else -> otros». Sin «else» los demás valores se conservan.",
            ),
            OVERWRITE,
        ),
        code=_recode_code,
        example={"roles": {"col": "g"}, "params": {"name": "grupo2", "rules": "A -> primero\nB, C -> resto"}},
    ),
    _spec(
        "bin",
        "Agrupar en intervalos (discretizar)",
        "Convierte una columna numérica en categorías: intervalos de igual ancho, cuantiles o puntos de corte propios.",
        _bin,
        roles=(RoleSpec("col", "Columna numérica", "numeric"),),
        params=(
            NAME,
            FieldSpec(
                "method",
                "Método",
                "select",
                "width",
                (
                    ("width", "Igual ancho"),
                    ("quantile", "Cuantiles (igual cantidad de filas)"),
                    ("custom", "Puntos de corte propios"),
                ),
            ),
            FieldSpec(
                "bins",
                "Número de intervalos",
                "integer",
                5,
                minimum=2,
                maximum=100,
                visible_if=("method", ("width", "quantile")),
            ),
            FieldSpec(
                "cuts",
                "Puntos de corte",
                "text",
                help="Números separados por espacios o «;»: 18 30 65",
                visible_if=("method", ("custom",)),
            ),
            FieldSpec(
                "labels",
                "Etiquetas",
                "select",
                "interval",
                (("interval", "El intervalo, p. ej. (10, 20]"), ("number", "El número del intervalo (1, 2, 3…)")),
            ),
            OVERWRITE,
        ),
        code=_bin_code,
        example={"roles": {"col": "x1"}, "params": {"name": "x1_grupo", "method": "quantile", "bins": 4}},
    ),
    _spec(
        "dummies",
        "Variables ficticias (dummies)",
        "Crea una columna 0/1 por cada categoría de las columnas elegidas (útil para modelos).",
        _dummies,
        roles=(RoleSpec("cols", "Columnas categóricas", "categorical", multiple=True),),
        params=(
            FieldSpec(
                "drop_first",
                "Omitir la primera categoría (referencia)",
                "boolean",
                True,
                help="Evita la colinealidad perfecta en regresiones con constante.",
            ),
            FieldSpec("keep_original", "Conservar las columnas originales", "boolean", True),
            FieldSpec("separator", "Separador del nombre", "text", "_"),
        ),
        code=_dummies_code,
        example={"roles": {"cols": ["g", "h"]}, "params": {}},
    ),
    _spec(
        "standardize",
        "Tipificar y normalizar",
        "Estandariza o reescala columnas numéricas (z, [0,1], robusto, centrado) y muestra los parámetros usados.",
        _standardize,
        roles=(RoleSpec("cols", "Columnas numéricas", "numeric", multiple=True),),
        params=(FieldSpec("method", "Método", "select", "zscore", tuple((m, lbl) for m, lbl, _ in _SCALES)), MODE),
        code=_standardize_code,
        example={"roles": {"cols": ["x1", "x2"]}, "params": {"method": "zscore"}},
    ),
    _spec(
        "math",
        "Transformaciones matemáticas (log, √, Box-Cox…)",
        "Aplica una función a columnas numéricas; Box-Cox y Yeo-Johnson estiman λ para reducir la asimetría.",
        _math,
        roles=(RoleSpec("cols", "Columnas numéricas", "numeric", multiple=True),),
        params=(FieldSpec("function", "Función", "select", "log", tuple((n, lbl) for n, lbl, _ in _FUNCTIONS)), MODE),
        code=_math_code,
        example={"roles": {"cols": ["x1", "x3"]}, "params": {"function": "yeojohnson"}},
    ),
    _spec(
        "time_ops",
        "Rezagos, diferencias, tasas y acumulados",
        "Rezagos y adelantos, diferencias (también estacionales), variaciones, acumulados y medias móviles, opcionalmente por grupo.",
        _time_ops,
        roles=(
            RoleSpec("cols", "Columnas numéricas", "numeric", multiple=True),
            RoleSpec(
                "time",
                "Ordenar por fecha (opcional)",
                "datetime",
                optional=True,
                help="Si se omite se usa el orden actual de las filas.",
            ),
            RoleSpec("by", "Calcular por grupo (opcional)", "categorical", optional=True),
        ),
        params=(
            FieldSpec("operation", "Operación", "select", "lag", tuple((o, lbl) for o, lbl, _ in _TIME_OPS)),
            FieldSpec(
                "periods",
                "k (periodos o ventana)",
                "integer",
                1,
                minimum=1,
                maximum=10000,
                help="Para diferencias estacionales use la longitud del ciclo (12 en datos mensuales).",
            ),
            FieldSpec(
                "alpha",
                "α de la media exponencial",
                "number",
                0.3,
                minimum=0.001,
                maximum=1,
                visible_if=("operation", ("ewma",)),
            ),
            MODE,
        ),
        code=_time_ops_code,
        example={"roles": {"cols": ["s"], "time": "t"}, "params": {"operation": "diff", "periods": 12}},
    ),
    _spec(
        "date_parts",
        "Extraer partes de una fecha",
        "Año, trimestre, mes, semana, día de la semana, fin de semana… como columnas nuevas (útil para estacionalidad).",
        _date_parts,
        roles=(RoleSpec("col", "Columna de fecha", "datetime"),),
        params=(FieldSpec("parts", "Partes", "multiselect", ["year", "month"], _DATE_PARTS),),
        code=_date_parts_code,
        example={"roles": {"col": "t"}, "params": {"parts": ["year", "quarter", "month_name", "weekday"]}},
    ),
    _spec(
        "text_ops",
        "Limpiar texto",
        "Quita espacios, unifica mayúsculas y tildes en columnas de texto para que las categorías coincidan.",
        _text_ops,
        roles=(RoleSpec("cols", "Columnas de texto", "categorical", multiple=True),),
        params=(
            FieldSpec("operations", "Operaciones", "multiselect", ["trim", "collapse"], _TEXT_OPS),
            MODE,
        ),
        code=_text_ops_code,
        example={"roles": {"cols": ["g"]}, "params": {"operations": ["trim", "upper"], "mode": "new"}},
    ),
)

registry.register(*SPECS)
