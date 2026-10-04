"""Graphs (Minitab *Graph*, SPSS *Graphs*/Chart Builder, EViews graph views): distributions, relationships, comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.interpolate import griddata
from statsmodels.nonparametric.smoothers_lowess import lowess

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import (
    bar_chart,
    box_chart,
    heatmap_chart,
    hist_chart,
    interval_chart,
    line_chart,
    panels_chart,
    pie_chart,
    raw_chart,
    scatter_chart,
)
from sigma.domain.analysis.output import Output, P, bad, strong
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import InvalidInputError

MAX_GROUPS = 30
MAX_PANELS = 12
MIN_POINTS = 3
AREA = "Distribución"


# ============================================================================ helpers
def numbers(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def axis(series: pd.Series) -> list[Any] | np.ndarray:
    """Values for a chart axis: dates become ISO text, categories stay as text."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "", regex=False).tolist()
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=float)
    return series.astype(str).tolist()


def level_order(series: pd.Series) -> list[Any]:
    levels = list(pd.unique(series.dropna()))
    try:
        levels = sorted(levels)
    except TypeError:
        levels = sorted(levels, key=str)
    if len(levels) > MAX_GROUPS:
        raise InvalidInputError(
            f"«{series.name}» tiene {len(levels)} categorías; para graficar se admiten como máximo {MAX_GROUPS}."
        )
    return levels


def groups_of(frame: pd.DataFrame, value: str, group: str | None) -> list[tuple[str, pd.Series]]:
    """``(label, values)`` per category of ``group`` (one entry with the whole column when there is none)."""
    values = numbers(frame, value)
    if not group:
        return [(value, values.dropna())]
    result = []
    for level in level_order(frame[group]):
        part = values[frame[group] == level].dropna()
        if len(part):
            result.append((str(level), part))
    if not result:
        raise InvalidInputError("No hay datos numéricos en ninguna categoría.")
    return result


def need(values: pd.Series, name: str, count: int = MIN_POINTS) -> None:
    if len(values) < count:
        raise InvalidInputError(f"«{name}» necesita al menos {count} datos válidos para este gráfico.")


def summary(values: pd.Series) -> list[float]:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return [
        len(values),
        values.mean(),
        values.std(ddof=1) if len(values) > 1 else np.nan,
        values.min(),
        q1,
        median,
        q3,
        values.max(),
    ]


SUMMARY_COLUMNS = ["n", "Media", "Desv. est.", "Mínimo", "Q1", "Mediana", "Q3", "Máximo"]


def dropped_note(out: Output, before: int, after: int) -> None:
    if after < before:
        out.alert("info", f"Se omitieron {before - after} fila(s) con datos faltantes en las columnas del gráfico.")


# ============================================================================ histogram
def _curve(kind: str, values: pd.Series) -> list[dict[str, Any]]:
    if kind == "none" or values.std(ddof=1) == 0 or len(values) < 2:
        return []
    spread = float(values.max() - values.min())
    x = np.linspace(values.min() - 0.1 * spread, values.max() + 0.1 * spread, 200)
    if kind == "normal":
        return [{"name": "Normal ajustada", "x": x, "y": stats.norm.pdf(x, values.mean(), values.std(ddof=1))}]
    return [{"name": "Densidad (KDE)", "x": x, "y": stats.gaussian_kde(values.to_numpy())(x)}]


def _histogram(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, by = roles["var"], roles.get("group")
    parts = groups_of(frame, var, by)
    for label, values in parts:
        need(values, label)
    bins = int(p["bins"]) if p["bins"] else None
    out = Output(f"Histograma de {var}", f"por {by}" if by else None)
    pooled = pd.concat([v for _, v in parts])
    out.tiles(
        [
            ("Observaciones", len(pooled)),
            ("Media", pooled.mean()),
            ("Desv. estándar", pooled.std(ddof=1)),
            ("Asimetría", float(stats.skew(pooled))),
        ]
    )
    if by and p["layout"] == "panels":
        if len(parts) > MAX_PANELS:
            raise InvalidInputError(f"Hay {len(parts)} categorías; los paneles admiten como máximo {MAX_PANELS}.")
        panels = [
            {
                **hist_chart(
                    label,
                    var,
                    groups=[{"name": label, "values": v.to_numpy()}],
                    bins=bins,
                    norm=p["norm"],
                    curves=_curve(p["curve"], v),
                ),
                "title": label,
            }
            for label, v in parts
        ]
        out.chart(panels_chart(f"Histograma de {var} por {by}", panels))
    else:
        curves = _curve(p["curve"], pooled) if len(parts) == 1 else []
        out.chart(
            hist_chart(
                f"Histograma de {var}",
                var,
                groups=[{"name": label, "values": v.to_numpy()} for label, v in parts],
                bins=bins,
                norm=p["norm"],
                curves=curves,
            )
        )
    rows = []
    for label, v in parts:
        shapiro = stats.shapiro(v.sample(5000, random_state=1) if len(v) > 5000 else v) if len(v) >= 3 else None
        rows.append(
            [
                label,
                *summary(v),
                float(stats.skew(v)),
                float(stats.kurtosis(v)),
                P(float(shapiro.pvalue)) if shapiro else None,
            ]
        )
    out.table(
        "Estadísticos",
        ["Grupo", *SUMMARY_COLUMNS, "Asimetría", "Curtosis", "p Shapiro-Wilk"],
        rows,
        note="Curtosis en exceso (0 = normal). Shapiro-Wilk con hasta 5 000 datos.",
    )
    return out.set(
        n=len(pooled), mean=float(pooled.mean()), sd=float(pooled.std(ddof=1)), skew=float(stats.skew(pooled))
    ).build()


def _histogram_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    color = f", color={py_name(roles['group'])}" if roles.get("group") else ""
    facet = f", facet_col={py_name(roles['group'])}" if roles.get("group") and p["layout"] == "panels" else ""
    bins = f", nbins={int(p['bins'])}" if p["bins"] else ""
    norm = {"density": "probability density", "count": "", "percent": "percent"}[p["norm"]]
    norm = f", histnorm={norm!r}" if norm else ""
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.histogram(df, x={py_name(roles['var'])}{color}{facet}{bins}{norm}, barmode='overlay', opacity=0.65).show()\n"
    )


# ============================================================================ box / violin / individual values
_BOX_STYLES = (
    ("box", "Diagrama de caja"),
    ("box_points", "Caja con todos los puntos"),
    ("violin", "Violín"),
    ("strip", "Valores individuales (puntos)"),
)


def _fences(values: pd.Series) -> tuple[float, float, int]:
    q1, q3 = values.quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return float(low), float(high), int(((values < low) | (values > high)).sum())


def _boxplot(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    vars_, by = roles["vars"], roles.get("group")
    if by and len(vars_) > 1:
        raise InvalidInputError("Con una columna de grupo elija una sola variable.")
    parts = groups_of(frame, vars_[0], by) if by else [(name, numbers(frame, name).dropna()) for name in vars_]
    for label, values in parts:
        need(values, label)
    style = p["style"]
    title = {
        "box": "Diagrama de caja",
        "box_points": "Diagrama de caja",
        "violin": "Gráfico de violín",
        "strip": "Valores individuales",
    }[style]
    out = Output(f"{title}: {', '.join(vars_)}", f"por {by}" if by else None)
    out.tiles([("Grupos", len(parts)), ("Observaciones", sum(len(v) for _, v in parts))])
    out.chart(
        box_chart(
            f"{title} de {', '.join(vars_)}" + (f" por {by}" if by else ""),
            vars_[0] if len(vars_) == 1 else "Valor",
            [{"name": label, "values": v.to_numpy()} for label, v in parts],
            violin=style == "violin",
            points=style in ("box_points", "violin"),
            strip=style == "strip",
        )
    )
    rows = []
    for label, v in parts:
        _, _, n_out = _fences(v)
        rows.append([label, *summary(v), n_out])
    out.table(
        "Estadísticos",
        ["Grupo", *SUMMARY_COLUMNS, "Atípicos (Tukey)"],
        rows,
        note="Atípicos: fuera de Q1 − 1.5·RIC o Q3 + 1.5·RIC.",
    )
    return out.set(groups=len(parts), medians={label: float(v.median()) for label, v in parts}).build()


def _boxplot_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if roles.get("group"):
        call = (
            f"px.{'violin' if p['style'] == 'violin' else 'box'}(df, x={py_name(roles['group'])}, y={py_name(roles['vars'][0])}"
            + (", points='all'" if p["style"] != "box" else "")
            + ")"
        )
        return "import plotly.express as px\n\n" + load_line(dataset) + f"{call}.show()\n"
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.box(df.melt(value_vars={roles['vars']!r}), x='variable', y='value').show()\n"
    )


# ============================================================================ scatter / bubble
_FITS = (
    ("none", "Sin ajuste"),
    ("linear", "Recta (mínimos cuadrados)"),
    ("quadratic", "Polinomio de grado 2"),
    ("cubic", "Polinomio de grado 3"),
    ("lowess", "Suavizado LOWESS"),
)
_DEGREE = {"linear": 1, "quadratic": 2, "cubic": 3}


def _fit_line(x: np.ndarray, y: np.ndarray, kind: str, smooth: float, want_band: bool):
    """(line points, band or None, statistics row) of one fitted curve."""
    grid = np.linspace(x.min(), x.max(), 120)
    if kind == "lowess":
        fitted = lowess(y, x, frac=smooth, return_sorted=True)
        return {"x": fitted[:, 0], "y": fitted[:, 1]}, None, ["LOWESS", f"ventana {smooth:g}", None]
    degree = _DEGREE[kind]
    if len(x) <= degree + 1 or np.ptp(x) == 0:
        raise InvalidInputError("Hay muy pocos datos (o la variable X es constante) para ese ajuste.")
    coefficients = np.polyfit(x, y, degree)
    predicted = np.polyval(coefficients, x)
    ss_res, ss_tot = float(((y - predicted) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    names = ["x³", "x²", "x", ""][4 - degree - 1 :]
    equation = "y = " + " + ".join(f"{c:.4g}{n}" for c, n in zip(coefficients, names, strict=True))
    band = None
    if degree == 1 and want_band:
        n = len(x)
        s = np.sqrt(ss_res / (n - 2))
        sxx = float(((x - x.mean()) ** 2).sum())
        half = stats.t.ppf(0.975, n - 2) * s * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
        centre = np.polyval(coefficients, grid)
        band = {"name": "IC 95 % de la media", "x": grid, "lower": centre - half, "upper": centre + half}
    return {"x": grid, "y": np.polyval(coefficients, grid)}, band, [equation, r2, None]


def _scatter(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    xn, yn, by, size = roles["x"], roles["y"], roles.get("group"), roles.get("size")
    data = pd.DataFrame({"x": numbers(frame, xn), "y": numbers(frame, yn)})
    if size:
        data["size"] = numbers(frame, size)
    if by:
        data["g"] = frame[by]
    clean = data.dropna()
    need(clean["x"], "los pares (X, Y)", 4)
    out = Output(f"Dispersión de {yn} según {xn}", f"por {by}" if by else None)
    dropped_note(out, len(data), len(clean))
    if size:
        smin, smax = clean["size"].min(), clean["size"].max()
        clean = clean.assign(px=6 + 30 * (clean["size"] - smin) / (smax - smin if smax > smin else 1))
    parts = [(str(level), clean[clean["g"] == level]) for level in level_order(clean["g"])] if by else [(yn, clean)]
    rows, series, lines, bands = [], [], [], []
    for label, part in parts:
        if len(part) < 2:
            continue
        x, y = part["x"].to_numpy(), part["y"].to_numpy()
        item: dict[str, Any] = {"name": label, "x": x, "y": y}
        if size:
            item["sizes"] = part["px"].to_numpy()
            item["text"] = [f"{size} = {v:.4g}" for v in part["size"]]
        series.append(item)
        r = stats.pearsonr(x, y) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
        rho = stats.spearmanr(x, y) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
        row = [
            label,
            len(part),
            float(r.statistic) if r else None,
            P(float(r.pvalue)) if r else None,
            float(rho.statistic) if rho else None,
        ]
        if p["fit"] != "none":
            line, band, info = _fit_line(x, y, p["fit"], p["smooth"], p["ci"] and len(parts) == 1)
            lines.append({"name": f"Ajuste {label}" if by else "Ajuste", **line})
            if band:
                bands.append(band)
            row += [info[0], info[1]]
        rows.append(row)
    if not series:
        raise InvalidInputError("Ninguna categoría tiene suficientes datos.")
    head = ["Serie", "n", "r de Pearson", "p", "ρ de Spearman"] + (["Ajuste", "R²"] if p["fit"] != "none" else [])
    if len(rows) == 1:
        out.tiles([("Observaciones", len(clean)), ("r de Pearson", rows[0][2])])
    else:
        out.tiles([("Observaciones", len(clean)), ("Grupos", len(series))])
    if by and p["layout"] == "panels":
        if len(series) > MAX_PANELS:
            raise InvalidInputError(f"Hay {len(series)} categorías; los paneles admiten como máximo {MAX_PANELS}.")
        panels = [
            {
                **scatter_chart(
                    s["name"], xn, yn, [s], lines=[line for line in lines if line["name"].endswith(s["name"])]
                ),
                "title": s["name"],
            }
            for s in series
        ]
        out.chart(panels_chart(f"{yn} según {xn} por {by}", panels))
    else:
        out.chart(
            scatter_chart(
                f"{yn} según {xn}" + (" (burbujas: " + size + ")" if size else ""),
                xn,
                yn,
                series,
                lines=lines,
                bands=bands,
            )
        )
    out.table("Asociación", head, rows)
    return out.set(n=len(clean), pearson=rows[0][2] if len(rows) == 1 else None).build()


def _scatter_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    extra = ""
    if roles.get("group"):
        extra += f", color={py_name(roles['group'])}"
        if p["layout"] == "panels":
            extra += f", facet_col={py_name(roles['group'])}"
    if roles.get("size"):
        extra += f", size={py_name(roles['size'])}"
    trend = {"linear": ", trendline='ols'", "lowess": ", trendline='lowess'"}.get(p["fit"], "")
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.scatter(df, x={py_name(roles['x'])}, y={py_name(roles['y'])}{extra}{trend}).show()\n"
    )


# ============================================================================ scatter matrix
def _splom(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, by = roles["vars"], roles.get("group")
    data = frame[names].apply(pd.to_numeric, errors="coerce")
    if by:
        data[by] = frame[by]
    clean = data.dropna()
    need(clean[names[0]], "las filas completas", 4)
    if len(clean) > 2000:
        clean = clean.sample(2000, random_state=1).sort_index()
    out = Output("Matriz de dispersión", f"por {by}" if by else None)
    dropped_note(out, len(data), len(frame[names].dropna()) if not by else len(clean))
    parts = [(str(level), clean[clean[by] == level]) for level in level_order(clean[by])] if by else [("Datos", clean)]
    traces = [
        {
            "type": "splom",
            "name": label,
            "dimensions": [{"label": n, "values": part[n].to_numpy()} for n in names],
            "diagonal": {"visible": False},
            "showupperhalf": False,
            "marker": {"size": 4, "opacity": 0.65},
        }
        for label, part in parts
    ]
    out.chart(
        raw_chart(
            traces,
            {
                "dragmode": "select",
                "hovermode": "closest",
                "showlegend": bool(by),
                # declared so the page can theme them (plotly would only create them while drawing)
                **{f"{axis}{i if i > 1 else ''}": {} for i in range(1, len(names) + 1) for axis in ("xaxis", "yaxis")},
            },
            "Matriz de dispersión",
            height=max(420, 170 * len(names)),
        )
    )
    matrix = frame[names].apply(pd.to_numeric, errors="coerce").corr()
    out.table(
        "Correlaciones de Pearson",
        ["Variable", *names],
        [
            [
                name,
                *(
                    strong(float(matrix.at[name, other]))
                    if name != other and abs(matrix.at[name, other]) >= 0.7
                    else float(matrix.at[name, other])
                    for other in names
                ),
            ]
            for name in names
        ],
        note="En negrita: |r| ≥ 0.7. El gráfico muestra hasta 2 000 filas.",
    )
    return out.set(correlations=matrix.round(4).to_dict()).build()


def _splom_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    color = f", color={py_name(roles['group'])}" if roles.get("group") else ""
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.scatter_matrix(df, dimensions={roles['vars']!r}{color}).show()\n"
    )


# ============================================================================ line plot
_SCALES = (("none", "Valores originales"), ("index100", "Índice (primer dato = 100)"), ("zscore", "Tipificados (z)"))


def _line_plot(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ys, xn, by = roles["ys"], roles.get("x"), roles.get("group")
    if by and len(ys) > 1:
        raise InvalidInputError("Con una columna de grupo elija una sola variable Y.")
    order = frame.sort_values(xn, kind="stable").index if xn and p["sort"] else frame.index
    data = frame.loc[order]
    xs = axis(data[xn]) if xn else np.arange(1, len(data) + 1)
    xs_series = pd.Series(xs, index=data.index)
    series, rows = [], []

    def scaled(values: pd.Series) -> pd.Series:
        if p["scale"] == "index100":
            first = values.dropna()
            return 100 * values / first.iloc[0] if len(first) and first.iloc[0] != 0 else values
        if p["scale"] == "zscore":
            return (values - values.mean()) / values.std(ddof=1)
        return values

    members = (
        [(str(level), data[data[by] == level], ys[0]) for level in level_order(data[by])]
        if by
        else [(y, data, y) for y in ys]
    )
    for label, part, column in members:
        values = scaled(numbers(part, column))
        valid = values.dropna()
        need(valid, label, 2)
        series.append(
            {
                "name": label,
                "x": xs_series.loc[part.index].to_numpy(),
                "y": values.to_numpy(),
                "markers": bool(p["markers"]),
            }
        )
        if p["smooth"] > 1:
            series.append(
                {
                    "name": f"{label} (media móvil {p['smooth']})",
                    "x": xs_series.loc[part.index].to_numpy(),
                    "y": values.rolling(int(p["smooth"]), min_periods=int(p["smooth"])).mean().to_numpy(),
                    "dash": "dash",
                }
            )
        rows.append(
            [
                label,
                len(valid),
                float(valid.mean()),
                float(valid.std(ddof=1)) if len(valid) > 1 else None,
                float(valid.min()),
                float(valid.max()),
            ]
        )
    out = Output(f"Gráfico de líneas: {', '.join(ys)}", f"según {xn}" if xn else "según el orden de las filas")
    out.tiles([("Series", len(members)), ("Observaciones", len(data))])
    title = "Gráfico de líneas" + (
        " (valores " + dict(_SCALES)[p["scale"]].lower() + ")" if p["scale"] != "none" else ""
    )
    out.chart(line_chart(title, xn or "Observación", ys[0] if len(ys) == 1 else "Valor", series))
    out.table("Resumen de cada serie", ["Serie", "n", "Media", "Desv. est.", "Mínimo", "Máximo"], rows)
    return out.set(series=len(members)).build()


def _line_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    x = py_name(roles["x"]) if roles.get("x") else "df.index"
    color = f", color={py_name(roles['group'])}" if roles.get("group") else ""
    sort = f"df = df.sort_values({roles['x']!r})\n" if roles.get("x") and p["sort"] else ""
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + sort
        + f"px.line(df, x={x}, y={roles['ys']!r}{color}).show()\n"
    )


# ============================================================================ bar chart
_STATS = (
    ("count", "Cantidad de filas"),
    ("percent", "Porcentaje de las filas"),
    ("mean", "Media"),
    ("median", "Mediana"),
    ("sum", "Suma"),
    ("min", "Mínimo"),
    ("max", "Máximo"),
)
_ERRORS = (
    ("none", "Sin barras de error"),
    ("sd", "± 1 desviación estándar"),
    ("se", "± 1 error estándar"),
    ("ci", "IC 95 % de la media"),
)


def _group_stat(values: pd.Series, how: str, error: str) -> tuple[float, float]:
    n = int(values.count())
    value = (
        {
            "mean": values.mean(),
            "median": values.median(),
            "sum": values.sum(),
            "min": values.min(),
            "max": values.max(),
        }[how]
        if n
        else np.nan
    )
    if how != "mean" or error == "none" or n < 2:
        return float(value), 0.0
    sd = float(values.std(ddof=1))
    half = {"sd": sd, "se": sd / np.sqrt(n), "ci": stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)}[error]
    return float(value), float(half)


def _bar(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cat, value, by = roles["category"], roles.get("value"), roles.get("group")
    how = p["stat"]
    if how not in ("count", "percent") and not value:
        raise InvalidInputError("Esa medida necesita una columna numérica de valores.")
    data = frame[[cat] + ([by] if by else [])].copy()
    if value:
        data["v"] = numbers(frame, value)
    data = data.dropna(subset=[cat] + ([by] if by else []))
    if data.empty:
        raise InvalidInputError("No hay datos para graficar.")
    cats = level_order(data[cat])
    levels = level_order(data[by]) if by else [None]
    table: dict[Any, list[tuple[float, float]]] = {}
    totals = {lv: float(len(data[data[by] == lv]) if by else len(data)) for lv in levels}
    for lv in levels:
        part = data if lv is None else data[data[by] == lv]
        cells = []
        for c in cats:
            sub = part[part[cat] == c]
            if how == "count":
                cells.append((float(len(sub)), 0.0))
            elif how == "percent":
                cells.append((100 * len(sub) / max(totals[lv], 1), 0.0))
            else:
                cells.append(_group_stat(sub["v"], how, p["error"]))
        table[lv] = cells
    order = list(range(len(cats)))
    if p["sort"] != "none" and not by:
        order.sort(key=lambda i: table[None][i][0], reverse=p["sort"] == "desc")
    order = order[: int(p["top"])]
    labels = [str(cats[i]) for i in order]
    label = {"count": "Cantidad", "percent": "Porcentaje"}.get(how, f"{dict(_STATS)[how]} de {value}")
    series = [
        {"name": "Datos" if lv is None else str(lv), "y": np.array([table[lv][i][0] for i in order])} for lv in levels
    ]
    errors = [[table[lv][i][1] for i in order] for lv in levels] if p["error"] != "none" and how == "mean" else None
    out = Output(f"Gráfico de barras: {label} por {cat}", f"por {by}" if by else None)
    out.tiles([("Categorías", len(cats)), ("Filas", len(data))])
    if len(cats) > len(order):
        out.alert("info", f"Se muestran las {len(order)} primeras categorías de {len(cats)}.")
    out.chart(
        bar_chart(
            f"{label} por {cat}",
            cat,
            label,
            labels,
            series,
            stacked=bool(p["stacked"]) and bool(by),
            horizontal=p["orientation"] == "h",
            errors=errors,
        )
    )
    head = [cat, *("Datos" if lv is None else str(lv) for lv in levels)]
    out.table("Valores", head, [[labels[k], *(table[lv][i][0] for lv in levels)] for k, i in enumerate(order)])
    return out.set(categories=len(cats)).build()


def _bar_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cat, how = roles["category"], p["stat"]
    if how == "count":
        body = f"summary = df.groupby({cat!r}).size().reset_index(name='n')\npx.bar(summary, x={cat!r}, y='n').show()\n"
    else:
        column = roles.get("value") or cat
        func = "size" if how == "percent" else how
        body = f"summary = df.groupby({cat!r})[{column!r}].{func}().reset_index()\npx.bar(summary, x={cat!r}, y={column!r}).show()\n"
    return "import plotly.express as px\n\n" + load_line(dataset) + body


# ============================================================================ pareto
def _pareto(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cat, value = roles["category"], roles.get("value")
    data = frame[[cat]].copy()
    data["v"] = numbers(frame, value) if value else 1.0
    totals = data.dropna().groupby(cat, observed=True)["v"].sum().sort_values(ascending=False)
    if (totals < 0).any():
        raise InvalidInputError("Un diagrama de Pareto necesita valores no negativos.")
    if len(totals) < 2 or totals.sum() <= 0:
        raise InvalidInputError("Se necesitan al menos 2 categorías con valores positivos.")
    top = int(p["top"])
    if len(totals) > top:
        rest = totals.iloc[top:].sum()
        totals = totals.iloc[:top]
        if p["other"]:
            totals = pd.concat([totals, pd.Series({"Otros": rest})])
    labels = [str(i) for i in totals.index]
    share = 100 * totals / totals.sum()
    cumulative = share.cumsum()
    vital = int((cumulative.shift(fill_value=0) < 80).sum())
    out = Output(f"Diagrama de Pareto de {cat}", f"según {value}" if value else "según la frecuencia")
    out.tiles([("Categorías", len(labels)), ("Total", float(totals.sum())), ("Categorías hasta el 80 %", vital)])
    out.alert("info", f"Las primeras {vital} categorías explican el {cumulative.iloc[vital - 1]:.1f} % del total.")
    traces = [
        {"type": "bar", "x": labels, "y": totals.to_numpy(), "name": value or "Frecuencia"},
        {
            "type": "scatter",
            "x": labels,
            "y": cumulative.to_numpy(),
            "yaxis": "y2",
            "mode": "lines+markers",
            "name": "% acumulado",
        },
    ]
    layout = {
        "xaxis": {"type": "category", "title": {"text": cat}},
        "yaxis": {"title": {"text": value or "Frecuencia"}},
        "yaxis2": {
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "tickvals": [0, 20, 40, 60, 80, 100],
            "ticksuffix": " %",
            "title": {"text": "% acumulado"},
            "showgrid": False,
        },
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y2",
                "y0": 80,
                "y1": 80,
                "line": {"dash": "dot", "color": "rgba(128,128,128,0.8)"},
            }
        ],
        "showlegend": True,
    }
    out.chart(raw_chart(traces, layout, f"Diagrama de Pareto de {cat}"))
    out.table(
        "Categorías",
        [cat, "Valor", "%", "% acumulado"],
        [
            [labels[i], float(totals.iloc[i]), float(share.iloc[i]), float(cumulative.iloc[i])]
            for i in range(len(labels))
        ],
    )
    return out.set(vital_few=vital, categories=len(labels)).build()


def _pareto_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cat, value = roles["category"], roles.get("value")
    series = f"df.groupby({cat!r})[{value!r}].sum()" if value else f"df[{cat!r}].value_counts()"
    return (
        "import plotly.graph_objects as go\n\n"
        + load_line(dataset)
        + f"totals = {series}.sort_values(ascending=False)\ncum = 100 * totals.cumsum() / totals.sum()\n"
        + "fig = go.Figure([go.Bar(x=totals.index.astype(str), y=totals), go.Scatter(x=totals.index.astype(str), y=cum, yaxis='y2')])\n"
        + "fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 105]))\nfig.show()\n"
    )


# ============================================================================ pie
def _pie(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cat, value = roles["category"], roles.get("value")
    data = frame[[cat]].copy()
    data["v"] = numbers(frame, value) if value else 1.0
    totals = data.dropna().groupby(cat, observed=True)["v"].sum().sort_values(ascending=False)
    if (totals < 0).any() or totals.sum() <= 0:
        raise InvalidInputError("Un gráfico circular necesita valores no negativos y al menos uno positivo.")
    top = int(p["top"])
    if len(totals) > top:
        totals = pd.concat([totals.iloc[:top], pd.Series({"Otros": totals.iloc[top:].sum()})])
    share = 100 * totals / totals.sum()
    out = Output(f"Gráfico circular de {cat}", f"suma de {value}" if value else "frecuencia")
    out.tiles([("Categorías", len(totals)), ("Total", float(totals.sum()))])
    out.chart(
        pie_chart(f"Distribución de {cat}", [str(i) for i in totals.index], totals.to_numpy(), donut=bool(p["donut"]))
    )
    out.table("Porciones", [cat, "Valor", "%"], [[str(k), float(v), float(share[k])] for k, v in totals.items()])
    return out.set(categories=len(totals)).build()


def _pie_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cat, value = roles["category"], roles.get("value")
    totals = f"df.groupby({cat!r})[{value!r}].sum()" if value else f"df[{cat!r}].value_counts()"
    hole = ", hole=0.45" if p["donut"] else ""
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"totals = {totals}\npx.pie(values=totals.values, names=totals.index.astype(str){hole}).show()\n"
    )


# ============================================================================ interval plot
def _interval(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, by = roles["var"], roles["group"]
    level = p["confidence"] / 100
    parts = groups_of(frame, var, by)
    groups, rows = [], []
    for label, values in parts:
        if len(values) < 2:
            continue
        mean, sd, n = float(values.mean()), float(values.std(ddof=1)), len(values)
        half = float(stats.t.ppf((1 + level) / 2, n - 1) * sd / np.sqrt(n))
        groups.append(
            {
                "name": label,
                "mean": mean,
                "lower": mean - half,
                "upper": mean + half,
                "values": values.to_numpy() if p["points"] else None,
            }
        )
        rows.append([label, n, mean, sd, sd / np.sqrt(n), mean - half, mean + half])
    if len(groups) < 1:
        raise InvalidInputError("Cada grupo necesita al menos 2 datos.")
    out = Output(f"Gráfico de intervalos de {var}", f"IC del {p['confidence']:g} % de la media, por {by}")
    out.tiles([("Grupos", len(groups)), ("Confianza", f"{p['confidence']:g} %")])
    overlap = _non_overlapping(rows)
    if overlap:
        out.alert(
            "info",
            "Intervalos que no se solapan (diferencia probable): "
            + "; ".join(f"{a} y {b}" for a, b in overlap[:6])
            + ". Es solo una guía: use ANOVA para comparar con rigor.",
        )
    out.chart(interval_chart(f"IC del {p['confidence']:g} % para la media de {var}", by, var, groups))
    out.table(
        "Intervalos de confianza",
        ["Grupo", "n", "Media", "Desv. est.", "Error est.", "Límite inf.", "Límite sup."],
        rows,
    )
    return out.set(means={r[0]: r[2] for r in rows}).build()


def _non_overlapping(rows: list[list[Any]]) -> list[tuple[str, str]]:
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a[5] > b[6] or b[5] > a[6]:
                pairs.append((a[0], b[0]))
    return pairs


def _interval_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    var, by = py_name(roles["var"]), py_name(roles["group"])
    return (
        "import plotly.graph_objects as go\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"g = df.groupby({by})[{var}].agg(['mean', 'std', 'count'])\n"
        + f"half = stats.t.ppf({(1 + p['confidence'] / 100) / 2}, g['count'] - 1) * g['std'] / g['count'] ** 0.5\n"
        + "go.Figure(go.Scatter(x=g.index.astype(str), y=g['mean'], mode='markers', error_y=dict(type='data', array=half))).show()\n"
    )


# ============================================================================ probability plot
_DISTRIBUTIONS = (
    ("norm", "Normal"),
    ("lognorm", "Lognormal"),
    ("expon", "Exponencial"),
    ("weibull_min", "Weibull"),
    ("gamma", "Gamma"),
    ("logistic", "Logística"),
    ("gumbel_r", "Valor extremo (Gumbel)"),
    ("laplace", "Laplace"),
    ("uniform", "Uniforme"),
)
_POSITIVE = {"lognorm", "expon", "weibull_min", "gamma"}
_ANDERSON = {"norm": "norm", "expon": "expon", "logistic": "logistic", "gumbel_r": "gumbel_r"}


def _probability(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, name = roles["var"], p["distribution"]
    values = numbers(frame, var).dropna().sort_values()
    need(values, var, 8)
    if name in _POSITIVE and (values <= 0).any():
        raise InvalidInputError(
            f"La distribución {dict(_DISTRIBUTIONS)[name]} necesita valores estrictamente positivos."
        )
    dist = getattr(stats, name)
    floc = {"lognorm": 0, "expon": 0, "weibull_min": 0, "gamma": 0}.get(name)
    parameters = dist.fit(values.to_numpy(), floc=floc) if floc is not None else dist.fit(values.to_numpy())
    n = len(values)
    positions = (np.arange(1, n + 1) - 0.3) / (n + 0.4)  # median ranks (Benard)
    theoretical = dist.ppf(positions, *parameters)
    r = float(np.corrcoef(theoretical, values.to_numpy())[0, 1])
    low, high = float(min(theoretical.min(), values.min())), float(max(theoretical.max(), values.max()))
    ks = stats.kstest(values.to_numpy(), dist(*parameters).cdf)
    out = Output(
        f"Gráfico de probabilidad de {var}",
        f"distribución {dict(_DISTRIBUTIONS)[name]} ajustada por máxima verosimilitud",
    )
    tiles: list[tuple] = [("Observaciones", n), ("Correlación del gráfico", r)]
    anderson_stat = anderson_p = None
    if name in _ANDERSON:
        anderson = stats.anderson(values.to_numpy(), dist=_ANDERSON[name], method="interpolate")
        anderson_stat, anderson_p = float(anderson.statistic), float(anderson.pvalue)
        tiles.append(("A² de Anderson-Darling", anderson_stat))
    out.tiles(tiles)
    out.chart(
        scatter_chart(
            f"Gráfico de probabilidad ({dict(_DISTRIBUTIONS)[name]})",
            "Cuantil teórico de la distribución ajustada",
            var,
            [{"name": "Datos", "x": theoretical, "y": values.to_numpy()}],
            lines=[{"name": "Ajuste perfecto", "x": np.array([low, high]), "y": np.array([low, high]), "dash": "dash"}],
        )
    )
    names = {
        "norm": ["media", "desv. estándar"],
        "lognorm": ["forma σ", "ubicación", "escala"],
        "expon": ["ubicación", "escala"],
        "weibull_min": ["forma", "ubicación", "escala"],
        "gamma": ["forma", "ubicación", "escala"],
        "logistic": ["ubicación", "escala"],
        "gumbel_r": ["ubicación", "escala"],
        "laplace": ["ubicación", "escala"],
        "uniform": ["mínimo", "amplitud"],
    }[name]
    out.table(
        "Parámetros estimados",
        ["Parámetro", "Valor"],
        [[label, float(v)] for label, v in zip(names, parameters, strict=False)],
    )
    out.table(
        "Bondad de ajuste",
        ["Medida", "Valor", "p"],
        [
            ["Kolmogorov-Smirnov (parámetros estimados)", float(ks.statistic), P(float(ks.pvalue))],
            ["Correlación del gráfico", r, None],
        ]
        + ([["Anderson-Darling A²", anderson_stat, P(anderson_p)]] if anderson_stat is not None else []),
        note="El p de Kolmogorov-Smirnov es optimista cuando los parámetros se estiman con los mismos datos; el de "
        "Anderson-Darling se interpola de tablas (entre 0.01 y 0.15) y sí tiene en cuenta que se estimaron.",
    )
    verdict = "close" if r >= 0.99 else "far"
    out.alert(
        "success" if verdict == "close" else "warning",
        "Los puntos siguen de cerca la recta: la distribución describe bien los datos."
        if verdict == "close"
        else "Los puntos se apartan de la recta: revise la distribución elegida.",
    )
    return out.set(n=n, correlation=r, parameters=[float(v) for v in parameters], ks_p=float(ks.pvalue)).build()


def _probability_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nimport plotly.express as px\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"x = np.sort(df[{roles['var']!r}].dropna().to_numpy())\nparams = stats.{p['distribution']}.fit(x)\n"
        + f"q = stats.{p['distribution']}.ppf((np.arange(1, len(x) + 1) - 0.3) / (len(x) + 0.4), *params)\npx.scatter(x=q, y=x).show()\n"
    )


# ============================================================================ correlation heat map
_CORRELATIONS = (("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall"))


def _correlation_test(method: str, x: pd.Series, y: pd.Series):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method](x, y)


def _heatmap(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, method = roles["vars"], p["method"]
    data = frame[names].apply(pd.to_numeric, errors="coerce")
    matrix = data.corr(method=method)
    if matrix.isna().all().all():
        raise InvalidInputError(
            "No se pudo calcular ninguna correlación: revise que las columnas varíen y tengan datos."
        )
    filled = matrix.fillna(0.0)
    if p["cluster"] and len(names) > 2:
        order = hierarchy.leaves_list(
            hierarchy.linkage(1 - np.abs(filled.to_numpy())[np.triu_indices(len(names), 1)], method="average")
        )
        names = [names[i] for i in order]
        matrix = matrix.loc[names, names]
    out = Output(f"Mapa de calor de correlaciones ({dict(_CORRELATIONS)[method]})", f"{len(names)} variables")
    out.tiles([("Variables", len(names)), ("Filas con algún dato", int(data.notna().any(axis=1).sum()))])
    out.chart(
        heatmap_chart(
            f"Correlaciones de {dict(_CORRELATIONS)[method]}",
            matrix.to_numpy(),
            names,
            names,
            annotate=len(names) <= 14,
            midpoint=0.0,
            height=max(360, 55 * len(names) + 160),
        )
    )
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair = data[[a, b]].dropna()
            if len(pair) < 3:
                continue
            test = _correlation_test(method, pair[a], pair[b])
            if test is None:
                continue
            pairs.append(
                [
                    a,
                    b,
                    len(pair),
                    float(test.statistic if hasattr(test, "statistic") else test[0]),
                    P(float(test.pvalue if hasattr(test, "pvalue") else test[1])),
                ]
            )
    pairs.sort(key=lambda row: -abs(row[3]))
    out.table(
        "Pares con mayor correlación",
        ["Variable 1", "Variable 2", "n", "Coeficiente", "Valor p"],
        [[a, b, n, bad(r) if abs(r) >= 0.7 else r, pv] for a, b, n, r, pv in pairs[:15]],
        note="En rojo: |coeficiente| ≥ 0.7.",
    )
    return out.set(strongest=pairs[0][:2] + [pairs[0][3]] if pairs else None).build()


def _heatmap_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.imshow(df[{roles['vars']!r}].corr(method={p['method']!r}), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', text_auto='.2f').show()\n"
    )


# ============================================================================ contour / surface
_SURFACES = (("contour", "Contornos (2D)"), ("surface", "Superficie (3D)"))
_INTERP = (("linear", "Lineal"), ("cubic", "Cúbica (suave)"), ("nearest", "Vecino más cercano"))


def _contour(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    xn, yn, zn = roles["x"], roles["y"], roles.get("z")
    data = pd.DataFrame({"x": numbers(frame, xn), "y": numbers(frame, yn)})
    if zn:
        data["z"] = numbers(frame, zn)
    clean = data.dropna()
    need(clean["x"], "los puntos", 10)
    n = int(p["grid"])
    gx, gy = np.linspace(clean["x"].min(), clean["x"].max(), n), np.linspace(clean["y"].min(), clean["y"].max(), n)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    if zn:
        z = griddata((clean["x"], clean["y"]), clean["z"], (mesh_x, mesh_y), method=p["interpolation"])
        label = zn
    else:
        density = stats.gaussian_kde(np.vstack([clean["x"], clean["y"]]))
        z = density(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
        label = "Densidad"
    if not np.isfinite(z).any():
        raise InvalidInputError("No se pudo interpolar: los puntos están alineados o son muy pocos.")
    values = [[None if not np.isfinite(v) else float(v) for v in row] for row in z]
    kind = p["kind"]
    trace = (
        {
            "type": "contour",
            "x": gx,
            "y": gy,
            "z": values,
            "colorscale": "Viridis",
            "colorbar": {"title": {"text": label}},
            "contours": {"showlabels": True},
        }
        if kind == "contour"
        else {
            "type": "surface",
            "x": gx,
            "y": gy,
            "z": values,
            "colorscale": "Viridis",
            "colorbar": {"title": {"text": label}},
        }
    )
    layout = (
        {"xaxis": {"title": {"text": xn}}, "yaxis": {"title": {"text": yn}}}
        if kind == "contour"
        else {
            "scene": {
                "xaxis": {"title": {"text": xn}},
                "yaxis": {"title": {"text": yn}},
                "zaxis": {"title": {"text": label}},
            }
        }
    )
    out = Output(
        f"{'Superficie' if kind == 'surface' else 'Contornos'} de {label}",
        f"según {xn} y {yn}" + ("" if zn else " (densidad conjunta)"),
    )
    out.tiles(
        [
            ("Puntos", len(clean)),
            ("Celdas de la rejilla", n * n),
            ("Mínimo", float(np.nanmin(z))),
            ("Máximo", float(np.nanmax(z))),
        ]
    )
    if zn and p["interpolation"] != "nearest":
        out.alert("info", "Las celdas fuera de la nube de puntos quedan en blanco: no se extrapola.")
    out.chart(raw_chart([trace], layout, f"{label} según {xn} y {yn}", height=520))
    return out.set(n=len(clean)).build()


def _contour_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    x, y, z = py_name(roles["x"]), py_name(roles["y"]), roles.get("z")
    if z:
        return (
            "import numpy as np\nimport plotly.graph_objects as go\nfrom scipy.interpolate import griddata\n\n"
            + load_line(dataset)
            + f"d = df[[{x}, {y}, {py_name(z)}]].dropna()\ngx = np.linspace(d[{x}].min(), d[{x}].max(), {int(p['grid'])})\ngy = np.linspace(d[{y}].min(), d[{y}].max(), {int(p['grid'])})\n"
            + f"zz = griddata((d[{x}], d[{y}]), d[{py_name(z)}], np.meshgrid(gx, gy), method={p['interpolation']!r})\ngo.Figure(go.{'Contour' if p['kind'] == 'contour' else 'Surface'}(x=gx, y=gy, z=zz)).show()\n"
        )
    return "import plotly.express as px\n\n" + load_line(dataset) + f"px.density_contour(df, x={x}, y={y}).show()\n"


# ============================================================================ 3-D scatter
def _scatter3d(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    xn, yn, zn, by = roles["x"], roles["y"], roles["z"], roles.get("group")
    data = pd.DataFrame({"x": numbers(frame, xn), "y": numbers(frame, yn), "z": numbers(frame, zn)})
    if by:
        data["g"] = frame[by]
    clean = data.dropna()
    need(clean["x"], "las filas completas", 4)
    if len(clean) > 3000:
        clean = clean.sample(3000, random_state=1)
    parts = (
        [(str(level), clean[clean["g"] == level]) for level in level_order(clean["g"])] if by else [("Datos", clean)]
    )
    traces = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": label,
            "x": part["x"].to_numpy(),
            "y": part["y"].to_numpy(),
            "z": part["z"].to_numpy(),
            "marker": {"size": 3.5, "opacity": 0.8},
        }
        for label, part in parts
    ]
    out = Output(f"Dispersión 3D: {xn}, {yn}, {zn}", f"por {by}" if by else None)
    dropped_note(out, len(data), len(data.dropna()))
    out.tiles([("Puntos", len(clean)), ("Grupos", len(parts))])
    layout = {
        "scene": {"xaxis": {"title": {"text": xn}}, "yaxis": {"title": {"text": yn}}, "zaxis": {"title": {"text": zn}}},
        "showlegend": bool(by),
    }
    out.chart(
        raw_chart(traces, layout, "Dispersión tridimensional", height=540),
        "Arrastre para girar; la rueda acerca y aleja.",
    )
    return out.set(n=len(clean)).build()


def _scatter3d_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    color = f", color={py_name(roles['group'])}" if roles.get("group") else ""
    return (
        "import plotly.express as px\n\n"
        + load_line(dataset)
        + f"px.scatter_3d(df, x={py_name(roles['x'])}, y={py_name(roles['y'])}, z={py_name(roles['z'])}{color}).show()\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, "graficos", description, compute, **kw)


_LAYOUT = FieldSpec(
    "layout",
    "Varios grupos",
    "select",
    "overlay",
    (("overlay", "En un solo gráfico"), ("panels", "Un panel por grupo")),
)
_GROUP = RoleSpec("group", "Grupos (opcional)", "categorical", optional=True)

SPECS = (
    _spec(
        "histogram",
        "Histograma",
        AREA,
        "Distribución de una variable con curva normal o de densidad, por grupos superpuestos o en paneles.",
        _histogram,
        roles=(RoleSpec("var", "Variable", "numeric"), _GROUP),
        params=(
            FieldSpec(
                "bins", "Número de intervalos (vacío = automático)", "integer", optional=True, minimum=2, maximum=200
            ),
            FieldSpec(
                "norm",
                "Eje vertical",
                "select",
                "density",
                (("density", "Densidad"), ("count", "Frecuencia"), ("percent", "Porcentaje")),
            ),
            FieldSpec(
                "curve",
                "Curva",
                "select",
                "normal",
                (("none", "Ninguna"), ("normal", "Normal ajustada"), ("kde", "Densidad (KDE)")),
            ),
            _LAYOUT,
        ),
        code=_histogram_code,
        example={"roles": {"var": "x1", "group": "g"}, "params": {"curve": "normal"}},
    ),
    _spec(
        "boxplot",
        "Diagrama de caja, violín y valores individuales",
        AREA,
        "Compara la distribución de una o varias variables (o de una variable por grupos) con cajas, violines o puntos.",
        _boxplot,
        roles=(
            RoleSpec("vars", "Variable(s)", "numeric", multiple=True),
            RoleSpec("group", "Grupos (opcional, con una sola variable)", "categorical", optional=True),
        ),
        params=(FieldSpec("style", "Estilo", "select", "box", _BOX_STYLES),),
        code=_boxplot_code,
        example={"roles": {"vars": ["y"], "group": "g"}, "params": {"style": "violin"}},
    ),
    _spec(
        "probability",
        "Gráfico de probabilidad (Q-Q)",
        AREA,
        "Compara los datos con los cuantiles de una distribución ajustada (normal, lognormal, Weibull…) y mide la bondad de ajuste.",
        _probability,
        roles=(RoleSpec("var", "Variable", "numeric"),),
        params=(FieldSpec("distribution", "Distribución", "select", "norm", _DISTRIBUTIONS),),
        code=_probability_code,
        example={"roles": {"var": "y"}, "params": {"distribution": "norm"}},
    ),
    _spec(
        "scatter",
        "Dispersión y burbujas (con ajuste)",
        "Relaciones",
        "Gráfico X–Y con grupos, tamaño de burbuja, recta, polinomio o suavizado LOWESS; correlaciones de Pearson y Spearman.",
        _scatter,
        roles=(
            RoleSpec("x", "Eje X", "numeric"),
            RoleSpec("y", "Eje Y", "numeric"),
            _GROUP,
            RoleSpec("size", "Tamaño de burbuja (opcional)", "numeric", optional=True),
        ),
        params=(
            FieldSpec("fit", "Ajuste", "select", "linear", _FITS),
            FieldSpec(
                "smooth",
                "Suavizado LOWESS (fracción de datos)",
                "number",
                0.6,
                minimum=0.1,
                maximum=1,
                visible_if=("fit", ("lowess",)),
            ),
            FieldSpec(
                "ci",
                "Banda de confianza 95 % (recta con un solo grupo)",
                "boolean",
                True,
                visible_if=("fit", ("linear",)),
            ),
            _LAYOUT,
        ),
        code=_scatter_code,
        example={"roles": {"x": "x1", "y": "y", "group": "g"}, "params": {"fit": "linear"}},
    ),
    _spec(
        "splom",
        "Matriz de dispersión",
        "Relaciones",
        "Todos los pares de variables a la vez, con la matriz de correlaciones.",
        _splom,
        roles=(RoleSpec("vars", "Variables (2 a 8)", "numeric", multiple=True, min_count=2, max_count=8), _GROUP),
        code=_splom_code,
        example={"roles": {"vars": ["y", "x1", "x2"], "group": "g"}, "params": {}},
    ),
    _spec(
        "heatmap",
        "Mapa de calor de correlaciones",
        "Relaciones",
        "Matriz de correlaciones (Pearson, Spearman o Kendall) en colores, con orden por agrupamiento jerárquico opcional.",
        _heatmap,
        roles=(RoleSpec("vars", "Variables (2 o más)", "numeric", multiple=True, min_count=2),),
        params=(
            FieldSpec("method", "Coeficiente", "select", "pearson", _CORRELATIONS),
            FieldSpec("cluster", "Ordenar por similitud (clúster jerárquico)", "boolean", True),
        ),
        code=_heatmap_code,
        example={"roles": {"vars": ["y", "x1", "x2", "x3"]}, "params": {"method": "spearman"}},
    ),
    _spec(
        "contour",
        "Contornos y superficie",
        "Relaciones",
        "Curvas de nivel o superficie 3D de Z según X e Y (interpolada), o densidad conjunta de X e Y si no hay Z.",
        _contour,
        roles=(
            RoleSpec("x", "Eje X", "numeric"),
            RoleSpec("y", "Eje Y", "numeric"),
            RoleSpec("z", "Valor Z (vacío = densidad de los puntos)", "numeric", optional=True),
        ),
        params=(
            FieldSpec("kind", "Tipo", "select", "contour", _SURFACES),
            FieldSpec("interpolation", "Interpolación", "select", "linear", _INTERP),
            FieldSpec("grid", "Tamaño de la rejilla", "integer", 40, minimum=10, maximum=120),
        ),
        code=_contour_code,
        example={"roles": {"x": "x1", "y": "x2", "z": "y"}, "params": {"kind": "contour"}},
    ),
    _spec(
        "scatter3d",
        "Dispersión 3D",
        "Relaciones",
        "Nube de puntos tridimensional que se puede girar, con grupos en colores.",
        _scatter3d,
        roles=(
            RoleSpec("x", "Eje X", "numeric"),
            RoleSpec("y", "Eje Y", "numeric"),
            RoleSpec("z", "Eje Z", "numeric"),
            _GROUP,
        ),
        code=_scatter3d_code,
        example={"roles": {"x": "x1", "y": "x2", "z": "y", "group": "g"}, "params": {}},
    ),
    _spec(
        "line_plot",
        "Gráfico de líneas",
        "Series y perfiles",
        "Una o varias series sobre un eje (fecha, número u orden), con índice 100, tipificación y media móvil opcionales.",
        _line_plot,
        roles=(
            RoleSpec("ys", "Variable(s) Y", "numeric", multiple=True, max_count=8),
            RoleSpec(
                "x",
                "Eje X (opcional)",
                "any",
                optional=True,
                help="Fecha o número; sin él se usa el orden de las filas.",
            ),
            RoleSpec("group", "Una línea por grupo (opcional, con una sola Y)", "categorical", optional=True),
        ),
        params=(
            FieldSpec("scale", "Escala", "select", "none", _SCALES),
            FieldSpec("sort", "Ordenar por X", "boolean", True),
            FieldSpec("markers", "Marcar los puntos", "boolean", False),
            FieldSpec("smooth", "Media móvil (0 = no)", "integer", 0, minimum=0, maximum=500),
        ),
        code=_line_code,
        example={"roles": {"ys": ["s", "y"], "x": "t"}, "params": {"scale": "zscore", "smooth": 6}},
    ),
    _spec(
        "bar",
        "Gráfico de barras",
        "Categorías",
        "Cantidades, porcentajes o estadísticos (media, mediana, suma…) por categoría, con grupos y barras de error.",
        _bar,
        roles=(
            RoleSpec("category", "Categoría", "categorical"),
            RoleSpec("value", "Valor numérico (según la medida)", "numeric", optional=True),
            RoleSpec("group", "Subgrupos (opcional)", "categorical", optional=True),
        ),
        params=(
            FieldSpec("stat", "Medida", "select", "count", _STATS),
            FieldSpec("error", "Barras de error (solo medias)", "select", "none", _ERRORS),
            FieldSpec(
                "sort",
                "Orden de las categorías",
                "select",
                "none",
                (("none", "El natural"), ("desc", "De mayor a menor"), ("asc", "De menor a mayor")),
            ),
            FieldSpec("orientation", "Orientación", "select", "v", (("v", "Verticales"), ("h", "Horizontales"))),
            FieldSpec("stacked", "Apiladas (con subgrupos)", "boolean", False),
            FieldSpec("top", "Máximo de categorías", "integer", 30, minimum=2, maximum=100),
        ),
        code=_bar_code,
        example={"roles": {"category": "g", "value": "y", "group": "h"}, "params": {"stat": "mean", "error": "ci"}},
    ),
    _spec(
        "pareto",
        "Diagrama de Pareto",
        "Categorías",
        "Categorías ordenadas por importancia con el porcentaje acumulado: encuentra los «pocos vitales».",
        _pareto,
        roles=(
            RoleSpec("category", "Categoría", "categorical"),
            RoleSpec("value", "Valor (opcional; vacío = contar filas)", "numeric", optional=True),
        ),
        params=(
            FieldSpec("top", "Categorías a mostrar", "integer", 15, minimum=2, maximum=60),
            FieldSpec("other", "Agrupar el resto en «Otros»", "boolean", True),
        ),
        code=_pareto_code,
        example={"roles": {"category": "g", "value": "count"}, "params": {}},
    ),
    _spec(
        "pie",
        "Gráfico circular",
        "Categorías",
        "Reparto de un total entre categorías, con opción de anillo.",
        _pie,
        roles=(
            RoleSpec("category", "Categoría", "categorical"),
            RoleSpec("value", "Valor (opcional; vacío = contar filas)", "numeric", optional=True),
        ),
        params=(
            FieldSpec("top", "Categorías a mostrar", "integer", 10, minimum=2, maximum=40),
            FieldSpec("donut", "Anillo", "boolean", False),
        ),
        code=_pie_code,
        example={"roles": {"category": "g"}, "params": {"donut": True}},
    ),
    _spec(
        "interval",
        "Gráfico de intervalos de confianza",
        "Categorías",
        "Media e intervalo de confianza de una variable en cada grupo, para ver de un vistazo qué grupos se diferencian.",
        _interval,
        roles=(RoleSpec("var", "Variable", "numeric"), RoleSpec("group", "Grupos", "categorical")),
        params=(
            FieldSpec("confidence", "Nivel de confianza (%)", "number", 95, minimum=50, maximum=99.9),
            FieldSpec("points", "Mostrar los datos individuales", "boolean", True),
        ),
        code=_interval_code,
        example={"roles": {"var": "y", "group": "g"}, "params": {}},
    ),
)

registry.register(*SPECS)
