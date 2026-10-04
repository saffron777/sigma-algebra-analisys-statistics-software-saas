"""Exploring a series: plots and moving statistics, cross-correlation, spectrum, decomposition, trend, filters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal, stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import interval_chart, line_chart, panels_chart, scatter_chart, stem_chart
from sigma.domain.analysis.output import Output, P, bad
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.analysis.ts_core import (
    ACCURACY_COLUMNS,
    TS,
    accuracy,
    accuracy_row,
    forecast_rows,
    future_labels,
    history_and_forecast,
    load_series,
    need_cycles,
    note_trimmed,
)
from sigma.domain.errors import InvalidInputError

MODULE = "series"
GROUP_EXPLORE = "Exploración"
GROUP_COMPONENTS = "Componentes, tendencia y filtros"
SERIES = RoleSpec("series", "Serie", "numeric")
TIME = RoleSpec(
    "time", "Columna de tiempo (opcional)", "datetime", optional=True, help="Sin ella se usa el orden de las filas."
)
PERIOD = FieldSpec(
    "period",
    "Periodo estacional (0 = según la fecha)",
    "integer",
    0,
    minimum=0,
    maximum=400,
    help="Observaciones por ciclo: 12 mensual, 4 trimestral, 7 diario, 24 por hora.",
)
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)


def _series_code(roles: dict[str, Any], dataset: str, extra: str = "") -> str:
    time = roles.get("time")
    sort = f"df = df.sort_values({py_name(time)})\n" if time else ""
    return load_line(dataset) + sort + f"y = df[{py_name(roles['series'])}].dropna().to_numpy(dtype=float)\n" + extra


# ============================================================================ plot and moving statistics
def _ts_plot(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"])
    n = ts.n
    window = int(p["window"]) if p["window"] else max(3, n // 10)
    window = max(2, min(window, n // 2))
    s = pd.Series(ts.y)
    mean, sd = (
        s.rolling(window, center=True, min_periods=window).mean(),
        s.rolling(window, center=True, min_periods=window).std(),
    )
    t = np.arange(1, n + 1)
    trend = sm.OLS(ts.y, sm.add_constant(t)).fit()
    lag1 = float(np.corrcoef(ts.y[1:], ts.y[:-1])[0, 1])
    out = Output(
        f"Serie {ts.name}",
        f"n = {n} · ventana móvil de {window}" + (f" · periodo {ts.period}" if ts.period > 1 else ""),
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Observaciones", n),
            ("Media", float(ts.y.mean())),
            ("Desv. estándar", float(ts.y.std(ddof=1))),
            ("Tendencia por periodo", float(trend.params[1])),
            ("Autocorrelación lag 1", lag1),
        ]
    )
    valid = mean.notna().to_numpy()
    x = np.asarray(ts.labels, dtype=object)
    out.chart(
        line_chart(
            "Serie con media y desviación móviles",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            [
                {"name": "Serie", "x": x, "y": ts.y},
                {"name": f"Media móvil ({window})", "x": x[valid], "y": mean.to_numpy()[valid], "dash": "dash"},
            ],
            bands=[
                {
                    "name": "± 2 desv. est. móvil",
                    "x": x[valid],
                    "lower": (mean - 2 * sd).to_numpy()[valid],
                    "upper": (mean + 2 * sd).to_numpy()[valid],
                }
            ],
        )
    )
    out.table(
        "Tendencia lineal",
        ["Pendiente por periodo", "Error est.", "Valor p", "R²"],
        [[float(trend.params[1]), float(trend.bse[1]), P(float(trend.pvalues[1])), float(trend.rsquared)]],
        note="Regresión de la serie sobre el número de observación.",
    )
    if ts.period > 1 and n >= 2 * ts.period:
        position = np.arange(n) % ts.period
        groups = []
        for k in range(ts.period):
            values = ts.y[position == k]
            half = (
                stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            groups.append(
                {
                    "name": str(k + 1),
                    "mean": float(values.mean()),
                    "lower": float(values.mean() - half),
                    "upper": float(values.mean() + half),
                }
            )
        out.chart(
            interval_chart(
                f"Promedio por posición dentro del ciclo (periodo {ts.period})", "Posición en el ciclo", ts.name, groups
            )
        )
    out.chart(
        scatter_chart(
            f"Gráfico de rezago 1 (r = {lag1:.3f})",
            f"{ts.name}(t − 1)",
            f"{ts.name}(t)",
            [{"name": "Pares", "x": ts.y[:-1], "y": ts.y[1:]}],
        )
    )
    return out.set(n=n, trend_slope=float(trend.params[1]), trend_p=float(trend.pvalues[1]), lag1=lag1).build()


def _ts_plot_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "import pandas as pd\nimport plotly.express as px\n\n" + _series_code(
        roles,
        dataset,
        f"s = pd.Series(y)\npx.line(pd.DataFrame({{'serie': s, 'media_movil': s.rolling({int(p['window'] or 12)}, center=True).mean()}})).show()\n",
    )


# ============================================================================ cross-correlation
def cross_correlations(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """``r[k]`` for k = −max_lag…max_lag: the correlation of x(t) with y(t + k) (positive k: x leads y)."""
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = n * x.std() * y.std()
    lags = np.arange(-max_lag, max_lag + 1)
    values = []
    for k in lags:
        if k >= 0:
            values.append(float((xc[: n - k] * yc[k:]).sum() / denominator))
        else:
            values.append(float((xc[-k:] * yc[: n + k]).sum() / denominator))
    return lags, np.array(values)


def _ccf(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    xn, yn = roles["x"], roles["y"]
    data = pd.DataFrame(
        {"x": pd.to_numeric(frame[xn], errors="coerce"), "y": pd.to_numeric(frame[yn], errors="coerce")}
    )
    if roles.get("time"):
        data["t"] = frame[roles["time"]]
        data = data.sort_values("t", kind="stable")
    data = data.dropna(subset=["x", "y"])
    x, y = data["x"].to_numpy(float), data["y"].to_numpy(float)
    if p["difference"]:
        x, y = np.diff(x), np.diff(y)
    n = len(x)
    if n < 15:
        raise InvalidInputError("Se necesitan al menos 15 observaciones pareadas.")
    if x.std() == 0 or y.std() == 0:
        raise InvalidInputError("Una de las series es constante.")
    max_lag = max(1, min(int(p["max_lag"]), n // 3))
    lags, r = cross_correlations(x, y, max_lag)
    band = float(stats.norm.ppf(1 - p["alpha"] / 2) / np.sqrt(n))
    strongest = int(lags[np.argmax(np.abs(r))])
    out = Output(
        f"Correlación cruzada: {xn} y {yn}",
        f"{n} pares"
        + (" · en primeras diferencias" if p["difference"] else "")
        + f" · bandas al {100 * (1 - p['alpha']):g} %",
    )
    out.tiles(
        [
            ("Pares", n),
            ("Mayor |correlación|", float(r[lags == strongest][0])),
            ("Rezago de la mayor", strongest),
            ("Rezagos significativos", int((np.abs(r) > band).sum())),
        ]
    )
    meaning = (
        "al mismo tiempo"
        if strongest == 0
        else f"{xn} {'se adelanta a' if strongest > 0 else 'va detrás de'} {yn} por {abs(strongest)} periodo(s)"
    )
    out.alert("info", f"La asociación más fuerte ocurre {meaning}.")
    out.chart(
        stem_chart(
            f"Correlación cruzada (positivo: {xn} adelanta a {yn})", "Rezago k", "r(x[t], y[t + k])", lags, r, band=band
        )
    )
    out.table(
        "Rezagos fuera de las bandas",
        ["Rezago", "Correlación"],
        [[int(k), bad(float(v))] for k, v in zip(lags, r, strict=True) if abs(v) > band][:30],
        note="Las series con tendencia o estacionalidad dan correlaciones cruzadas engañosas: pruebe con primeras diferencias.",
    )
    return out.set(n=n, strongest_lag=strongest, strongest_r=float(r[lags == strongest][0])).build()


def _ccf_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.tsa.stattools import ccf\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['x'])}, {py_name(roles['y'])}]].dropna()\n"
        + ("d = d.diff().dropna()\n" if p["difference"] else "")
        + f"print(ccf(d[{py_name(roles['y'])}], d[{py_name(roles['x'])}], adjusted=False)[:{int(p['max_lag']) + 1}])  # x adelanta a y\n"
    )


# ============================================================================ periodogram
def fisher_g_test(power: np.ndarray) -> tuple[float, float]:
    """Fisher's test for the largest periodogram ordinate: (g, p-value)."""
    m = len(power)
    g = float(power.max() / power.sum())
    terms = int(np.floor(1 / g))
    pvalue = sum((-1) ** (j - 1) * _choose(m, j) * (1 - j * g) ** (m - 1) for j in range(1, terms + 1))
    return g, float(min(max(pvalue, 0.0), 1.0))


def _choose(m: int, j: int) -> float:
    from math import comb

    return float(comb(m, j))


def _periodogram(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, minimum=16)
    detrend = {"none": "constant", "linear": "linear"}[p["detrend"]]
    freq, power = signal.periodogram(ts.y, fs=1.0, detrend=detrend, window="boxcar", scaling="spectrum")
    freq, power = freq[1:], power[1:]
    if (power <= 0).all():
        raise InvalidInputError("La serie no tiene variación después de quitar la tendencia.")
    peaks, _ = signal.find_peaks(power)
    order = peaks[np.argsort(power[peaks])[::-1]][: int(p["peaks"])]
    g, pvalue = fisher_g_test(power)
    total = float(power.sum())
    out = Output(
        f"Periodograma de {ts.name}",
        f"n = {ts.n} · {'sin' if p['detrend'] == 'none' else 'con'} tendencia lineal quitada",
    )
    note_trimmed(out, ts)
    best = order[0] if len(order) else int(np.argmax(power))
    out.tiles(
        [
            ("Periodo dominante", float(1 / freq[best])),
            ("Frecuencia dominante", float(freq[best])),
            ("% de la potencia", 100 * float(power[best] / total)),
            ("Valor p (Fisher)", P(pvalue)),
        ]
    )
    significant = pvalue < p["alpha"]
    out.alert(
        "success" if significant else "info",
        (
            f"Hay una periodicidad significativa de unos {1 / freq[best]:.3g} periodos (prueba de Fisher p = {pvalue:.4f})."
            if significant
            else f"No hay evidencia de una periodicidad dominante (prueba de Fisher p = {pvalue:.4f})."
        ),
    )
    out.chart(
        line_chart(
            "Periodograma",
            "Frecuencia (ciclos por observación)",
            "Potencia",
            [{"name": "Potencia", "x": freq, "y": power}],
            vlines=[{"value": float(freq[best]), "label": f"periodo {1 / freq[best]:.3g}"}],
        )
    )
    out.table(
        "Picos más altos",
        ["#", "Frecuencia", "Periodo", "Potencia", "% del total"],
        [
            [i + 1, float(freq[k]), float(1 / freq[k]), float(power[k]), 100 * float(power[k] / total)]
            for i, k in enumerate(order)
        ],
        note="Periodo = 1 / frecuencia, en unidades de observación.",
    )
    return out.set(dominant_period=float(1 / freq[best]), fisher_p=pvalue, fisher_g=g).build()


def _periodogram_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "from scipy import signal\n\n" + _series_code(
        roles,
        dataset,
        f"f, power = signal.periodogram(y, detrend={'linear' if p['detrend'] == 'linear' else 'constant'!r}, scaling='spectrum')\nprint(1 / f[1:][power[1:].argmax()])\n",
    )


# ============================================================================ decomposition
_DECOMPOSITIONS = (
    ("classical_add", "Clásica aditiva (y = T + S + R)"),
    ("classical_mul", "Clásica multiplicativa (y = T × S × R)"),
    ("stl", "STL (LOESS, aditiva)"),
    ("stl_robust", "STL robusta a atípicos"),
)


def _strength(remainder: np.ndarray, other: np.ndarray) -> float:
    mask = np.isfinite(remainder) & np.isfinite(other)
    combined = remainder[mask] + other[mask]
    return float(max(0.0, 1 - np.var(remainder[mask]) / np.var(combined))) if np.var(combined) > 0 else 0.0


def _components_frame(ts: TS, observed, trend, seasonal, remainder) -> pd.DataFrame:
    stamps = pd.Series(pd.DatetimeIndex(ts.index)) if ts.dated else pd.Series(np.arange(1, ts.n + 1))
    return pd.DataFrame(
        {"tiempo": stamps, ts.name: observed, "tendencia": trend, "estacional": seasonal, "residuo": remainder}
    )


def _decompose(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"])
    need_cycles(ts)
    method = p["method"]
    series = ts.series()
    multiplicative = method == "classical_mul"
    if multiplicative and (ts.y <= 0).any():
        raise InvalidInputError("La descomposición multiplicativa necesita valores estrictamente positivos.")
    if method.startswith("classical"):
        result = seasonal_decompose(series, model="multiplicative" if multiplicative else "additive", period=ts.period)
        trend, seasonal, remainder = (
            np.asarray(result.trend, float),
            np.asarray(result.seasonal, float),
            np.asarray(result.resid, float),
        )
    else:
        result = STL(series, period=ts.period, robust=method == "stl_robust").fit()
        trend, seasonal, remainder = (
            np.asarray(result.trend, float),
            np.asarray(result.seasonal, float),
            np.asarray(result.resid, float),
        )
    adjusted = ts.y / seasonal if multiplicative else ts.y - seasonal
    if multiplicative:
        strength_t = _strength(
            np.log(np.where(np.isfinite(remainder) & (remainder > 0), remainder, np.nan)),
            np.log(np.where(np.isfinite(trend) & (trend > 0), trend, np.nan)),
        )
        strength_s = _strength(
            np.log(np.where(np.isfinite(remainder) & (remainder > 0), remainder, np.nan)),
            np.log(np.where(seasonal > 0, seasonal, np.nan)),
        )
    else:
        strength_t, strength_s = _strength(remainder, trend), _strength(remainder, seasonal)
    out = Output(f"Descomposición de {ts.name}", f"{dict(_DECOMPOSITIONS)[method]} · periodo {ts.period} · n = {ts.n}")
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Periodo", ts.period),
            ("Fuerza de la tendencia", strength_t),
            ("Fuerza de la estacionalidad", strength_s),
            ("Desv. est. del residuo", float(np.nanstd(remainder))),
        ]
    )
    out.alert(
        "info",
        "Fuerza = 1 − Var(residuo) / Var(componente + residuo): cerca de 1 el componente domina, cerca de 0 casi no existe (Hyndman y Athanasopoulos).",
    )
    x = np.asarray(ts.labels, dtype=object)
    panels = []
    for title, values in (("Observado", ts.y), ("Tendencia", trend), ("Estacional", seasonal), ("Residuo", remainder)):
        ok = np.isfinite(values)
        panels.append(
            {
                **line_chart(title, None, None, [{"name": title, "x": x[ok], "y": values[ok]}], height=220),
                "title": title,
            }
        )
    out.chart(panels_chart(f"Componentes de {ts.name}", panels, columns=1, height=760))
    cycle = (
        seasonal[: ts.period]
        if not ts.dated
        else pd.Series(seasonal).groupby(np.arange(ts.n) % ts.period).mean().to_numpy()
    )
    out.table(
        "Índices estacionales (posición en el ciclo)",
        ["Posición", "Índice estacional" if multiplicative else "Efecto estacional"],
        [[k + 1, float(v)] for k, v in enumerate(cycle)],
        note="Multiplicativo: índice 1.05 = 5 % por encima de lo normal."
        if multiplicative
        else "Aditivo: efecto en las unidades de la serie.",
    )
    out.dataset(
        "componentes",
        "Componentes de la serie",
        _components_frame(ts, ts.y, trend, seasonal, remainder).assign(ajustada_estacionalmente=adjusted),
    )
    return out.set(period=ts.period, trend_strength=strength_t, seasonal_strength=strength_s).build()


def _decompose_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if p["method"].startswith("stl"):
        return "from statsmodels.tsa.seasonal import STL\n\n" + _series_code(
            roles,
            dataset,
            f"result = STL(y, period={int(p['period'] or 12)}, robust={p['method'] == 'stl_robust'}).fit()\nprint(result.trend, result.seasonal, result.resid)\n",
        )
    return "from statsmodels.tsa.seasonal import seasonal_decompose\n\n" + _series_code(
        roles,
        dataset,
        f"result = seasonal_decompose(y, model={'multiplicative' if p['method'] == 'classical_mul' else 'additive'!r}, period={int(p['period'] or 12)})\n",
    )


# ============================================================================ trend analysis
_TRENDS = (
    ("linear", "Lineal: y = a + b·t"),
    ("quadratic", "Cuadrático: y = a + b·t + c·t²"),
    ("exponential", "Crecimiento exponencial: y = a·e^(b·t)"),
)


def _trend(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"])
    n, h, alpha, kind = ts.n, int(p["horizon"]), p["alpha"], p["model"]
    t = np.arange(1, n + 1, dtype=float)
    if kind == "exponential":
        if (ts.y <= 0).any():
            raise InvalidInputError("El modelo exponencial necesita valores estrictamente positivos.")
        target = np.log(ts.y)
    else:
        target = ts.y

    def design(values: np.ndarray) -> np.ndarray:
        columns = [np.ones_like(values), values] + ([values**2] if kind == "quadratic" else [])
        return np.column_stack(columns)

    fit = sm.OLS(target, design(t)).fit()
    fitted = fit.fittedvalues if kind != "exponential" else np.exp(fit.fittedvalues)
    future_t = np.arange(n + 1, n + h + 1, dtype=float)
    prediction = fit.get_prediction(design(future_t)).summary_frame(alpha=alpha) if h else None
    out = Output(f"Análisis de tendencia de {ts.name}", f"modelo {dict(_TRENDS)[kind].split(':')[0].lower()} · n = {n}")
    note_trimmed(out, ts)
    metrics = accuracy(ts.y, np.asarray(fitted), ts.y, ts.period)
    out.tiles(
        [
            ("R²", float(fit.rsquared)),
            ("MAPE %", metrics["mape"]),
            ("Durbin-Watson", float(durbin_watson(fit.resid))),
            ("Observaciones", n),
        ]
    )
    names = ["a (constante)", "b (lineal)"] + (["c (cuadrático)"] if kind == "quadratic" else [])
    rows = [
        [names[i], float(fit.params[i]), float(fit.bse[i]), float(fit.tvalues[i]), P(float(fit.pvalues[i]))]
        for i in range(len(fit.params))
    ]
    out.table(
        "Coeficientes" + (" (en escala logarítmica)" if kind == "exponential" else ""),
        ["Parámetro", "Estimación", "Error est.", "t", "Valor p"],
        rows,
        note=(
            f"Crecimiento: {100 * (np.exp(fit.params[1]) - 1):.3g} % por periodo." if kind == "exponential" else None
        ),
    )
    out.table("Exactitud del ajuste", ["Medida", *ACCURACY_COLUMNS[:-1]], [accuracy_row("En la muestra", metrics)[:4]])
    if abs(float(durbin_watson(fit.resid)) - 2) > 0.7:
        out.alert(
            "warning",
            "Durbin-Watson lejos de 2: los residuos están autocorrelacionados; los intervalos de predicción de una tendencia simple serán demasiado estrechos. Considere ARIMA o suavizamiento.",
        )
    if h and prediction is not None:
        mean, lower, upper = (
            np.asarray(prediction["mean"]),
            np.asarray(prediction["obs_ci_lower"]),
            np.asarray(prediction["obs_ci_upper"]),
        )
        if kind == "exponential":
            mean, lower, upper = np.exp(mean), np.exp(lower), np.exp(upper)
        labels = future_labels(ts, h)
        out.table(
            "Pronósticos",
            ["Periodo", "Pronóstico", f"PI {100 * (1 - alpha):g} % inf.", "PI sup."],
            forecast_rows(labels, mean, lower, upper),
        )
        out.chart(history_and_forecast(ts, np.asarray(fitted), mean, lower, upper, alpha))
    else:
        out.chart(
            line_chart(
                "Serie y tendencia",
                "Tiempo" if ts.dated else "Observación",
                ts.name,
                [
                    {"name": "Datos", "x": ts.labels, "y": ts.y},
                    {"name": "Tendencia", "x": ts.labels, "y": np.asarray(fitted), "dash": "dash"},
                ],
            )
        )
    return out.set(
        n=n, r2=float(fit.rsquared), mape=metrics["mape"], coefficients=[float(v) for v in fit.params]
    ).build()


def _trend_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    degree = 2 if p["model"] == "quadratic" else 1
    target = "np.log(y)" if p["model"] == "exponential" else "y"
    return "import numpy as np\n\n" + _series_code(
        roles,
        dataset,
        f"t = np.arange(1, len(y) + 1)\ncoefficients = np.polyfit(t, {target}, {degree})\nprint(coefficients)\n",
    )


# ============================================================================ Mann-Kendall
def mann_kendall(values: np.ndarray, groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """Mann-Kendall S, its variance (ties corrected) and z; with ``groups`` the statistics of each season are summed."""
    total_s, total_var = 0.0, 0.0
    parts = [values] if groups is None else [values[groups == g] for g in np.unique(groups)]
    for part in parts:
        n = len(part)
        s = (
            sum(np.sign(part[j] - part[i]) for i in range(n - 1) for j in range(i + 1, n))
            if n < 400
            else float(np.sum(np.sign(part[None, :] - part[:, None]) * np.triu(np.ones((n, n)), 1)))
        )
        _, counts = np.unique(part, return_counts=True)
        variance = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
        total_s += float(s)
        total_var += float(variance)
    z = 0.0 if total_s == 0 else (total_s - np.sign(total_s)) / np.sqrt(total_var)
    return total_s, total_var, float(z)


def _mann_kendall(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=8)
    seasonal = bool(p["seasonal"])
    if seasonal:
        need_cycles(ts)
    groups = np.arange(ts.n) % ts.period if seasonal else None
    s, variance, z = mann_kendall(ts.y, groups)
    pvalue = float(2 * stats.norm.sf(abs(z)))
    t = np.arange(1, ts.n + 1, dtype=float)
    sen = stats.theilslopes(ts.y, t, alpha=1 - p["alpha"])
    out = Output(
        f"Tendencia de Mann-Kendall{' estacional' if seasonal else ''}: {ts.name}",
        f"n = {ts.n}" + (f" · periodo {ts.period}" if seasonal else ""),
    )
    note_trimmed(out, ts)
    direction = "creciente" if z > 0 else "decreciente" if z < 0 else "sin dirección"
    out.tiles([("S de Mann-Kendall", s), ("Z", z), ("Valor p", P(pvalue)), ("Pendiente de Sen", float(sen[0]))])
    out.alert(
        "warning" if pvalue < p["alpha"] else "success",
        (
            f"Hay una tendencia monótona {direction} significativa (p = {pvalue:.4f})."
            if pvalue < p["alpha"]
            else f"No hay evidencia de una tendencia monótona (p = {pvalue:.4f})."
        ),
    )
    out.table(
        "Resultado",
        ["S", "Var(S)", "Z", "Valor p", "Pendiente de Sen", f"IC {100 * (1 - p['alpha']):g} % inf.", "IC sup."],
        [[s, variance, z, P(pvalue), float(sen[0]), float(sen[2]), float(sen[3])]],
        note="Prueba no paramétrica: no supone normalidad. La pendiente de Sen es la mediana de las pendientes entre todos los pares y resiste atípicos. Supone observaciones independientes (sin autocorrelación fuerte).",
    )
    labels = np.asarray(ts.labels, dtype=object)
    out.chart(
        line_chart(
            "Serie y pendiente de Sen",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            [
                {"name": "Datos", "x": labels, "y": ts.y},
                {"name": "Pendiente de Sen", "x": labels, "y": sen[1] + sen[0] * t, "dash": "dash"},
            ],
        )
    )
    return out.set(s=s, z=z, p=pvalue, sen_slope=float(sen[0])).build()


def _mann_kendall_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "from scipy import stats\n\n" + _series_code(
        roles,
        dataset,
        "tau = stats.kendalltau(range(len(y)), y)  # equivale a Mann-Kendall (sin corrección por empates en S)\nprint(tau, stats.theilslopes(y))\n",
    )


# ============================================================================ filters
_FILTERS = (
    ("hp", "Hodrick-Prescott (tendencia suave y ciclo)"),
    ("bk", "Baxter-King (pasa-banda)"),
    ("cf", "Christiano-Fitzgerald (pasa-banda)"),
)
_HP_LAMBDA = {12: 129600.0, 4: 1600.0, 1: 6.25}


def _filters(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=20)
    kind = p["filter"]
    x = np.asarray(ts.labels, dtype=object)
    series = ts.series()
    if kind == "hp":
        lamb = float(p["lambda_"]) if p["lambda_"] else _HP_LAMBDA.get(ts.period, 1600.0)
        cycle, trend = hpfilter(series.to_numpy(), lamb=lamb)
        cycle, trend = np.asarray(cycle, float), np.asarray(trend, float)
        detail = f"λ = {lamb:g}"
    else:
        low, high = float(p["low"]), float(p["high"])
        if not 2 <= low < high:
            raise InvalidInputError("Los periodos deben cumplir 2 ≤ mínimo < máximo.")
        if kind == "bk":
            lags = int(p["lags"])
            if ts.n <= 2 * lags + 2:
                raise InvalidInputError(
                    f"Baxter-King pierde {lags} observaciones en cada extremo: necesita más de {2 * lags + 2} datos."
                )
            filtered = np.asarray(bkfilter(ts.y, low=low, high=high, K=lags), float).ravel()
            cycle = np.full(ts.n, np.nan)
            cycle[lags : ts.n - lags] = filtered
        else:
            cycle = np.asarray(cffilter(ts.y, low=low, high=high, drift=bool(p["drift"]))[0], float).ravel()
        trend = ts.y - np.nan_to_num(cycle)
        trend = np.where(np.isfinite(cycle), trend, np.nan)
        detail = f"ciclos de {low:g} a {high:g} periodos"
    out = Output(f"Filtro {dict(_FILTERS)[kind].split(' (')[0]}: {ts.name}", f"n = {ts.n} · {detail}")
    note_trimmed(out, ts)
    ok = np.isfinite(cycle)
    out.tiles(
        [
            ("Desv. est. del ciclo", float(np.nanstd(cycle))),
            ("% de la varianza en el ciclo", 100 * float(np.nanvar(cycle) / np.nanvar(ts.y))),
            ("Observaciones usadas", int(ok.sum())),
        ]
    )
    out.chart(
        line_chart(
            "Serie y componente de tendencia",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            [
                {"name": "Serie", "x": x, "y": ts.y},
                {"name": "Tendencia", "x": x[np.isfinite(trend)], "y": trend[np.isfinite(trend)], "dash": "dash"},
            ],
        )
    )
    out.chart(
        line_chart(
            "Componente cíclico",
            "Tiempo" if ts.dated else "Observación",
            "Ciclo",
            [{"name": "Ciclo", "x": x[ok], "y": cycle[ok]}],
            hlines=[{"value": 0.0}],
        )
    )
    out.dataset(
        "filtrada",
        "Serie, tendencia y ciclo",
        _components_frame(ts, ts.y, trend, np.zeros(ts.n), cycle)
        .drop(columns="estacional")
        .rename(columns={"residuo": "ciclo"}),
    )
    return out.set(cycle_sd=float(np.nanstd(cycle))).build()


def _filters_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    call = {
        "hp": f"hpfilter(y, lamb={p['lambda_'] or 1600})",
        "bk": f"bkfilter(y, low={p['low']}, high={p['high']}, K={int(p['lags'])})",
        "cf": f"cffilter(y, low={p['low']}, high={p['high']})",
    }[p["filter"]]
    return (
        "from statsmodels.tsa.filters.bk_filter import bkfilter\nfrom statsmodels.tsa.filters.cf_filter import cffilter\nfrom statsmodels.tsa.filters.hp_filter import hpfilter\n\n"
        + _series_code(roles, dataset, f"print({call})\n")
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec("ts_plot", "Gráfico de la serie y estadísticos móviles", GROUP_EXPLORE, "La serie con media y desviación móviles, tendencia lineal, promedios por posición del ciclo y gráfico de rezago.", _ts_plot, roles=(SERIES, TIME), params=(FieldSpec("window", "Ventana móvil (vacío = automática)", "integer", optional=True, minimum=2, maximum=1000), PERIOD), code=_ts_plot_code, example={"roles": {"series": "s", "time": "t"}, "params": {}}),
    _spec("ccf", "Correlación cruzada", GROUP_EXPLORE, "Qué serie se adelanta a la otra y por cuántos periodos, con bandas de confianza.", _ccf, roles=(RoleSpec("x", "Serie X", "numeric"), RoleSpec("y", "Serie Y", "numeric"), TIME), params=(FieldSpec("max_lag", "Rezago máximo", "integer", 20, minimum=1, maximum=200), FieldSpec("difference", "Usar primeras diferencias", "boolean", False, help="Recomendado con series con tendencia."), ALPHA), code=_ccf_code, example={"roles": {"x": "x1", "y": "y", "time": "t"}, "params": {"difference": True}}),
    _spec("periodogram", "Periodograma y periodicidades", GROUP_EXPLORE, "Detecta ciclos ocultos con el espectro de la serie y la prueba g de Fisher.", _periodogram, roles=(SERIES, TIME), params=(FieldSpec("detrend", "Tendencia", "select", "linear", (("linear", "Quitar la tendencia lineal"), ("none", "Solo quitar la media"))), FieldSpec("peaks", "Picos a listar", "integer", 5, minimum=1, maximum=20), ALPHA), code=_periodogram_code, example={"roles": {"series": "s", "time": "t"}, "params": {}}),
    _spec("decompose", "Descomposición (clásica y STL)", GROUP_COMPONENTS, "Separa tendencia, estacionalidad y residuo; mide la fuerza de cada componente y entrega la serie desestacionalizada.", _decompose, roles=(SERIES, TIME), params=(FieldSpec("method", "Método", "select", "stl", _DECOMPOSITIONS), PERIOD), outputs=(OutputSpec("componentes", "Componentes de la serie"),), code=_decompose_code, example={"roles": {"series": "s", "time": "t"}, "params": {"method": "stl_robust"}}),
    _spec("trend", "Análisis de tendencia y pronóstico", GROUP_COMPONENTS, "Ajusta una tendencia lineal, cuadrática o exponencial y la proyecta con intervalos de predicción.", _trend, roles=(SERIES, TIME), params=(FieldSpec("model", "Modelo", "select", "linear", _TRENDS), FieldSpec("horizon", "Periodos a pronosticar", "integer", 12, minimum=0, maximum=500), PERIOD, ALPHA), code=_trend_code, example={"roles": {"series": "s", "time": "t"}, "params": {"model": "quadratic", "horizon": 6}}),
    _spec("mann_kendall", "Tendencia de Mann-Kendall (no paramétrica)", GROUP_COMPONENTS, "Prueba si hay tendencia monótona sin suponer normalidad, con la pendiente de Sen; admite la versión estacional.", _mann_kendall, roles=(SERIES, TIME), params=(FieldSpec("seasonal", "Versión estacional (compara dentro de cada posición del ciclo)", "boolean", False), PERIOD, ALPHA), code=_mann_kendall_code, example={"roles": {"series": "s", "time": "t"}, "params": {"seasonal": True}}),
    _spec("filters", "Filtros: Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald", GROUP_COMPONENTS, "Separa la tendencia del ciclo económico; entrega los componentes como dataset.", _filters, roles=(SERIES, TIME), params=(FieldSpec("filter", "Filtro", "select", "hp", _FILTERS), FieldSpec("lambda_", "λ de Hodrick-Prescott (vacío = 129 600 mensual, 1 600 trimestral, 6.25 anual)", "number", optional=True, minimum=0.1, visible_if=("filter", ("hp",))), FieldSpec("low", "Periodo mínimo del ciclo", "number", 6, minimum=2, visible_if=("filter", ("bk", "cf"))), FieldSpec("high", "Periodo máximo del ciclo", "number", 32, minimum=3, visible_if=("filter", ("bk", "cf"))), FieldSpec("lags", "Rezagos K de Baxter-King", "integer", 12, minimum=1, maximum=100, visible_if=("filter", ("bk",))), FieldSpec("drift", "Quitar la deriva (Christiano-Fitzgerald)", "boolean", False, visible_if=("filter", ("cf",))), PERIOD), outputs=(OutputSpec("filtrada", "Serie, tendencia y ciclo"),), code=_filters_code, example={"roles": {"series": "s", "time": "t"}, "params": {"filter": "bk"}}),
)  # fmt: skip

registry.register(*SPECS)
