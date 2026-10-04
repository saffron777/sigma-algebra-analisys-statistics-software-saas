"""Models for a single series: exponential smoothing (ETS), unit-root tests, ARIMA/SARIMA(X) and forecast comparison."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from arch.unitroot import DFGLS, PhillipsPerron
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, zivot_andrews

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart, panels_chart, stem_chart
from sigma.domain.analysis.output import Output, P, bad, good, strong
from sigma.domain.analysis.series_explore import ALPHA, PERIOD, SERIES, TIME, _series_code
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.ts_core import (
    ACCURACY_COLUMNS,
    accuracy,
    accuracy_row,
    forecast_rows,
    future_labels,
    history_and_forecast,
    ljung_box,
    load_series,
    note_trimmed,
)
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "series"
GROUP_MODELS = "Modelos y pronósticos"
GROUP_ROOTS = "Estacionariedad"
FIT_ERRORS = (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError, OverflowError)


# ============================================================================ exponential smoothing (ETS)
_ETS_FIXED = {
    "ses": ("add", None, False, None),
    "holt": ("add", "add", False, None),
    "damped": ("add", "add", True, None),
    "hw_add": ("add", "add", False, "add"),
    "hw_mul": ("add", "add", False, "mul"),
}
_ETS_METHODS = (
    ("auto", "Automático (el de menor AICc)"),
    ("ses", "Exponencial simple (nivel)"),
    ("holt", "Holt (nivel y tendencia)"),
    ("damped", "Holt con tendencia amortiguada"),
    ("hw_add", "Holt-Winters aditivo"),
    ("hw_mul", "Holt-Winters multiplicativo"),
)


def _ets_fit(y: np.ndarray, config: tuple, period: int):
    error, trend, damped, seasonal = config
    model = ETSModel(
        pd.Series(np.asarray(y, float)),  # statsmodels needs a pandas series to build the prediction frame
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=period if seasonal else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=200)


def _ets_label(config: tuple) -> str:
    error, trend, damped, seasonal = config
    letter = lambda v: {None: "N", "add": "A", "mul": "M"}[v]  # noqa: E731
    return f"ETS({letter(error)},{letter(trend)}{'d' if damped else ''},{letter(seasonal)})"


def _ets_candidates(y: np.ndarray, period: int) -> list[tuple]:
    positive = bool((y > 0).all())
    seasonal_options: list[str | None] = [None]
    if period >= 2 and len(y) >= 2 * period + 2:
        seasonal_options += ["add"] + (["mul"] if positive else [])
    candidates = []
    for error in ("add",) + (("mul",) if positive else ()):
        for trend, damped in ((None, False), ("add", False), ("add", True)):
            for seasonal in seasonal_options:
                candidates.append((error, trend, damped, seasonal))
    return candidates


def ets_best(y: np.ndarray, period: int) -> tuple[tuple, Any, list[tuple[tuple, float]]]:
    """Fit every admissible ETS model and keep the one with the lowest AICc: (config, results, ranking)."""
    ranking, fits = [], {}
    for config in _ets_candidates(y, period):
        try:
            result = _ets_fit(y, config, period)
        except FIT_ERRORS:
            continue
        if np.isfinite(result.aicc):
            ranking.append((config, float(result.aicc)))
            fits[config] = result
    if not fits:
        raise ComputationError("Ningún modelo de suavizamiento exponencial se pudo ajustar a esta serie.")
    ranking.sort(key=lambda item: item[1])
    best = ranking[0][0]
    return best, fits[best], ranking


def _smoothing(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"])
    method, h, alpha, holdout = p["method"], int(p["horizon"]), p["alpha"], int(p["holdout"])
    seasonal_needed = method in ("hw_add", "hw_mul")
    if seasonal_needed and (ts.period < 2 or ts.n < 2 * ts.period + 2):
        raise InvalidInputError("Holt-Winters necesita el periodo estacional y al menos dos ciclos completos.")
    if method == "hw_mul" and (ts.y <= 0).any():
        raise InvalidInputError("Holt-Winters multiplicativo necesita valores estrictamente positivos.")
    ranking: list[tuple[tuple, float]] = []
    if method == "auto":
        config, result, ranking = ets_best(ts.y, ts.period)
    else:
        config = _ETS_FIXED[method]
        try:
            result = _ets_fit(ts.y, config, ts.period)
        except FIT_ERRORS as error:
            raise ComputationError(f"No se pudo ajustar el modelo: {error}") from None
    fitted = np.asarray(result.fittedvalues, float)
    resid = ts.y - fitted
    metrics = accuracy(ts.y, fitted, ts.y, ts.period)
    names = list(result.param_names)
    labels = {
        "smoothing_level": "α (nivel)",
        "smoothing_trend": "β (tendencia)",
        "smoothing_seasonal": "γ (estacional)",
        "damping_trend": "φ (amortiguamiento)",
        "initial_level": "Nivel inicial",
        "initial_trend": "Tendencia inicial",
    }
    out = Output(
        f"Suavizamiento exponencial: {ts.name}",
        f"{_ets_label(config)} · n = {ts.n}" + (f" · periodo {ts.period}" if config[3] else ""),
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Modelo", _ets_label(config)),
            ("AICc", float(result.aicc)),
            ("RMSE", metrics["rmse"]),
            ("MAPE %", metrics["mape"]),
        ]
    )
    if ranking:
        out.table(
            "Modelos comparados (menor AICc = mejor)",
            ["Modelo", "AICc", "Δ AICc"],
            [[_ets_label(c), v, v - ranking[0][1]] for c, v in ranking[:8]],
            note="Se ajustaron todas las combinaciones admisibles de error, tendencia y estacionalidad.",
        )
    out.table(
        "Parámetros",
        ["Parámetro", "Estimación"],
        [
            [labels.get(name, name.replace("initial_seasonal.", "Estacional inicial ")), float(value)]
            for name, value in zip(names, result.params, strict=True)
            if not name.startswith("initial_seasonal") or len(names) < 40
        ],
    )
    out.table("Exactitud en la muestra", ["Medida", *ACCURACY_COLUMNS], [accuracy_row("Ajuste", metrics)])
    lb = ljung_box(
        resid, min(10, ts.n // 5), fitted_params=sum(n in labels and not n.startswith("initial") for n in names)
    )
    out.alert(
        "success" if lb[1] >= alpha else "warning",
        f"Ljung-Box de los residuos: Q = {lb[0]:.2f}, p = {lb[1]:.4f} → "
        + (
            "sin autocorrelación residual: el modelo captura la dinámica."
            if lb[1] >= alpha
            else "los residuos aún tienen autocorrelación: pruebe otro modelo (p. ej. ARIMA)."
        ),
    )
    if holdout:
        if ts.n - holdout < 12:
            raise InvalidInputError("La validación deja muy pocos datos de entrenamiento.")
        train = ts.y[:-holdout]
        sub = _ets_fit(train, config, ts.period)
        predicted = np.asarray(
            sub.get_prediction(start=len(train), end=len(train) + holdout - 1).summary_frame()["mean"], float
        )
        out_metrics = accuracy(ts.y[-holdout:], predicted, train, ts.period)
        out.table(
            f"Validación con los últimos {holdout} datos (entrenando con el resto)",
            ["Medida", *ACCURACY_COLUMNS],
            [accuracy_row("Fuera de la muestra", out_metrics)],
        )
    if h:
        frame_f = result.get_prediction(start=ts.n, end=ts.n + h - 1).summary_frame(alpha=alpha)
        mean, lower, upper = (
            np.asarray(frame_f["mean"], float),
            np.asarray(frame_f["pi_lower"], float),
            np.asarray(frame_f["pi_upper"], float),
        )
        out.table(
            "Pronósticos",
            ["Periodo", "Pronóstico", f"PI {100 * (1 - alpha):g} % inf.", "PI sup."],
            forecast_rows(future_labels(ts, h), mean, lower, upper),
        )
        out.chart(history_and_forecast(ts, fitted, mean, lower, upper, alpha))
    else:
        out.chart(
            line_chart(
                "Serie y ajuste",
                "Tiempo" if ts.dated else "Observación",
                ts.name,
                [
                    {"name": "Datos", "x": ts.labels, "y": ts.y},
                    {"name": "Ajustado", "x": ts.labels, "y": fitted, "dash": "dash"},
                ],
            )
        )
    stamps = pd.Series(pd.DatetimeIndex(ts.index)) if ts.dated else pd.Series(np.arange(1, ts.n + 1))
    out.dataset(
        "ajuste",
        "Ajuste y residuos",
        pd.DataFrame({"tiempo": stamps, ts.name: ts.y, "ajustado": fitted, "residuo": resid}),
    )
    return out.set(
        model=_ets_label(config),
        aicc=float(result.aicc),
        rmse=metrics["rmse"],
        mape=metrics["mape"],
        ljung_box_p=lb[1],
        parameters={
            n: float(v) for n, v in zip(names, result.params, strict=True) if not n.startswith("initial_seasonal")
        },
    ).build()


def _smoothing_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    error, trend, damped, seasonal = _ETS_FIXED.get(p["method"], ("add", "add", False, "add"))
    period = int(p["period"] or 12)
    return "from statsmodels.tsa.exponential_smoothing.ets import ETSModel\n\n" + _series_code(
        roles,
        dataset,
        f"fit = ETSModel(y, error={error!r}, trend={trend!r}, damped_trend={damped}, seasonal={seasonal!r}, seasonal_periods={period if seasonal else None}).fit()\nprint(fit.summary())\nprint(fit.get_prediction(start=len(y), end=len(y) + {max(int(p['horizon']), 1) - 1}).summary_frame())\n",
    )


# ============================================================================ unit roots
_ROOT_TESTS = (
    ("adf", "Dickey-Fuller aumentada (ADF)"),
    ("pp", "Phillips-Perron"),
    ("dfgls", "DF-GLS (Elliott-Rothenberg-Stock)"),
    ("kpss", "KPSS"),
    ("za", "Zivot-Andrews (con un quiebre)"),
)
_TREND_NAMES = {"c": "constante", "ct": "constante y tendencia", "n": "sin término determinista"}


def _unit_root_rows(
    y: np.ndarray, tests: list[str], regression: str, alpha: float, max_lags: int | None
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for test in tests:
            if test == "adf":
                stat, pvalue, lags, _, critical, _ = adfuller(y, regression=regression, autolag="AIC", maxlag=max_lags)
                rows.append(
                    [
                        "ADF",
                        "raíz unitaria (no estacionaria)",
                        float(stat),
                        float(pvalue),
                        float(critical["5%"]),
                        int(lags),
                        pvalue < alpha,
                    ]
                )
            elif test == "pp":
                result = PhillipsPerron(y, trend=regression)
                rows.append(
                    [
                        "Phillips-Perron",
                        "raíz unitaria (no estacionaria)",
                        float(result.stat),
                        float(result.pvalue),
                        float(result.critical_values["5%"]),
                        int(result.lags),
                        result.pvalue < alpha,
                    ]
                )
            elif test == "dfgls":
                result = DFGLS(y, trend="ct" if regression == "ct" else "c")
                rows.append(
                    [
                        "DF-GLS",
                        "raíz unitaria (no estacionaria)",
                        float(result.stat),
                        float(result.pvalue),
                        float(result.critical_values["5%"]),
                        int(result.lags),
                        result.pvalue < alpha,
                    ]
                )
            elif test == "kpss":
                stat, pvalue, lags, critical = kpss(y, regression="ct" if regression == "ct" else "c", nlags="auto")
                rows.append(
                    [
                        "KPSS",
                        "estacionariedad",
                        float(stat),
                        float(pvalue),
                        float(critical["5%"]),
                        int(lags),
                        pvalue >= alpha,
                    ]
                )
            elif test == "za":
                stat, pvalue, critical, lags, breakpoint_ = zivot_andrews(
                    y, regression={"c": "c", "ct": "ct", "n": "c"}[regression], autolag="AIC"
                )
                rows.append(
                    [
                        f"Zivot-Andrews (quiebre en la obs. {int(breakpoint_) + 1})",
                        "raíz unitaria (no estacionaria)",
                        float(stat),
                        float(pvalue),
                        float(critical["5%"]),
                        int(lags),
                        pvalue < alpha,
                    ]
                )
    return rows


def _unit_root(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=15)
    tests, alpha, regression = p["tests"], p["alpha"], p["regression"]
    if not tests:
        raise InvalidInputError("Elija al menos una prueba.")
    orders = {"0": [0], "01": [0, 1], "012": [0, 1, 2]}[p["differences"]]
    out = Output(
        f"Pruebas de raíz unitaria y estacionariedad: {ts.name}",
        f"n = {ts.n} · especificación: {_TREND_NAMES[regression]} · α = {alpha:g}",
    )
    note_trimmed(out, ts)
    verdicts: dict[int, list[bool]] = {}
    for d in orders:
        values = np.diff(ts.y, d) if d else ts.y
        if len(values) < 12:
            continue
        rows = _unit_root_rows(
            values,
            tests,
            "c" if d > 0 and regression == "ct" else regression,
            alpha,
            int(p["max_lags"]) if p["max_lags"] else None,
        )
        verdicts[d] = [r[6] for r in rows]
        title = "Serie en niveles" if d == 0 else f"Serie en {'primeras' if d == 1 else 'segundas'} diferencias"
        out.table(
            title,
            ["Prueba", "Hipótesis nula", "Estadístico", "Valor p", "Valor crítico 5 %", "Rezagos", "¿Estacionaria?"],
            [[r[0], r[1], r[2], P(r[3]), r[4], r[5], good("sí") if r[6] else bad("no")] for r in rows],
            note="Para ADF, PP, DF-GLS y Zivot-Andrews se rechaza la raíz unitaria con un valor p pequeño; para KPSS, un valor p pequeño indica NO estacionariedad. Los valores p de KPSS se acotan a [0.01, 0.10]."
            if d == 0
            else None,
        )
    integrated = next((d for d in orders if d in verdicts and sum(verdicts[d]) > len(verdicts[d]) / 2), None)
    if integrated is None:
        out.alert(
            "warning",
            "La mayoría de las pruebas no indican estacionariedad ni siquiera en las diferencias probadas: pruebe más diferencias o transforme la serie.",
        )
    elif integrated == 0:
        out.alert("success", "La mayoría de las pruebas indican que la serie es estacionaria en niveles.")
    else:
        out.alert(
            "info",
            f"La serie parece integrada de orden {integrated}: se vuelve estacionaria tras {integrated} diferencia(s).",
        )
    out.tiles(
        [
            ("Observaciones", ts.n),
            ("Pruebas", len(tests)),
            ("Orden de integración sugerido", integrated if integrated is not None else "—"),
        ]
    )
    if len(orders) > 1 and ts.n > 15:
        x = np.asarray(ts.labels, dtype=object)
        panels = [
            {**line_chart(t, None, None, [{"name": t, "x": xs, "y": v}], height=230), "title": t}
            for t, xs, v in (("Niveles", x, ts.y), ("Primeras diferencias", x[1:], np.diff(ts.y)))
        ]
        out.chart(panels_chart(f"{ts.name}: niveles y primeras diferencias", panels, columns=1, height=500))
    return out.set(integration_order=integrated, verdicts={str(d): v for d, v in verdicts.items()}).build()


def _unit_root_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from arch.unitroot import DFGLS, PhillipsPerron\nfrom statsmodels.tsa.stattools import adfuller, kpss\n\n"
        + _series_code(
            roles,
            dataset,
            f"print(adfuller(y, regression={p['regression']!r}, autolag='AIC'))\nprint(kpss(y, regression={'ct' if p['regression'] == 'ct' else 'c'!r}, nlags='auto'))\nprint(PhillipsPerron(y, trend={p['regression']!r}).summary())\n",
        )
    )


# ============================================================================ ARIMA / SARIMA / SARIMAX
def _seasonal_strength(y: np.ndarray, period: int) -> float:
    if period < 2 or len(y) < 2 * period + 2:
        return 0.0
    result = STL(y, period=period, robust=True).fit()
    remainder, seasonal = np.asarray(result.resid), np.asarray(result.seasonal)
    return float(max(0.0, 1 - np.var(remainder) / np.var(remainder + seasonal)))


def choose_orders(y: np.ndarray, period: int) -> tuple[int, int]:
    """Differences (d, D) by repeated KPSS tests and the strength of the seasonality (Hyndman and Khandakar)."""
    seasonal_d = 1 if _seasonal_strength(y, period) > 0.64 else 0
    work = np.diff(y, period) if seasonal_d and len(y) > period + 15 else y
    d = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while d < 2 and len(work) > 15 and kpss(work, regression="c", nlags="auto")[1] < 0.05:
            work = np.diff(work)
            d += 1
    return d, seasonal_d


def _sarimax(y: np.ndarray, order: tuple, seasonal: tuple, trend: str, exog=None):
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal,
        trend=trend,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=200)


def arima_search(
    y: np.ndarray, period: int, d: int, big_d: int, max_pq: int, seasonal_terms: bool
) -> list[tuple[tuple, tuple, str, float]]:
    """Grid search by AICc: a list of ``(order, seasonal_order, trend, aicc)`` best first."""
    trend = "c" if d + big_d == 0 else "n"
    found = []
    seasonal_range = range(0, 2) if seasonal_terms else range(0, 1)
    for p, q, sp, sq in product(range(max_pq + 1), range(max_pq + 1), seasonal_range, seasonal_range):
        if p + q + sp + sq == 0 and trend == "n" and not big_d and not d:
            continue
        order, seasonal = (p, d, q), ((sp, big_d, sq, period) if seasonal_terms else (0, 0, 0, 0))
        try:
            result = _sarimax(y, order, seasonal, trend)
        except FIT_ERRORS:
            continue
        if np.isfinite(result.aicc):
            found.append((order, seasonal, trend, float(result.aicc)))
    found.sort(key=lambda item: item[3])
    return found


def _order_text(order: tuple, seasonal: tuple) -> str:
    base = f"ARIMA({order[0]},{order[1]},{order[2]})"
    return base + (
        f"({seasonal[0]},{seasonal[1]},{seasonal[2]})[{seasonal[3]}]" if seasonal[3] > 1 and any(seasonal[:3]) else ""
    )


def _arima(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=20)
    alpha, h, holdout = p["alpha"], int(p["horizon"]), int(p["holdout"])
    exog = None
    if roles.get("exog"):
        if h:
            raise InvalidInputError(
                "Con variables exógenas el pronóstico necesita sus valores futuros: ponga el horizonte en 0 para ajustar y diagnosticar el modelo."
            )
        columns = frame.loc[ts.rows, roles["exog"]].apply(pd.to_numeric, errors="coerce")
        if columns.isna().any().any():
            raise InvalidInputError("Las variables exógenas no deben tener faltantes en las filas de la serie.")
        exog = columns.to_numpy(float)
    seasonal_terms = ts.period > 1 and ts.n >= 2 * ts.period + 5
    candidates: list[tuple[tuple, tuple, str, float]] = []
    if p["mode"] == "auto":
        d, big_d = choose_orders(ts.y, ts.period) if seasonal_terms else choose_orders(ts.y, 1)
        if not seasonal_terms:
            big_d = 0
        candidates = arima_search(ts.y, ts.period, d, big_d, int(p["max_order"]), seasonal_terms)
        if not candidates:
            raise ComputationError("Ningún modelo ARIMA se pudo ajustar a esta serie.")
        order, seasonal, trend, _ = candidates[0]
    else:
        order = (int(p["p"]), int(p["d"]), int(p["q"]))
        seasonal = (
            (int(p["sp"]), int(p["sd"]), int(p["sq"]), ts.period if ts.period > 1 else 0)
            if (p["sp"] or p["sd"] or p["sq"])
            else (0, 0, 0, 0)
        )
        if seasonal[3] == 0 and any(seasonal[:3]):
            raise InvalidInputError("Para la parte estacional indique el periodo.")
        trend = p["trend"] if p["trend"] != "auto" else ("c" if order[1] + seasonal[1] == 0 else "n")
    try:
        result = _sarimax(ts.y, order, seasonal, trend, exog)
    except FIT_ERRORS as error:
        raise ComputationError(f"No se pudo ajustar el modelo {_order_text(order, seasonal)}: {error}") from None
    fitted = np.asarray(result.fittedvalues, float)
    resid = np.asarray(result.resid, float)
    burn = order[1] + seasonal[1] * max(seasonal[3], 1)
    clean = resid[burn:]
    metrics = accuracy(ts.y[burn:], fitted[burn:], ts.y, ts.period)
    arma_terms = order[0] + order[2] + seasonal[0] + seasonal[2]
    lb = ljung_box(clean, min(max(10, 2 * max(seasonal[3], 1)), len(clean) // 4), fitted_params=arma_terms)
    jb = jarque_bera(clean / clean.std())
    label = _order_text(order, seasonal) + (" con constante" if trend == "c" else "")
    out = Output(
        f"{label}: {ts.name}",
        f"n = {ts.n}"
        + (f" · periodo {ts.period}" if seasonal[3] > 1 else "")
        + (" · selección automática por AICc" if p["mode"] == "auto" else ""),
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Modelo", _order_text(order, seasonal)),
            ("AICc", float(result.aicc)),
            ("BIC", float(result.bic)),
            ("RMSE", metrics["rmse"]),
            ("Ljung-Box p", P(lb[1])),
        ]
    )
    if candidates:
        out.table(
            "Mejores modelos encontrados (menor AICc)",
            ["Modelo", "AICc", "Δ AICc"],
            [
                [_order_text(o, s) + (" + const." if t == "c" else ""), a, a - candidates[0][3]]
                for o, s, t, a in candidates[:8]
            ],
            note=f"Se probaron {len(candidates)} modelos con d = {candidates[0][0][1]}, D = {candidates[0][1][1]}.",
        )
    names = list(result.param_names)
    z = stats.norm.ppf(1 - alpha / 2)
    rows = [
        [
            name,
            float(result.params[i]),
            float(result.bse[i]),
            float(result.zvalues[i]),
            P(float(result.pvalues[i])),
            float(result.params[i] - z * result.bse[i]),
            float(result.params[i] + z * result.bse[i]),
        ]
        for i, name in enumerate(names)
    ]
    out.table(
        "Parámetros",
        ["Parámetro", "Estimación", "Error est.", "z", "Valor p", f"IC {100 * (1 - alpha):g} % inf.", "IC sup."],
        rows,
        note="ar = autorregresivo, ma = media móvil, S = estacional; sigma2 = varianza del error.",
    )
    out.table(
        "Diagnóstico de los residuos",
        ["Prueba", "Estadístico", "Valor p", "Lectura"],
        [
            [
                "Ljung-Box (autocorrelación)",
                lb[0],
                P(lb[1]),
                "residuos sin autocorrelación"
                if lb[1] >= alpha
                else "queda autocorrelación: el modelo es insuficiente",
            ],
            [
                "Jarque-Bera (normalidad)",
                float(jb[0]),
                P(float(jb[1])),
                "compatible con normalidad" if jb[1] >= alpha else "residuos no normales (afecta los intervalos)",
            ],
        ],
    )
    lags = min(24, len(clean) // 3)
    values, band = acf(clean, nlags=lags, fft=True)[1:], 1.96 / np.sqrt(len(clean))
    out.chart(stem_chart("Autocorrelación de los residuos", "Rezago", "ACF", np.arange(1, lags + 1), values, band=band))
    x = np.asarray(ts.labels, dtype=object)
    out.chart(
        line_chart(
            "Residuos",
            "Tiempo" if ts.dated else "Observación",
            "Residuo",
            [{"name": "Residuo", "x": x[burn:], "y": clean}],
            hlines=[{"value": 0.0}],
        )
    )
    if holdout:
        if ts.n - holdout < 20:
            raise InvalidInputError("La validación deja muy pocos datos de entrenamiento.")
        train = ts.y[:-holdout]
        sub = _sarimax(train, order, seasonal, trend)
        predicted = np.asarray(sub.get_forecast(holdout).predicted_mean, float)
        out.table(
            f"Validación con los últimos {holdout} datos (entrenando con el resto)",
            ["Medida", *ACCURACY_COLUMNS],
            [accuracy_row("Fuera de la muestra", accuracy(ts.y[-holdout:], predicted, train, ts.period))],
        )
    if h:
        frame_f = result.get_forecast(h).summary_frame(alpha=alpha)
        mean, lower, upper = (
            np.asarray(frame_f["mean"], float),
            np.asarray(frame_f["mean_ci_lower"], float),
            np.asarray(frame_f["mean_ci_upper"], float),
        )
        out.table(
            "Pronósticos",
            ["Periodo", "Pronóstico", f"PI {100 * (1 - alpha):g} % inf.", "PI sup."],
            forecast_rows(future_labels(ts, h), mean, lower, upper),
        )
        out.chart(history_and_forecast(ts, fitted, mean, lower, upper, alpha))
    stamps = pd.Series(pd.DatetimeIndex(ts.index)) if ts.dated else pd.Series(np.arange(1, ts.n + 1))
    out.dataset(
        "ajuste",
        "Ajuste y residuos",
        pd.DataFrame({"tiempo": stamps, ts.name: ts.y, "ajustado": fitted, "residuo": resid}),
    )
    return out.set(
        order=list(order),
        seasonal_order=list(seasonal),
        aicc=float(result.aicc),
        bic=float(result.bic),
        ljung_box_p=lb[1],
        rmse=metrics["rmse"],
        parameters={n: float(v) for n, v in zip(names, result.params, strict=True)},
    ).build()


def _arima_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    order = (p["p"], p["d"], p["q"]) if p["mode"] == "manual" else (1, 1, 1)
    seasonal = (
        (p["sp"], p["sd"], p["sq"], int(p["period"] or 12))
        if p["mode"] == "manual"
        else (0, 1, 1, int(p["period"] or 12))
    )
    return "from statsmodels.tsa.statespace.sarimax import SARIMAX\n\n" + _series_code(
        roles,
        dataset,
        f"fit = SARIMAX(y, order={tuple(order)}, seasonal_order={tuple(seasonal)}).fit(disp=False)\nprint(fit.summary())\nprint(fit.get_forecast({max(int(p['horizon']), 1)}).summary_frame())\n"
        + (
            f"# modo automático: pmdarima.auto_arima(y, seasonal=True, m={int(p['period'] or 12)})\n"
            if p["mode"] == "auto"
            else ""
        ),
    )


# ============================================================================ forecast comparison (rolling-origin backtest)
_BACKTEST = (
    ("naive", "Ingenuo (último valor)"),
    ("snaive", "Ingenuo estacional"),
    ("drift", "Con deriva (extrapola la pendiente)"),
    ("mean", "Media histórica"),
    ("ses", "Suavizamiento exponencial simple"),
    ("holt", "Holt (tendencia)"),
    ("ets", "ETS automático"),
    ("arima", "ARIMA automático (más lento)"),
)


def _forecast_with(method: str, train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    n = len(train)
    steps = np.arange(1, horizon + 1)
    if method == "naive":
        return np.repeat(train[-1], horizon)
    if method == "snaive":
        if period < 2 or n < period:
            raise InvalidInputError("El ingenuo estacional necesita el periodo y al menos un ciclo.")
        return np.array([train[-period + ((i - 1) % period)] for i in steps])
    if method == "drift":
        return train[-1] + steps * (train[-1] - train[0]) / (n - 1)
    if method == "mean":
        return np.repeat(train.mean(), horizon)
    if method in ("ses", "holt"):
        result = _ets_fit(train, _ETS_FIXED[method], period)
    elif method == "ets":
        result = ets_best(train, period)[1]
    else:
        seasonal_terms = period > 1 and n >= 2 * period + 5
        d, big_d = choose_orders(train, period if seasonal_terms else 1)
        top = arima_search(train, period, d, big_d if seasonal_terms else 0, 1, seasonal_terms)
        if not top:
            raise ComputationError("ARIMA no se pudo ajustar en uno de los orígenes.")
        order, seasonal, trend, _ = top[0]
        return np.asarray(_sarimax(train, order, seasonal, trend).get_forecast(horizon).predicted_mean, float)
    return np.asarray(result.get_prediction(start=n, end=n + horizon - 1).summary_frame()["mean"], float)


def _backtest(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=24)
    methods, horizon, origins = p["methods"], int(p["horizon"]), int(p["origins"])
    if not methods:
        raise InvalidInputError("Elija al menos un método.")
    step = max(1, int(p["step"]))
    last_origin = ts.n - horizon
    first_origin = last_origin - (origins - 1) * step
    if first_origin < max(15, 2 * ts.period if ts.period > 1 else 15):
        raise InvalidInputError(
            "No hay datos suficientes para tantos orígenes y ese horizonte: reduzca los orígenes o el horizonte."
        )
    cut_points = [first_origin + i * step for i in range(origins)]
    errors: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    last_forecasts: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for method in methods:
        for cut in cut_points:
            try:
                forecast = _forecast_with(method, ts.y[:cut], horizon, ts.period)
            except (InvalidInputError, ComputationError) as error:
                skipped[method] = error.message
                break
            errors[method].append(ts.y[cut : cut + horizon] - forecast)
            if cut == cut_points[-1]:
                last_forecasts[method] = forecast
    train0 = ts.y[: cut_points[0]]
    scale_lag = ts.period if ts.period > 1 and len(train0) > ts.period else 1
    scale = float(np.mean(np.abs(np.diff(train0, scale_lag))))
    rows, by_horizon = [], {}
    for method in methods:
        if method in skipped or not errors[method]:
            continue
        e = np.vstack(errors[method])
        actual = np.vstack([ts.y[c : c + horizon] for c in cut_points[: len(e)]])
        by_horizon[method] = np.sqrt((e**2).mean(axis=0))
        nonzero = actual != 0
        rows.append(
            [
                dict(_BACKTEST)[method],
                float(np.abs(e).mean()),
                float(np.sqrt((e**2).mean())),
                float(100 * np.abs(e[nonzero] / actual[nonzero]).mean()) if nonzero.any() else float("nan"),
                float(np.abs(e).mean() / scale) if scale > 0 else float("nan"),
            ]
        )
    if not rows:
        raise InvalidInputError(
            "Ningún método se pudo evaluar: " + "; ".join(f"{dict(_BACKTEST)[m]}: {msg}" for m, msg in skipped.items())
        )
    rows.sort(key=lambda r: r[2])
    out = Output(
        f"Comparación de pronósticos de {ts.name}",
        f"{origins} orígenes de pronóstico · horizonte {horizon} · n = {ts.n}",
    )
    note_trimmed(out, ts)
    out.tiles(
        [("Mejor método", rows[0][0]), ("RMSE del mejor", rows[0][2]), ("Orígenes", origins), ("Horizonte", horizon)]
    )
    out.table(
        "Error de pronóstico por método (menor RMSE primero)",
        ["Método", "MAE", "RMSE", "MAPE %", "MASE"],
        [[strong(r[0]) if i == 0 else r[0], *r[1:]] for i, r in enumerate(rows)],
        note="Evaluación con origen móvil: se ajusta con los datos hasta cada origen y se pronostican los siguientes periodos. MASE < 1 supera al ingenuo (estacional) del entrenamiento.",
    )
    if skipped:
        out.alert(
            "info", "No se evaluaron: " + "; ".join(f"{dict(_BACKTEST)[m]} ({msg})" for m, msg in skipped.items())
        )
    steps = np.arange(1, horizon + 1)
    out.chart(
        line_chart(
            "RMSE según el horizonte de pronóstico",
            "Periodos hacia adelante",
            "RMSE",
            [{"name": dict(_BACKTEST)[m], "x": steps, "y": by_horizon[m], "markers": True} for m in by_horizon],
        )
    )
    cut = cut_points[-1]
    x = np.asarray(ts.labels, dtype=object)
    keep = max(0, cut - 3 * horizon)
    series = [{"name": "Datos", "x": x[keep : cut + horizon], "y": ts.y[keep : cut + horizon]}] + [
        {"name": dict(_BACKTEST)[m], "x": x[cut : cut + horizon], "y": f, "dash": "dash"}
        for m, f in last_forecasts.items()
    ]
    out.chart(
        line_chart(
            "Pronósticos del último origen frente a lo ocurrido",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            series,
        )
    )
    return out.set(best=rows[0][0], ranking=[r[0] for r in rows], rmse={r[0]: r[2] for r in rows}).build()


def _backtest_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "import numpy as np\n\n" + _series_code(
        roles,
        dataset,
        f"h = {int(p['horizon'])}\ntrain, test = y[:-h], y[-h:]\nnaive = np.repeat(train[-1], h)\nprint(np.sqrt(np.mean((test - naive) ** 2)))  # RMSE del ingenuo\n",
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


HORIZON = FieldSpec("horizon", "Periodos a pronosticar", "integer", 12, minimum=0, maximum=500)

SPECS = (
    _spec("smoothing", "Suavizamiento exponencial y ETS", GROUP_MODELS, "Simple, Holt, Holt-Winters o ETS automático por AICc, con pronósticos e intervalos de predicción.", _smoothing, roles=(SERIES, TIME), params=(FieldSpec("method", "Método", "select", "auto", _ETS_METHODS), HORIZON, FieldSpec("holdout", "Validar con los últimos k datos (0 = no)", "integer", 0, minimum=0, maximum=500), PERIOD, ALPHA), outputs=(OutputSpec("ajuste", "Ajuste y residuos"),), code=_smoothing_code, example={"roles": {"series": "s", "time": "t"}, "params": {"method": "hw_add", "horizon": 6, "holdout": 12}}),
    _spec("arima", "ARIMA / SARIMA / SARIMAX", GROUP_MODELS, "Ajuste manual o selección automática de órdenes por AICc, diagnóstico de residuos, validación y pronósticos con intervalos.", _arima, heavy=True, roles=(SERIES, TIME, RoleSpec("exog", "Variables exógenas (SARIMAX, opcional)", "numeric", multiple=True, optional=True)), params=(FieldSpec("mode", "Orden", "select", "auto", (("auto", "Automático (búsqueda por AICc)"), ("manual", "Manual"))), FieldSpec("max_order", "Máximo de p y q a probar", "integer", 2, minimum=1, maximum=4, visible_if=("mode", ("auto",))), FieldSpec("p", "p (autorregresivo)", "integer", 1, minimum=0, maximum=10, visible_if=("mode", ("manual",))), FieldSpec("d", "d (diferencias)", "integer", 1, minimum=0, maximum=3, visible_if=("mode", ("manual",))), FieldSpec("q", "q (media móvil)", "integer", 1, minimum=0, maximum=10, visible_if=("mode", ("manual",))), FieldSpec("sp", "P estacional", "integer", 0, minimum=0, maximum=3, visible_if=("mode", ("manual",))), FieldSpec("sd", "D estacional", "integer", 0, minimum=0, maximum=2, visible_if=("mode", ("manual",))), FieldSpec("sq", "Q estacional", "integer", 0, minimum=0, maximum=3, visible_if=("mode", ("manual",))), FieldSpec("trend", "Término determinista", "select", "auto", (("auto", "Constante solo si no hay diferencias"), ("n", "Ninguno"), ("c", "Constante"), ("t", "Tendencia lineal"), ("ct", "Constante y tendencia")), visible_if=("mode", ("manual",))), HORIZON, FieldSpec("holdout", "Validar con los últimos k datos (0 = no)", "integer", 0, minimum=0, maximum=500), PERIOD, ALPHA), outputs=(OutputSpec("ajuste", "Ajuste y residuos"),), code=_arima_code, example={"roles": {"series": "s", "time": "t"}, "params": {"mode": "manual", "p": 1, "d": 1, "q": 1, "sp": 0, "sd": 1, "sq": 1, "horizon": 6}}),
    _spec("backtest", "Comparar métodos de pronóstico (origen móvil)", GROUP_MODELS, "Evalúa ingenuo, deriva, suavizamientos, ETS y ARIMA con validación de origen móvil; MAE, RMSE, MAPE y MASE por horizonte.", _backtest, heavy=True, roles=(SERIES, TIME), params=(FieldSpec("methods", "Métodos", "multiselect", ["naive", "snaive", "drift", "ses", "holt", "ets"], _BACKTEST), FieldSpec("horizon", "Horizonte (periodos hacia adelante)", "integer", 6, minimum=1, maximum=100), FieldSpec("origins", "Orígenes de pronóstico", "integer", 6, minimum=1, maximum=24), FieldSpec("step", "Separación entre orígenes", "integer", 3, minimum=1, maximum=100), PERIOD), code=_backtest_code, example={"roles": {"series": "s", "time": "t"}, "params": {"methods": ["naive", "snaive", "ses", "holt"], "origins": 3, "horizon": 6}}),
    _spec("unit_root", "Raíces unitarias y estacionariedad (ADF, PP, DF-GLS, KPSS, Zivot-Andrews)", GROUP_ROOTS, "Decide si la serie es estacionaria y cuántas diferencias necesita, con varias pruebas a la vez.", _unit_root, roles=(SERIES, TIME), params=(FieldSpec("tests", "Pruebas", "multiselect", ["adf", "pp", "kpss"], _ROOT_TESTS), FieldSpec("regression", "Términos deterministas", "select", "c", (("c", "Constante"), ("ct", "Constante y tendencia"), ("n", "Ninguno (ADF y PP)"))), FieldSpec("differences", "Probar también en diferencias", "select", "01", (("0", "Solo en niveles"), ("01", "Niveles y primeras diferencias"), ("012", "Hasta segundas diferencias"))), FieldSpec("max_lags", "Rezagos máximos de ADF (vacío = automático)", "integer", optional=True, minimum=0, maximum=60), PERIOD, ALPHA), code=_unit_root_code, example={"roles": {"series": "s", "time": "t"}, "params": {"tests": ["adf", "pp", "dfgls", "kpss", "za"]}}),
)  # fmt: skip

registry.register(*SPECS)
