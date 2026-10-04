"""What every time-series analysis needs: the series in time order, its frequency and seasonal period, forecast dates, accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.analysis.output import Output, P
from sigma.domain.errors import InvalidInputError

_PERIODS = {
    "M": 12,
    "ME": 12,
    "MS": 12,
    "SM": 24,
    "SMS": 24,
    "Q": 4,
    "QE": 4,
    "QS": 4,
    "D": 7,
    "B": 5,
    "W": 52,
    "H": 24,
    "h": 24,
    "min": 60,
    "T": 60,
}
_YEARLY = {"A", "Y", "YE", "YS", "AS", "BA", "BY"}


@dataclass
class TS:
    name: str
    y: np.ndarray  # values in time order, no gaps
    index: pd.Index  # DatetimeIndex when the dates are regular, otherwise 1..n
    freq: str | None  # pandas offset alias of a regular DatetimeIndex
    period: int  # seasonal period (1 = none)
    labels: list[Any]  # what to put on a chart axis
    dropped: int  # missing values trimmed from the ends
    dated: bool
    irregular: bool = False  # dates were given but are not evenly spaced
    rows: Any = None  # the row labels of the original frame that the series came from, in time order

    @property
    def n(self) -> int:
        return len(self.y)

    def series(self) -> pd.Series:
        return pd.Series(self.y, index=self.index, name=self.name)


def infer_period(freq: str | None) -> int:
    """The usual seasonal period of a pandas frequency alias (monthly 12, quarterly 4, daily 7…); 1 when there is none."""
    if not freq:
        return 1
    base = freq.split("-")[0]
    if base in _YEARLY:
        return 1
    return _PERIODS.get(base, 1)


def load_series(
    frame: pd.DataFrame, roles: dict[str, Any], *, period: int = 0, minimum: int = 10, name: str = "series"
) -> TS:
    """Sort by the time column (when given), trim missing values at the ends and refuse gaps inside."""
    column = roles[name]
    time_name = roles.get("time")
    data = pd.DataFrame({"y": pd.to_numeric(frame[column], errors="coerce").astype(float)})
    if time_name:
        data["t"] = frame[time_name]
        data = data.dropna(subset=["t"]).sort_values("t", kind="stable")
        if data["t"].duplicated().any():
            raise InvalidInputError(f"La columna de tiempo «{time_name}» tiene fechas repetidas.")
    values = data["y"]
    valid = values.notna().to_numpy()
    if not valid.any():
        raise InvalidInputError(f"«{column}» no tiene valores numéricos.")
    first, last = int(np.argmax(valid)), len(valid) - int(np.argmax(valid[::-1]))
    dropped = len(values) - (last - first)
    data = data.iloc[first:last]
    rows = data.index
    if data["y"].isna().any():
        raise InvalidInputError(
            f"«{column}» tiene {int(data['y'].isna().sum())} valor(es) faltante(s) dentro de la serie. "
            "Reemplácelos antes (Transformar datos → Valores faltantes → interpolación) para no distorsionar el análisis."
        )
    if len(data) < minimum:
        raise InvalidInputError(f"La serie necesita al menos {minimum} observaciones válidas (tiene {len(data)}).")
    freq = None
    index: pd.Index = pd.RangeIndex(1, len(data) + 1)
    labels: list[Any] = list(range(1, len(data) + 1))
    dated = irregular = False
    if time_name:
        stamps = pd.DatetimeIndex(data["t"])
        freq = pd.infer_freq(stamps) if len(stamps) >= 3 else None
        if freq:
            index, dated = pd.DatetimeIndex(stamps, freq=freq), True
            labels = [str(s.date()) if s == s.normalize() else str(s) for s in stamps]
        else:  # irregular dates: the analysis uses the order of the observations
            irregular = True
    seasonal = int(period) if period and int(period) > 1 else infer_period(freq)
    return TS(column, data["y"].to_numpy(float), index, freq, seasonal, labels, dropped, dated, irregular, rows)


def future_labels(ts: TS, horizon: int) -> list[Any]:
    if ts.dated and ts.freq:
        future = pd.date_range(ts.index[-1], periods=horizon + 1, freq=ts.freq)[1:]
        return [str(s.date()) if s == s.normalize() else str(s) for s in future]
    return list(range(ts.n + 1, ts.n + horizon + 1))


def need_cycles(ts: TS, cycles: int = 2) -> None:
    if ts.period < 2:
        raise InvalidInputError(
            "Indique el periodo estacional (12 en datos mensuales, 4 en trimestrales, 7 en diarios)."
        )
    if ts.n < cycles * ts.period:
        raise InvalidInputError(
            f"La serie necesita al menos {cycles} ciclos completos ({cycles * ts.period} observaciones) para ese periodo."
        )


def accuracy(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray, period: int = 1) -> dict[str, float]:
    """MAE, RMSE, MAPE (%) and MASE of ``predicted`` against ``actual``; MASE scales by the in-sample seasonal naive error."""
    error = np.asarray(actual, float) - np.asarray(predicted, float)
    nonzero = np.asarray(actual) != 0
    lag = period if period > 1 and len(train) > period else 1
    scale = float(np.mean(np.abs(np.diff(train, lag)))) if len(train) > lag else np.nan
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(100 * np.mean(np.abs(error[nonzero] / np.asarray(actual)[nonzero])))
        if nonzero.any()
        else float("nan"),
        "mase": float(np.mean(np.abs(error)) / scale) if scale and np.isfinite(scale) and scale > 0 else float("nan"),
    }


ACCURACY_COLUMNS = ["MAE (MAD)", "RMSE", "MAPE %", "MASE"]


def accuracy_row(label: str, values: dict[str, float]) -> list[Any]:
    return [label, values["mae"], values["rmse"], values["mape"], values["mase"]]


def ljung_box(residuals: np.ndarray, lags: int, fitted_params: int = 0) -> tuple[float, float, int]:
    """Ljung-Box Q at ``lags`` with the degrees of freedom reduced by the fitted ARMA parameters: (Q, p, df)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lags = max(1, min(lags, len(residuals) // 2 - 1))
    table = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    q = float(table["lb_stat"].iloc[0])
    df = max(lags - fitted_params, 1)
    return q, float(stats.chi2.sf(q, df)), df


def forecast_rows(labels: list[Any], mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[list[Any]]:
    return [[labels[i], float(mean[i]), float(lower[i]), float(upper[i])] for i in range(len(mean))]


def history_and_forecast(
    ts: TS, fitted: np.ndarray | None, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict[str, Any]:
    """A line chart with the series, the fitted values and the forecasts with their prediction band."""
    from sigma.domain.analysis.charts import line_chart

    horizon = len(mean)
    future = future_labels(ts, horizon)
    keep = min(ts.n, max(4 * horizon, 60))
    history = {"name": "Datos", "x": ts.labels[-keep:], "y": ts.y[-keep:]}
    series = [history]
    if fitted is not None:
        series.append({"name": "Ajustado", "x": ts.labels[-keep:], "y": np.asarray(fitted)[-keep:], "dash": "dot"})
    series.append(
        {"name": "Pronóstico", "x": [ts.labels[-1], *future], "y": np.concatenate([[ts.y[-1]], mean]), "dash": "dash"}
    )
    band = {"name": f"Intervalo de predicción {100 * (1 - alpha):g} %", "x": future, "lower": lower, "upper": upper}
    return line_chart("Serie y pronóstico", "Tiempo" if ts.dated else "Observación", ts.name, series, bands=[band])


def fmt_p(value: float) -> P:
    return P(float(value))


def note_trimmed(out: Output, ts: TS) -> None:
    if ts.dropped:
        out.alert("info", f"Se omitieron {ts.dropped} valor(es) faltante(s) al principio o al final de la serie.")
    if ts.irregular:
        out.alert(
            "info",
            "Las fechas no son regulares: el análisis usa el orden de las observaciones, sin periodo calendario.",
        )


@dataclass
class MultiTS:
    """Several series observed together: ``data`` is ``n × k`` in time order with no gaps."""

    names: list[str]
    data: np.ndarray
    labels: list[Any]
    dated: bool
    freq: str | None
    period: int
    dropped: int
    rows: Any = None

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def k(self) -> int:
        return len(self.names)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=self.names)


def load_multi(
    frame: pd.DataFrame, roles: dict[str, Any], *, key: str = "vars", minimum: int = 20, period: int = 0
) -> MultiTS:
    """Several numeric columns in time order; rows incomplete at the ends are trimmed and gaps inside are refused."""
    names = list(roles[key])
    data = frame[names].apply(pd.to_numeric, errors="coerce").astype(float)
    time_name = roles.get("time")
    if time_name:
        data["__t"] = frame[time_name]
        data = data.dropna(subset=["__t"]).sort_values("__t", kind="stable")
        if data["__t"].duplicated().any():
            raise InvalidInputError(f"La columna de tiempo «{time_name}» tiene fechas repetidas.")
    complete = data[names].notna().all(axis=1).to_numpy()
    if not complete.any():
        raise InvalidInputError("No hay filas con todas las series completas.")
    first, last = int(np.argmax(complete)), len(complete) - int(np.argmax(complete[::-1]))
    trimmed = data.iloc[first:last]
    if trimmed[names].isna().any().any():
        raise InvalidInputError(
            "Alguna serie tiene faltantes dentro del periodo común. Reemplácelos antes (Transformar datos → Valores faltantes)."
        )
    if len(trimmed) < minimum:
        raise InvalidInputError(f"Las series necesitan al menos {minimum} observaciones comunes (hay {len(trimmed)}).")
    labels: list[Any] = list(range(1, len(trimmed) + 1))
    freq, dated = None, False
    if time_name:
        stamps = pd.DatetimeIndex(trimmed["__t"])
        freq = pd.infer_freq(stamps) if len(stamps) >= 3 else None
        if freq:
            dated = True
            labels = [str(s.date()) if s == s.normalize() else str(s) for s in stamps]
    seasonal = int(period) if period and int(period) > 1 else infer_period(freq)
    return MultiTS(
        names, trimmed[names].to_numpy(float), labels, dated, freq, seasonal, len(data) - len(trimmed), trimmed.index
    )


def future_labels_for(labels: list[Any], dated: bool, freq: str | None, horizon: int) -> list[Any]:
    if dated and freq:
        last = pd.Timestamp(labels[-1])
        future = pd.date_range(last, periods=horizon + 1, freq=freq)[1:]
        return [str(s.date()) if s == s.normalize() else str(s) for s in future]
    return list(range(len(labels) + 1, len(labels) + horizon + 1))
