"""Time series analyses against statsmodels / arch / scipy and against series with known structure."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, ccf, kpss

from sigma.domain.analysis.series_explore import cross_correlations, fisher_g_test, mann_kendall
from sigma.domain.analysis.ts_core import accuracy, infer_period, load_series
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def series(analysis_id, frame, roles=None, params=None):
    return run("series", analysis_id, frame, roles or {"series": "s", "time": "t"}, params)


def table(output, title_start):
    return next(b for b in output.blocks if b["type"] == "table" and b["title"].startswith(title_start))["rows"]


def value(cell):
    return getattr(cell, "value", cell)


def synthetic(n=144, period=12, seed=3, trend=0.2, amplitude=4.0, noise=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    pattern = amplitude * np.sin(2 * np.pi * t / period)
    return pd.DataFrame(
        {
            "t": pd.date_range("2010-01-01", periods=n, freq="MS"),
            "y": 10 + trend * t + pattern + rng.normal(0, noise, n),
        }
    )


# ------------------------------------------------------------------ loading
def test_the_loader_sorts_trims_and_infers_the_period(frame):
    shuffled = frame.sample(frac=1, random_state=2)
    ts = load_series(shuffled, {"series": "s", "time": "t"})
    assert ts.dated and ts.freq == "MS" and ts.period == 12 and ts.y.tolist() == frame["s"].tolist()
    gap = frame.assign(s=frame["s"].where(~frame.index.isin([0, 1, 239]))).copy()
    trimmed = load_series(gap, {"series": "s", "time": "t"})
    assert trimmed.dropped == 3 and trimmed.n == 237 and trimmed.labels[0] == "2000-03-01"
    inner = frame.assign(s=frame["s"].where(frame.index != 100))
    with pytest.raises(InvalidInputError, match="faltante"):
        load_series(inner, {"series": "s", "time": "t"})
    irregular = frame.drop(index=[5, 6, 7, 40])
    odd = load_series(irregular, {"series": "s", "time": "t"})
    assert odd.irregular and not odd.dated and odd.period == 1 and odd.labels[0] == 1
    assert [infer_period(f) for f in ("MS", "ME", "QS-JAN", "D", "W-SUN", "YE-DEC", None)] == [12, 12, 4, 7, 52, 1, 1]
    with pytest.raises(InvalidInputError, match="repetidas"):
        load_series(pd.concat([frame, frame.head(1)]), {"series": "s", "time": "t"})


def test_accuracy_measures():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = accuracy(np.array([6.0, 8.0]), np.array([6.0, 7.0]), train)
    assert result["mae"] == pytest.approx(0.5) and result["rmse"] == pytest.approx(np.sqrt(0.5))
    assert result["mape"] == pytest.approx(100 * np.mean([0, 1 / 8])) and result["mase"] == pytest.approx(
        0.5
    )  # naive error is 1


# ------------------------------------------------------------------ plots, cross-correlation, spectrum
def test_moving_statistics_and_trend(frame):
    out = series("ts_plot", frame, params={"window": 12})
    assert out.result["trend_slope"] == pytest.approx(np.polyfit(np.arange(1, 241), frame["s"], 1)[0])
    assert out.result["lag1"] == pytest.approx(np.corrcoef(frame["s"][1:], frame["s"][:-1])[0, 1])
    figure = [b for b in out.blocks if b["type"] == "chart"][0]["spec"]
    rolling = pd.Series(frame["s"]).rolling(12, center=True, min_periods=12).mean().dropna()
    assert figure["series"][1]["y"] == pytest.approx(rolling.to_numpy())
    assert sum(1 for b in out.blocks if b["type"] == "chart") == 3  # series, seasonal means, lag plot


def test_cross_correlation_matches_statsmodels_and_finds_the_lead():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = np.r_[np.zeros(3), x[:-3]] + 0.3 * rng.normal(size=300)  # y repeats x three periods later
    lags, r = cross_correlations(x, y, 10)
    reference = ccf(y, x, adjusted=False)[:11]
    assert r[10:] == pytest.approx(reference)  # lags 0…10: x(t) with y(t + k)
    data = pd.DataFrame({"x": x, "y": y})
    out = run("series", "ccf", data, {"x": "x", "y": "y"}, {"max_lag": 10})
    assert out.result["strongest_lag"] == 3 and out.result["strongest_r"] > 0.9
    reverse = run("series", "ccf", data, {"x": "y", "y": "x"}, {"max_lag": 10})
    assert reverse.result["strongest_lag"] == -3
    with pytest.raises(InvalidInputError, match="15 observaciones"):
        run("series", "ccf", data.head(10), {"x": "x", "y": "y"}, {})


def test_differencing_removes_spurious_cross_correlation():
    rng = np.random.default_rng(6)
    data = pd.DataFrame({"a": np.cumsum(rng.normal(size=400)), "b": np.cumsum(rng.normal(size=400))})
    levels = run("series", "ccf", data, {"x": "a", "y": "b"}, {})
    differences = run("series", "ccf", data, {"x": "a", "y": "b"}, {"difference": True})
    assert abs(levels.result["strongest_r"]) > abs(differences.result["strongest_r"]) + 0.15


def test_periodogram_finds_the_hidden_cycle():
    rng = np.random.default_rng(7)
    t = np.arange(200)
    data = pd.DataFrame({"y": 3 * np.sin(2 * np.pi * t / 8) + rng.normal(0, 1, 200)})
    out = run("series", "periodogram", data, {"series": "y"}, {})
    assert out.result["dominant_period"] == pytest.approx(8, rel=0.05) and out.result["fisher_p"] < 1e-6
    noise = run("series", "periodogram", pd.DataFrame({"y": rng.normal(size=200)}), {"series": "y"}, {})
    assert noise.result["fisher_p"] > 0.01
    g, p = fisher_g_test(np.array([5.0] + [0.5] * 9))
    assert g == pytest.approx(5 / 9.5) and 0 <= p <= 1
    assert fisher_g_test(np.array([1.0, 1.0]))[1] == pytest.approx(1.0)
    assert fisher_g_test(np.r_[1.0, np.full(9, 1e-12)])[1] == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------ decomposition, trend, filters
def test_classical_decomposition_matches_statsmodels_and_recovers_the_structure():
    data = synthetic(noise=0.3)
    out = run("series", "decompose", data, {"series": "y", "time": "t"}, {"method": "classical_add"})
    reference = seasonal_decompose(data.set_index("t")["y"], model="additive", period=12)
    components = offered(out, "componentes")
    assert components["tendencia"].to_numpy() == pytest.approx(reference.trend.to_numpy(), nan_ok=True)
    assert components["estacional"].to_numpy() == pytest.approx(reference.seasonal.to_numpy())
    assert out.result["seasonal_strength"] > 0.95 and out.result["trend_strength"] > 0.95
    seasonal_index = [value(r[1]) for r in table(out, "Índices estacionales")]
    assert max(seasonal_index) == pytest.approx(4.0, abs=0.4) and sum(seasonal_index) == pytest.approx(0, abs=1e-8)


def test_stl_components_add_up_and_a_multiplicative_series_needs_positive_values():
    data = synthetic()
    out = run("series", "decompose", data, {"series": "y", "time": "t"}, {"method": "stl"})
    c = offered(out, "componentes")
    assert (c["tendencia"] + c["estacional"] + c["residuo"]).to_numpy() == pytest.approx(data["y"].to_numpy())
    reference = STL(data.set_index("t")["y"], period=12, robust=False).fit()
    assert c["estacional"].to_numpy() == pytest.approx(reference.seasonal.to_numpy())
    assert c["ajustada_estacionalmente"].to_numpy() == pytest.approx((data["y"] - c["estacional"]).to_numpy())
    growth = data.assign(
        y=np.exp(0.01 * np.arange(len(data))) * (1 + 0.2 * np.sin(2 * np.pi * np.arange(len(data)) / 12))
    )
    mult = run("series", "decompose", growth, {"series": "y", "time": "t"}, {"method": "classical_mul"})
    indices = [value(r[1]) for r in table(mult, "Índices estacionales")]
    assert max(indices) == pytest.approx(1.2, abs=0.05) and np.mean(indices) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(InvalidInputError, match="positivos"):
        run(
            "series",
            "decompose",
            data.assign(y=data["y"] - 30),
            {"series": "y", "time": "t"},
            {"method": "classical_mul"},
        )
    with pytest.raises(InvalidInputError, match="periodo estacional"):
        run("series", "decompose", pd.DataFrame({"y": np.arange(50.0)}), {"series": "y"}, {})


def test_trend_models_match_least_squares(frame):
    data = synthetic(trend=0.5, amplitude=0.0, noise=1.0)
    out = run("series", "trend", data, {"series": "y", "time": "t"}, {"model": "linear", "horizon": 6})
    assert out.result["coefficients"] == pytest.approx(np.polyfit(np.arange(1, 145), data["y"], 1)[::-1])
    rows = table(out, "Pronósticos")
    assert (
        len(rows) == 6
        and rows[0][0] == "2022-01-01"
        and rows[0][1] == pytest.approx(out.result["coefficients"][0] + out.result["coefficients"][1] * 145)
    )
    assert all(r[2] < r[1] < r[3] for r in rows)
    growth = data.assign(y=5 * np.exp(0.02 * np.arange(1, 145)) * np.exp(np.random.default_rng(1).normal(0, 0.01, 144)))
    expo = run("series", "trend", growth, {"series": "y", "time": "t"}, {"model": "exponential", "horizon": 0})
    assert expo.result["coefficients"][1] == pytest.approx(0.02, abs=0.001)
    with pytest.raises(InvalidInputError, match="positivos"):
        run("series", "trend", data.assign(y=data["y"] - 40), {"series": "y", "time": "t"}, {"model": "exponential"})


def test_mann_kendall_matches_kendall_tau_and_theil_sen():
    rng = np.random.default_rng(8)
    x = np.arange(60, dtype=float)
    y = 0.4 * x + rng.normal(0, 3, 60)
    s, variance, z = mann_kendall(y)
    tau = stats.kendalltau(x, y)
    n = len(y)
    assert s == pytest.approx(tau.statistic * n * (n - 1) / 2) and variance == pytest.approx(
        n * (n - 1) * (2 * n + 5) / 18
    )
    assert 2 * stats.norm.sf(abs(z)) == pytest.approx(tau.pvalue, rel=0.2)
    out = run("series", "mann_kendall", pd.DataFrame({"y": y}), {"series": "y"}, {})
    sen = stats.theilslopes(y, x + 1)
    assert out.result["sen_slope"] == pytest.approx(sen[0]) and out.result["p"] < 1e-6
    flat = run("series", "mann_kendall", pd.DataFrame({"y": rng.normal(size=60)}), {"series": "y"}, {})
    assert flat.result["p"] > 0.01
    seasonal = run(
        "series", "mann_kendall", synthetic(trend=0.0, amplitude=5.0), {"series": "y", "time": "t"}, {"seasonal": True}
    )
    assert seasonal.result["p"] > 0.01  # a pure seasonal pattern has no trend within each month


def test_filters_match_statsmodels(frame):
    out = series("filters", frame, params={"filter": "hp"})
    cycle, trend = hpfilter(frame["s"].to_numpy(), lamb=129600)
    result = offered(out, "filtrada")
    assert result["tendencia"].to_numpy() == pytest.approx(trend) and result["ciclo"].to_numpy() == pytest.approx(cycle)
    assert (result["tendencia"] + result["ciclo"]).to_numpy() == pytest.approx(frame["s"].to_numpy())
    bk = series("filters", frame, params={"filter": "bk", "lags": 12})
    assert offered(bk, "filtrada")["ciclo"].isna().sum() == 24
    cf = series("filters", frame, params={"filter": "cf", "low": 6, "high": 32})
    assert offered(cf, "filtrada")["ciclo"].notna().all()
    with pytest.raises(InvalidInputError, match="Baxter-King pierde"):
        run("series", "filters", frame.head(20), {"series": "s"}, {"filter": "bk", "lags": 12})


# ------------------------------------------------------------------ smoothing, unit roots, ARIMA, backtest
def test_exponential_smoothing_matches_statsmodels():
    data = synthetic(noise=0.8)
    out = run("series", "smoothing", data, {"series": "y", "time": "t"}, {"method": "hw_add", "horizon": 12})
    reference = ETSModel(
        pd.Series(data["y"].to_numpy()), error="add", trend="add", seasonal="add", seasonal_periods=12
    ).fit(disp=False)
    assert out.result["aicc"] == pytest.approx(reference.aicc) and out.result["parameters"][
        "smoothing_level"
    ] == pytest.approx(reference.params[0])
    forecast = reference.get_prediction(start=144, end=155).summary_frame()
    rows = table(out, "Pronósticos")
    assert [r[1] for r in rows] == pytest.approx(forecast["mean"].tolist()) and [r[2] for r in rows] == pytest.approx(
        forecast["pi_lower"].tolist()
    )
    assert rows[-1][3] - rows[-1][2] > rows[0][3] - rows[0][2]  # uncertainty grows with the horizon


def test_automatic_ets_picks_seasonality_only_when_it_exists():
    seasonal = run(
        "series", "smoothing", synthetic(noise=0.6), {"series": "y", "time": "t"}, {"method": "auto", "horizon": 0}
    )
    assert seasonal.result["model"].endswith(("A)", "M)")) and not seasonal.result["model"].endswith(",N)")
    rng = np.random.default_rng(9)
    walk = pd.DataFrame(
        {"t": pd.date_range("2010-01-01", periods=120, freq="MS"), "y": 50 + np.cumsum(rng.normal(size=120))}
    )
    plain = run("series", "smoothing", walk, {"series": "y", "time": "t"}, {"method": "auto", "horizon": 3})
    assert plain.result["model"].endswith(",N)")
    held = run(
        "series", "smoothing", walk, {"series": "y", "time": "t"}, {"method": "ses", "horizon": 0, "holdout": 12}
    )
    assert any(b["type"] == "table" and b["title"].startswith("Validación") for b in held.blocks)
    with pytest.raises(InvalidInputError, match="periodo estacional"):
        run("series", "smoothing", pd.DataFrame({"y": rng.normal(size=60)}), {"series": "y"}, {"method": "hw_add"})


def test_unit_root_tests_agree_with_the_libraries_and_tell_a_walk_from_noise():
    rng = np.random.default_rng(10)
    walk = np.cumsum(rng.normal(size=300))
    stationary = np.zeros(300)
    for i in range(1, 300):
        stationary[i] = 0.5 * stationary[i - 1] + rng.normal()
    out = run(
        "series",
        "unit_root",
        pd.DataFrame({"y": walk}),
        {"series": "y"},
        {"tests": ["adf", "pp", "kpss"], "differences": "01"},
    )
    levels = table(out, "Serie en niveles")
    adf_ref = adfuller(walk, regression="c", autolag="AIC")
    assert levels[0][2] == pytest.approx(adf_ref[0]) and value(levels[0][3]) == pytest.approx(adf_ref[1])
    assert levels[1][2] == pytest.approx(PhillipsPerron(walk, trend="c").stat)
    assert levels[2][2] == pytest.approx(kpss(walk, regression="c", nlags="auto")[0])
    assert (
        out.result["integration_order"] == 1
        and out.result["verdicts"]["0"] == [False, False, False]
        and out.result["verdicts"]["1"] == [True, True, True]
    )
    noise = run(
        "series",
        "unit_root",
        pd.DataFrame({"y": stationary}),
        {"series": "y"},
        {"tests": ["adf", "pp", "dfgls", "kpss", "za"], "differences": "0"},
    )
    assert noise.result["integration_order"] == 0
    with pytest.raises(InvalidInputError, match="al menos una"):
        run("series", "unit_root", pd.DataFrame({"y": walk}), {"series": "y"}, {"tests": []})


def test_arima_recovers_an_autoregressive_process():
    rng = np.random.default_rng(12)
    n = 400
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    data = pd.DataFrame({"y": x + 20})
    manual = run("series", "arima", data, {"series": "y"}, {"mode": "manual", "p": 1, "d": 0, "q": 0, "horizon": 5})
    reference = SARIMAX(x + 20, order=(1, 0, 0), trend="c").fit(disp=False)
    assert manual.result["parameters"]["ar.L1"] == pytest.approx(reference.params[1], rel=1e-4) and manual.result[
        "aicc"
    ] == pytest.approx(reference.aicc)
    assert manual.result["parameters"]["ar.L1"] == pytest.approx(0.7, abs=0.08) and manual.result["ljung_box_p"] > 0.05
    assert len(table(manual, "Pronósticos")) == 5
    auto = run("series", "arima", data, {"series": "y"}, {"mode": "auto", "max_order": 2, "horizon": 0})
    assert auto.result["order"][0] >= 1 and auto.result["order"][1] == 0
    assert any(b["type"] == "table" and b["title"].startswith("Mejores modelos") for b in auto.blocks)


def test_seasonal_arima_matches_the_airline_style_specification(frame):
    out = series(
        "arima", frame, params={"mode": "manual", "p": 0, "d": 1, "q": 1, "sp": 0, "sd": 1, "sq": 1, "horizon": 6}
    )
    reference = SARIMAX(frame["s"].to_numpy(), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    assert out.result["aicc"] == pytest.approx(reference.aicc) and out.result["parameters"][
        "ma.S.L12"
    ] == pytest.approx(reference.params[1], rel=1e-4)
    assert offered(out, "ajuste")["residuo"].notna().all()
    with pytest.raises(InvalidInputError, match="valores futuros"):
        series("arima", frame, {"series": "s", "time": "t", "exog": ["x1"]}, {"mode": "manual", "horizon": 3})
    with_exog = series(
        "arima",
        frame,
        {"series": "s", "time": "t", "exog": ["x1"]},
        {"mode": "manual", "p": 1, "d": 1, "q": 0, "horizon": 0},
    )
    assert "x1" in with_exog.result["parameters"]


def test_backtest_ranks_methods_and_checks_the_naive_error(frame):
    out = series(
        "backtest", frame, params={"methods": ["naive", "mean", "drift"], "horizon": 6, "origins": 4, "step": 6}
    )
    cuts = [240 - 6 - (3 - i) * 6 for i in range(4)]
    y = frame["s"].to_numpy()
    naive_errors = np.concatenate([y[c : c + 6] - y[c - 1] for c in cuts])
    rows = {r[0] if not hasattr(r[0], "value") else r[0].value: r for r in table(out, "Error de pronóstico")}
    assert rows["Ingenuo (último valor)"][2] == pytest.approx(np.sqrt(np.mean(naive_errors**2)))
    assert out.result["ranking"][-1] == "Media histórica"  # a trending series is badly served by its mean
    assert out.result["best"] in ("Con deriva (extrapola la pendiente)", "Ingenuo (último valor)")
    skipped = run(
        "series",
        "backtest",
        pd.DataFrame({"y": np.arange(60.0) + np.random.default_rng(1).normal(size=60)}),
        {"series": "y"},
        {"methods": ["naive", "snaive"], "origins": 2, "horizon": 4},
    )
    assert any(b["type"] == "alert" and "No se evaluaron" in b["message"] for b in skipped.blocks)
    with pytest.raises(InvalidInputError, match="datos suficientes"):
        series("backtest", frame.head(40), params={"methods": ["naive"], "origins": 10, "horizon": 12})


def test_ols_trend_on_the_index_is_what_the_exact_series_helper_returns(frame):
    ts = load_series(frame, {"series": "s", "time": "t"})
    fit = sm.OLS(ts.y, sm.add_constant(np.arange(1, ts.n + 1))).fit()
    assert fit.params[1] > 0 and ts.series().index.freqstr == "MS"
