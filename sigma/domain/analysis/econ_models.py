"""Single-equation and panel econometrics: GARCH, ARDL bounds test, panel data, structural breaks, regime switching, state space."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ardl import UECM, ardl_select_order
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import bar_chart, line_chart, panels_chart
from sigma.domain.analysis.econ_multi import FIT_ERRORS
from sigma.domain.analysis.modeling import build_design
from sigma.domain.analysis.output import Output, P, good
from sigma.domain.analysis.series_explore import ALPHA, PERIOD, SERIES, TIME, _series_code
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.analysis.ts_core import (
    TS,
    future_labels,
    history_and_forecast,
    ljung_box,
    load_multi,
    load_series,
    note_trimmed,
)
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "econometria"
GROUP_VOL = "Volatilidad y regímenes"
GROUP_DYN = "Modelos dinámicos"
GROUP_PANEL = "Datos de panel"
GROUP_BREAKS = "Cambio estructural"


# ============================================================================ GARCH family
_VARIANTS = (
    ("garch", "GARCH(p, q)"),
    ("gjr", "GJR-GARCH (efecto apalancamiento)"),
    ("egarch", "EGARCH (logarítmico, asimétrico)"),
    ("tarch", "TARCH / Zakoian (desviación estándar)"),
    ("aparch", "APARCH"),
    ("arch", "ARCH(q)"),
)
_DISTRIBUTIONS = (
    ("normal", "Normal"),
    ("t", "t de Student"),
    ("skewt", "t asimétrica"),
    ("ged", "Error generalizado (GED)"),
)
_PARAMETER_LABELS = {
    "mu": "μ (media)",
    "omega": "ω (constante)",
    "nu": "ν (grados de libertad)",
    "lambda": "λ (asimetría de la t)",
    "eta": "η (grados de libertad)",
    "lambda_": "λ",
}


def _label(name: str) -> str:
    if name in _PARAMETER_LABELS:
        return _PARAMETER_LABELS[name]
    for key, text in (
        ("alpha", "α (ARCH)"),
        ("gamma", "γ (asimetría)"),
        ("beta", "β (GARCH)"),
        ("delta", "δ (potencia)"),
        ("Const", "constante"),
        ("y[", "AR "),
    ):
        if name.startswith(key):
            return f"{text} {name[len(key) :]}" if key in ("alpha", "gamma", "beta") else name
    return name


def kupiec(hits: int, n: int, level: float) -> tuple[float, float]:
    """Kupiec's proportion-of-failures test of a VaR at tail probability ``level``: (LR, p-value)."""

    def loglik(probability: float) -> float:
        misses = (n - hits) * np.log(1 - probability) if n - hits else 0.0
        return float(misses + (hits * np.log(probability) if hits else 0.0))

    lr = max(-2 * (loglik(level) - loglik(hits / n)), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def _garch(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, minimum=100)
    values = ts.y
    if p["input"] == "prices":
        if (values <= 0).any():
            raise InvalidInputError("Para pasar de precios a rendimientos los precios deben ser positivos.")
        values = np.diff(np.log(values))
        labels = ts.labels[1:]
    else:
        labels = ts.labels
    y = values * 100 if p["percent"] else values
    if np.std(y) == 0:
        raise InvalidInputError("La serie es constante.")
    variant, dist, horizon, level = p["variant"], p["distribution"], int(p["horizon"]), float(p["var_level"])
    kwargs: dict[str, Any] = {"p": int(p["p"]), "q": int(p["q"]), "o": 0}
    if variant in ("gjr", "tarch", "egarch", "aparch"):
        kwargs["o"] = max(int(p["o"]), 1)
    if variant == "tarch":
        kwargs["power"] = 1.0
    vol = {"garch": "GARCH", "gjr": "GARCH", "tarch": "GARCH", "egarch": "EGARCH", "aparch": "APARCH", "arch": "ARCH"}[
        variant
    ]
    if variant == "arch":
        kwargs = {"p": int(p["q"])}
    mean = {"constant": "Constant", "zero": "Zero", "ar": "AR"}[p["mean"]]
    model = arch_model(y, mean=mean, lags=int(p["ar_lags"]) if mean == "AR" else 0, vol=vol, dist=dist, **kwargs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", update_freq=0, options={"maxiter": 500})
    except FIT_ERRORS + (RuntimeError,) as error:
        raise ComputationError(f"No se pudo ajustar el modelo de volatilidad: {error}") from None
    n = len(y)
    names = list(result.params.index)
    alpha_ci = p["alpha"]
    ci = result.conf_int(alpha=alpha_ci)
    persistence = None
    group = lambda prefix: sum(float(result.params[k]) for k in names if k.startswith(prefix))  # noqa: E731
    if variant in ("garch", "gjr", "arch"):
        persistence = group("alpha") + group("beta") + (0.5 * group("gamma") if variant == "gjr" else 0.0)
    elif variant == "egarch":
        persistence = group("beta")
    uncond = (
        float(result.params["omega"] / (1 - persistence))
        if variant in ("garch", "gjr", "arch") and persistence is not None and persistence < 1
        else None
    )
    half_life = float(np.log(0.5) / np.log(persistence)) if persistence is not None and 0 < persistence < 1 else None
    std_resid = np.asarray(result.std_resid, float)
    std_resid = std_resid[np.isfinite(std_resid)]
    lb = ljung_box(std_resid, 10)
    lb2 = ljung_box(std_resid**2, 10)
    arch_lm = het_arch(std_resid, nlags=5)
    raw_lm = het_arch(y - y.mean(), nlags=5)
    jb = jarque_bera(std_resid)
    out = Output(
        f"Modelo de volatilidad {dict(_VARIANTS)[variant].split(' (')[0]}: {ts.name}",
        f"distribución {dict(_DISTRIBUTIONS)[dist].lower()} · n = {n}"
        + (" · rendimientos logarítmicos ×100 a partir de precios" if p["input"] == "prices" else ""),
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Observaciones", n),
            ("Log-verosimilitud", float(result.loglikelihood)),
            ("AIC", float(result.aic)),
            ("BIC", float(result.bic)),
            ("Persistencia", persistence if persistence is not None else "—"),
        ]
    )
    out.table(
        "Parámetros",
        ["Parámetro", "Estimación", "Error est.", "t", "Valor p", f"IC {100 * (1 - alpha_ci):g} % inf.", "IC sup."],
        [
            [
                _label(k),
                float(result.params[k]),
                float(result.std_err[k]),
                float(result.tvalues[k]),
                P(float(result.pvalues[k])),
                float(ci.loc[k].iloc[0]),
                float(ci.loc[k].iloc[1]),
            ]
            for k in names
        ],
    )
    facts = [
        ["Persistencia", persistence],
        ["Vida media de un choque (periodos)", half_life],
        ["Varianza incondicional", uncond],
        ["Volatilidad incondicional", float(np.sqrt(uncond)) if uncond else None],
    ]
    out.table(
        "Persistencia de la volatilidad",
        ["Medida", "Valor"],
        [r for r in facts if r[1] is not None] or [["Persistencia", None]],
        note=(
            "La persistencia cercana a 1 indica que los choques de volatilidad tardan mucho en disiparse; ≥ 1 sería no estacionario (IGARCH)."
            if persistence is not None
            else None
        ),
    )
    out.table(
        "Diagnóstico",
        ["Prueba", "Estadístico", "Valor p", "Lectura"],
        [
            [
                "Efectos ARCH en la serie (LM, 5 rezagos)",
                float(raw_lm[0]),
                P(float(raw_lm[1])),
                "hay heterocedasticidad condicional: un modelo de volatilidad tiene sentido"
                if raw_lm[1] < 0.05
                else "poca evidencia de efectos ARCH",
            ],
            [
                "Ljung-Box de los residuos estandarizados",
                lb[0],
                P(lb[1]),
                "la media está bien especificada" if lb[1] >= 0.05 else "queda autocorrelación en la media",
            ],
            [
                "Ljung-Box de los residuos² estandarizados",
                lb2[0],
                P(lb2[1]),
                "la varianza está bien modelada"
                if lb2[1] >= 0.05
                else "queda heterocedasticidad: aumente p o q, o cambie de modelo",
            ],
            [
                "ARCH-LM de los residuos estandarizados",
                float(arch_lm[0]),
                P(float(arch_lm[1])),
                "sin efectos ARCH residuales" if arch_lm[1] >= 0.05 else "aún hay efectos ARCH",
            ],
            [
                "Jarque-Bera de los residuos estandarizados",
                float(jb[0]),
                P(float(jb[1])),
                "compatible con normalidad" if jb[1] >= 0.05 else "colas pesadas: considere la distribución t",
            ],
        ],
    )
    volatility = np.asarray(result.conditional_volatility, float)
    x = np.asarray(labels, dtype=object)
    out.chart(
        line_chart(
            "Volatilidad condicional estimada",
            "Tiempo" if ts.dated else "Observación",
            "Desviación estándar condicional",
            [{"name": "σ(t)", "x": x, "y": volatility}],
        )
    )
    mean_t = y - np.asarray(result.resid, float)
    parameters = model.distribution.parameter_names()
    quantile = (
        float(model.distribution.ppf(np.array([level]), np.asarray(result.params[parameters], float))[0])
        if parameters
        else float(stats.norm.ppf(level))
    )
    var_line = -(mean_t + volatility * quantile)
    hits = int((y < -var_line).sum())
    lr, pk = kupiec(hits, n, level)
    out.table(
        f"Valor en riesgo al {100 * (1 - level):g} % (1 día)",
        ["Violaciones", "Esperadas", "Tasa observada", "Kupiec LR", "Valor p"],
        [[hits, float(level * n), hits / n, lr, P(pk)]],
        note="VaR = −(media + σ(t)·cuantil); una violación es un rendimiento peor que −VaR. Kupiec contrasta que la tasa de violaciones sea igual a la nominal."
        + (" Los rendimientos están en %." if p["percent"] else ""),
    )
    out.chart(
        line_chart(
            "Rendimientos y valor en riesgo",
            "Tiempo" if ts.dated else "Observación",
            "Rendimiento",
            [{"name": "Rendimiento", "x": x, "y": y}, {"name": "−VaR", "x": x, "y": -var_line, "dash": "dash"}],
        )
    )
    if horizon:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variance = np.asarray(result.forecast(horizon=horizon, reindex=False).variance.iloc[-1], float)
        future = future_labels(
            TS(ts.name, np.zeros(n), ts.index, ts.freq, ts.period, list(labels), 0, ts.dated), horizon
        )
        out.table(
            "Pronóstico de la volatilidad",
            ["Periodo", "Varianza", "Desviación estándar"],
            [[future[i], float(variance[i]), float(np.sqrt(variance[i]))] for i in range(horizon)],
        )
    stamps = pd.Series(labels)
    out.dataset(
        "volatilidad",
        "Volatilidad condicional",
        pd.DataFrame(
            {
                "tiempo": stamps.to_numpy(),
                "rendimiento": y,
                "volatilidad": volatility,
                "residuo_estandarizado": np.asarray(result.std_resid, float),
                "VaR": var_line,
            }
        ),
    )
    return out.set(
        n=n,
        persistence=persistence,
        half_life=half_life,
        aic=float(result.aic),
        bic=float(result.bic),
        parameters={k: float(result.params[k]) for k in names},
        lm_p=float(raw_lm[1]),
        residual_lm_p=float(arch_lm[1]),
        var_hits=hits,
        kupiec_p=pk,
    ).build()


def _garch_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    o = max(int(p["o"]), 1) if p["variant"] in ("gjr", "tarch", "egarch", "aparch") else 0
    vol = {"garch": "GARCH", "gjr": "GARCH", "tarch": "GARCH", "egarch": "EGARCH", "aparch": "APARCH", "arch": "ARCH"}[
        p["variant"]
    ]
    power = ", power=1.0" if p["variant"] == "tarch" else ""
    return "from arch import arch_model\n\n" + _series_code(
        roles,
        dataset,
        f"y = y * {100 if p['percent'] else 1}\nres = arch_model(y, mean={p['mean'].title()!r}, vol={vol!r}, p={int(p['p'])}, o={o}, q={int(p['q'])}{power}, dist={p['distribution']!r}).fit(disp='off')\nprint(res.summary())\n",
    )


# ============================================================================ ARDL and the bounds test
_CASES = (
    ("1", "Caso 1: sin constante ni tendencia"),
    ("3", "Caso 3: constante sin restringir"),
    ("5", "Caso 5: constante y tendencia sin restringir"),
)
_CASE_TREND = {"1": "n", "3": "c", "5": "ct"}


def _ardl(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    mts = load_multi(frame, {"vars": [roles["y"], *roles["x"]], "time": roles.get("time")}, minimum=40)
    names = mts.names
    y_name, x_names = names[0], names[1:]
    data = mts.frame()
    data.index = pd.RangeIndex(mts.n)
    case = p["case"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selection = ardl_select_order(
                data[y_name],
                int(p["max_lag_y"]),
                data[x_names],
                int(p["max_lag_x"]),
                trend=_CASE_TREND[case],
                ic=p["ic"],
            )
    except FIT_ERRORS as error:
        raise ComputationError(f"No se pudo seleccionar el orden del ARDL: {error}") from None
    model = selection.model
    if not model.dl_lags:
        raise InvalidInputError(
            "El criterio de información no incluyó ninguna variable explicativa: no hay relación dinámica entre las series."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit()
        uecm = UECM.from_ardl(model).fit()
        bounds = uecm.bounds_test(case=int(case))
    order = f"ARDL({', '.join(str(v) for v in model.ardl_order)})"
    out = Output(
        f"ARDL y prueba de límites: {y_name}",
        f"{order} elegido por {p['ic'].upper()} · {dict(_CASES)[case].lower()} · n = {result.nobs}",
    )
    out.tiles(
        [
            ("Modelo", order),
            ("Observaciones efectivas", int(result.nobs)),
            ("R²", float(result.rsquared)),
            ("Estadístico F de límites", float(bounds.statistic)),
        ]
    )
    z = stats.t.ppf(1 - p["alpha"] / 2, result.df_resid)
    out.table(
        "Coeficientes del ARDL",
        ["Término", "Coeficiente", "Error est.", "t", "Valor p", f"IC {100 * (1 - p['alpha']):g} % inf.", "IC sup."],
        [
            [
                name,
                float(result.params[name]),
                float(result.bse[name]),
                float(result.tvalues[name]),
                P(float(result.pvalues[name])),
                float(result.params[name] - z * result.bse[name]),
                float(result.params[name] + z * result.bse[name]),
            ]
            for name in result.params.index
        ],
        note="Los nombres indican la variable y su rezago (x.L1 = un periodo atrás).",
    )
    critical = bounds.critical_values
    rows = [[f"{level:g} %", float(row["lower"]), float(row["upper"])] for level, row in critical.iterrows()]
    stat = float(bounds.statistic)
    out.table(
        "Prueba de límites de Pesaran-Shin-Smith (H₀: no hay relación de largo plazo)",
        ["Nivel de significancia", "Límite inferior I(0)", "Límite superior I(1)"],
        rows,
        note=f"Estadístico F = {stat:.3f}. Por encima del límite superior se rechaza H₀ (hay cointegración); por debajo del inferior no; entre ambos la prueba no concluye.",
    )
    five = critical.loc[95.0] if 95.0 in critical.index else critical.iloc[1]
    if stat > float(five["upper"]):
        out.alert(
            "success",
            f"F = {stat:.2f} supera el límite superior al 5 % ({float(five['upper']):.2f}): hay una relación de largo plazo entre las series.",
        )
    elif stat < float(five["lower"]):
        out.alert(
            "warning",
            f"F = {stat:.2f} es menor que el límite inferior al 5 % ({float(five['lower']):.2f}): no hay evidencia de relación de largo plazo.",
        )
    else:
        out.alert(
            "info",
            f"F = {stat:.2f} cae entre los límites al 5 % ({float(five['lower']):.2f} y {float(five['upper']):.2f}): la prueba no concluye; determine antes el orden de integración.",
        )
    phi = sum(float(v) for k, v in result.params.items() if k.startswith(f"{y_name}.L"))
    long_run = []
    params, cov = np.asarray(result.params), np.asarray(result.cov_params())
    index = list(result.params.index)
    for x_name in x_names:
        positions = [i for i, k in enumerate(index) if k.startswith(f"{x_name}.L")]
        ar_positions = [i for i, k in enumerate(index) if k.startswith(f"{y_name}.L")]
        if not positions:
            continue
        gradient = np.zeros(len(params))
        beta_sum, denominator = params[positions].sum(), 1 - params[ar_positions].sum()
        gradient[positions] = 1 / denominator
        gradient[ar_positions] = beta_sum / denominator**2
        theta, se = beta_sum / denominator, float(np.sqrt(gradient @ cov @ gradient))
        long_run.append([x_name, float(theta), se, float(theta / se), P(float(2 * stats.norm.sf(abs(theta / se))))])
    out.table(
        "Multiplicadores de largo plazo (método delta)",
        ["Variable", "Efecto de largo plazo", "Error est.", "z", "Valor p"],
        long_run,
        note="Efecto acumulado de X sobre Y cuando el sistema vuelve al equilibrio: Σβ / (1 − Σφ).",
    )
    speed_name = next(k for k in uecm.params.index if k == f"{y_name}.L1")
    speed = float(uecm.params[speed_name])
    out.table(
        "Corrección de errores (de la forma UECM)",
        ["Coeficiente de ajuste", "Error est.", "Valor p"],
        [[speed, float(uecm.bse[speed_name]), P(float(uecm.pvalues[speed_name]))]],
        note="Debe ser negativo y significativo: la fracción de la desviación del equilibrio que se corrige cada periodo."
        if speed < 0
        else "Un coeficiente de ajuste no negativo contradice la corrección hacia el equilibrio.",
    )
    out.chart(
        line_chart(
            "Serie y valores ajustados",
            "Tiempo" if mts.dated else "Observación",
            y_name,
            [
                {
                    "name": "Datos",
                    "x": mts.labels[-len(result.fittedvalues) :],
                    "y": data[y_name].to_numpy()[-len(result.fittedvalues) :],
                },
                {
                    "name": "Ajustado",
                    "x": mts.labels[-len(result.fittedvalues) :],
                    "y": np.asarray(result.fittedvalues, float),
                    "dash": "dash",
                },
            ],
        )
    )
    return out.set(
        order=list(model.ardl_order),
        f_bounds=stat,
        bounds_p_upper=float(bounds.pvalue["upper"]),
        long_run={r[0]: r[1] for r in long_run},
        speed_of_adjustment=speed,
        ar_sum=phi,
    ).build()


def _ardl_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.tsa.ardl import UECM, ardl_select_order\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['y'])}, *{roles['x']!r}]].dropna().reset_index(drop=True)\n"
        + f"sel = ardl_select_order(d[{py_name(roles['y'])}], {int(p['max_lag_y'])}, d[{roles['x']!r}], {int(p['max_lag_x'])}, trend={_CASE_TREND[p['case']]!r}, ic={p['ic']!r})\n"
        + f"print(UECM.from_ardl(sel.model).fit().bounds_test(case={int(p['case'])}))\n"
    )


# ============================================================================ panel data
_PANEL_COV = (
    ("clustered", "Agrupados por individuo (recomendado)"),
    ("robust", "Robustos a heterocedasticidad"),
    ("unadjusted", "Clásicos"),
    ("kernel", "Driscoll-Kraay (dependencia entre individuos)"),
)


def breusch_pagan_lm(residuals: np.ndarray, entities: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan LM test of random effects from pooled residuals (unbalanced version): (LM, p-value)."""
    frame = pd.DataFrame({"e": residuals, "g": entities})
    s1 = float((frame.groupby("g")["e"].sum() ** 2).sum())
    s2 = float((frame["e"] ** 2).sum())
    sizes = frame.groupby("g").size().to_numpy(float)
    n = float(len(frame))
    lm = (n**2) * (s1 / s2 - 1) ** 2 / (2 * float((sizes * (sizes - 1)).sum()))
    return float(lm), float(stats.chi2.sf(lm, 1))


def _panel(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    from linearmodels.panel.results import PanelResults  # noqa: F401 - documents the type of the fits

    response, predictors, entity, time = roles["response"], roles["predictors"], roles["entity"], roles["time"]
    design = build_design(frame, response, predictors)
    keys = frame.loc[design.used, [entity, time]]
    if keys.duplicated().any():
        raise InvalidInputError(
            f"Hay combinaciones repetidas de «{entity}» y «{time}»: cada individuo solo puede tener una observación por periodo."
        )
    rhs = design.formula.split("~", 1)[1]
    x = patsy.dmatrix(rhs, design.data, return_type="dataframe").drop(columns="Intercept", errors="ignore")
    x.columns = [design.pretty(str(c)) for c in x.columns]
    moments = keys[time]
    if not (pd.api.types.is_numeric_dtype(moments) or pd.api.types.is_datetime64_any_dtype(moments)):
        order = {
            value: i for i, value in enumerate(sorted(moments.unique(), key=str))
        }  # text periods are ordered by name
        moments = moments.map(order)
    index = pd.MultiIndex.from_arrays(
        [keys[entity].astype(str).to_numpy(), moments.to_numpy()], names=["entidad", "tiempo"]
    )
    y = pd.Series(design.data[design.response].to_numpy(), index=index, name=response)
    x.index = index
    entities = keys[entity].astype(str).to_numpy()
    counts = pd.Series(entities).value_counts()
    if len(counts) < 3 or counts.max() < 2:
        raise InvalidInputError("Un panel necesita al menos 3 individuos con 2 o más periodos cada uno.")
    cov_kwargs = {
        "clustered": {"cov_type": "clustered", "cluster_entity": True},
        "robust": {"cov_type": "robust"},
        "unadjusted": {"cov_type": "unadjusted"},
        "kernel": {"cov_type": "kernel"},
    }[p["cov"]]
    with_const = sm.add_constant(x, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled = PooledOLS(y, with_const).fit(**cov_kwargs)
            fixed = PanelOLS(y, x, entity_effects=True, time_effects=bool(p["time_effects"])).fit(**cov_kwargs)
            random = RandomEffects(y, with_const).fit(**cov_kwargs)
    except Exception as error:  # noqa: BLE001 - linearmodels raises several types for absorbed or collinear regressors
        raise ComputationError(
            f"No se pudo ajustar el modelo de panel ({error}). Un regresor que no varía en el tiempo no puede estimarse con efectos fijos."
        ) from None
    n, groups = len(y), len(counts)
    out = Output(
        f"Datos de panel: {response}",
        f"{groups} individuos · {n} observaciones · errores {dict(_PANEL_COV)[p['cov']].split(' (')[0].lower()}"
        + (" · efectos de tiempo incluidos" if p["time_effects"] else ""),
    )
    if design.dropped:
        out.alert("info", f"Se omitieron {design.dropped} fila(s) con datos faltantes.")
    balanced = counts.nunique() == 1
    out.tiles(
        [
            ("Individuos", groups),
            ("Observaciones", n),
            ("Panel", "balanceado" if balanced else "no balanceado"),
            ("R² dentro (efectos fijos)", float(fixed.rsquared_within)),
        ]
    )
    rows = []
    for term in with_const.columns:
        row: list[Any] = [term]
        for fit in (pooled, fixed, random):
            if term in fit.params.index:
                row += [float(fit.params[term]), float(fit.std_errors[term]), P(float(fit.pvalues[term]))]
            else:
                row += [None, None, None]
        rows.append(row)
    out.table(
        "Coeficientes de los tres modelos",
        ["Término", "MCO agrupado", "EE", "p", "Efectos fijos", "EE", "p", "Efectos aleatorios", "EE", "p"],
        rows,
        note="Los efectos fijos eliminan toda diferencia constante entre individuos (por eso no tienen constante ni admiten regresores que no cambian en el tiempo).",
    )
    out.table(
        "Bondad de ajuste",
        ["Modelo", "R² dentro", "R² entre", "R² total"],
        [
            [name, float(fit.rsquared_within), float(fit.rsquared_between), float(fit.rsquared_overall)]
            for name, fit in (("MCO agrupado", pooled), ("Efectos fijos", fixed), ("Efectos aleatorios", random))
        ],
    )
    common = [c for c in x.columns if c in random.params.index and c in fixed.params.index]
    # Hausman's statistic is defined with the classical covariances (random effects is efficient under H0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed_classic = PanelOLS(y, x, entity_effects=True, time_effects=bool(p["time_effects"])).fit(
            cov_type="unadjusted"
        )
        random_classic = RandomEffects(y, with_const).fit(cov_type="unadjusted")
    difference = (fixed_classic.params[common] - random_classic.params[common]).to_numpy()
    cov_difference = (
        fixed_classic.cov.loc[common, common].to_numpy() - random_classic.cov.loc[common, common].to_numpy()
    )
    try:
        statistic = float(difference @ np.linalg.solve(cov_difference, difference))
        hausman = (statistic, float(stats.chi2.sf(statistic, len(common))), len(common)) if statistic >= 0 else None
    except np.linalg.LinAlgError:
        hausman = None
    lm = breusch_pagan_lm(np.asarray(pooled.resids), entities)
    tests = [
        [
            "F de poolabilidad (¿hay efectos individuales?)",
            float(fixed.f_pooled.stat),
            P(float(fixed.f_pooled.pval)),
            "sí: los individuos difieren, no use MCO agrupado"
            if fixed.f_pooled.pval < p["alpha"]
            else "no se detectan efectos individuales",
        ],
        [
            "Breusch-Pagan LM (¿efectos aleatorios frente a agrupado?)",
            lm[0],
            P(lm[1]),
            "hay varianza entre individuos" if lm[1] < p["alpha"] else "no hay evidencia de efectos aleatorios",
        ],
    ]
    if hausman:
        tests.append(
            [
                f"Hausman (efectos fijos frente a aleatorios, {hausman[2]} gl)",
                hausman[0],
                P(hausman[1]),
                "prefiera efectos fijos (los efectos aleatorios serían inconsistentes)"
                if hausman[1] < p["alpha"]
                else "efectos aleatorios es consistente y más eficiente",
            ]
        )
    else:
        tests.append(
            [
                "Hausman",
                None,
                None,
                "no calculable (la diferencia de varianzas no es positiva definida): use errores agrupados o la prueba de Mundlak",
            ]
        )
    out.table("Pruebas de especificación", ["Prueba", "Estadístico", "Valor p", "Lectura"], tests)
    theta = float(np.asarray(random.theta).mean())
    out.table(
        "Componentes de la varianza (efectos aleatorios)",
        ["Componente", "Valor"],
        [
            ["Desv. est. entre individuos (σ_u)", float(random.variance_decomposition["Effects"] ** 0.5)],
            ["Desv. est. residual (σ_e)", float(random.variance_decomposition["Residual"] ** 0.5)],
            ["Correlación intraclase (ρ)", float(random.variance_decomposition["Percent due to Effects"])],
            ["θ medio (fracción de la media que se resta)", theta],
        ],
    )
    effects = fixed.estimated_effects.groupby(level=0).mean()["estimated_effects"].sort_values()
    shown = (
        effects.iloc[np.r_[0 : min(10, len(effects)), max(len(effects) - 10, 10) : len(effects)]]
        if len(effects) > 20
        else effects
    )
    out.chart(
        bar_chart(
            "Efectos fijos estimados por individuo (los más bajos y los más altos)",
            "Individuo",
            "Efecto",
            [str(i) for i in shown.index],
            [{"name": "Efecto", "y": shown.to_numpy()}],
        )
    )
    out.dataset(
        "efectos", "Efectos fijos por individuo", effects.rename("efecto_fijo").rename_axis("individuo").reset_index()
    )
    return out.set(
        groups=groups,
        n=n,
        fixed={k: float(v) for k, v in fixed.params.items()},
        random={k: float(v) for k, v in random.params.items()},
        pooled={k: float(v) for k, v in pooled.params.items()},
        hausman_p=hausman[1] if hausman else None,
        poolability_p=float(fixed.f_pooled.pval),
        breusch_pagan_p=lm[1],
    ).build()


def _panel_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from linearmodels.panel import PanelOLS, RandomEffects\nimport statsmodels.api as sm\n\n"
        + load_line(dataset)
        + f"d = df.dropna(subset=[{py_name(roles['response'])}, *{roles['predictors']!r}]).set_index([{py_name(roles['entity'])}, {py_name(roles['time'])}])\n"
        + f"x = d[{roles['predictors']!r}]\nfixed = PanelOLS(d[{py_name(roles['response'])}], x, entity_effects=True{', time_effects=True' if p['time_effects'] else ''}).fit(cov_type={p['cov'] if p['cov'] != 'clustered' else 'clustered'!r}{', cluster_entity=True' if p['cov'] == 'clustered' else ''})\n"
        + f"random = RandomEffects(d[{py_name(roles['response'])}], sm.add_constant(x)).fit()\nprint(fixed.summary)\nprint(random.summary)\n"
    )


# ============================================================================ structural breaks
def _cumulative_moments(x: np.ndarray, y: np.ndarray):
    xtx = np.concatenate([np.zeros((1, x.shape[1], x.shape[1])), np.cumsum(x[:, :, None] * x[:, None, :], axis=0)])
    xty = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x * y[:, None], axis=0)])
    yty = np.concatenate([[0.0], np.cumsum(y**2)])
    return xtx, xty, yty


def _segment(moments, start: int, end: int) -> tuple[float, np.ndarray]:
    """SSR and coefficients of the regression on observations ``start … end − 1``."""
    xtx, xty, yty = moments
    a, b, c = xtx[end] - xtx[start], xty[end] - xty[start], yty[end] - yty[start]
    beta = np.linalg.lstsq(a, b, rcond=None)[0]
    return float(max(c - beta @ b, 0.0)), beta


def sup_f_path(x: np.ndarray, y: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray]:
    """Chow F statistics for every candidate break between ``trim·n`` and ``(1 − trim)·n``: (break positions, F)."""
    n, k = x.shape
    moments = _cumulative_moments(x, y)
    total = _segment(moments, 0, n)[0]
    low, high = max(int(np.ceil(trim * n)), k + 1), min(int(np.floor((1 - trim) * n)), n - k - 1)
    positions = np.arange(low, high + 1)
    split_ssr = np.array([_segment(moments, 0, cut)[0] + _segment(moments, cut, n)[0] for cut in positions])
    f = ((total - split_ssr) / k) / (split_ssr / (n - 2 * k))
    return positions, f


def andrews_p_value(wald: float, k: int, trim: float, reps: int = 4000, seed: int = 20240601) -> tuple[float, float]:
    """Monte Carlo p-value (and 5 % critical value) of the supremum Wald statistic: sup of the squared Bessel-bridge ratio."""
    rng = np.random.default_rng(seed)
    grid, chunk = 600, 500
    pi = np.arange(1, grid + 1) / grid
    inside = (pi >= trim) & (pi <= 1 - trim)
    weight = (pi[inside] * (1 - pi[inside]))[None, :]
    sups = []
    for _ in range(max(reps // chunk, 1)):
        walk = np.cumsum(rng.standard_normal((chunk, grid, k)) / np.sqrt(grid), axis=1)
        bridge = walk - pi[None, :, None] * walk[:, -1:, :]
        sups.append(((bridge[:, inside, :] ** 2).sum(axis=2) / weight).max(axis=1))
    sup = np.concatenate(sups)
    return float((sup >= wald).mean()), float(np.quantile(sup, 0.95))


def _design_for_breaks(frame: pd.DataFrame, roles: dict[str, Any], minimum: int):
    ts = load_series(frame, {"series": roles["series"], "time": roles.get("time")}, minimum=minimum)
    if roles.get("regressors"):
        columns = frame.loc[ts.rows, roles["regressors"]].apply(pd.to_numeric, errors="coerce")
        if columns.isna().any().any():
            raise InvalidInputError("Los regresores no deben tener faltantes en las filas de la serie.")
        x = np.column_stack([np.ones(ts.n), columns.to_numpy(float)])
        names = ["Constante", *roles["regressors"]]
    else:
        x, names = np.ones((ts.n, 1)), ["Constante (media)"]
    return ts, x, names


def _qa_break(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts, x, names = _design_for_breaks(frame, roles, 30)
    trim, alpha = float(p["trim"]), p["alpha"]
    k = x.shape[1]
    if ts.n > 5000:
        raise InvalidInputError("La prueba de Quandt-Andrews admite hasta 5 000 observaciones.")
    positions, f = sup_f_path(x, ts.y, trim)
    if len(positions) < 5:
        raise InvalidInputError("Con ese recorte quedan muy pocos puntos candidatos: reduzca el recorte.")
    best = int(np.argmax(f))
    wald = float(k * f[best])
    pvalue, critical = andrews_p_value(wald, k, trim)
    when = ts.labels[positions[best]]
    out = Output(
        f"Quandt-Andrews: cambio estructural en {ts.name}",
        f"{k} coeficiente(s) · recorte {100 * trim:g} % · n = {ts.n}",
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("sup-F", float(f[best])),
            ("sup-Wald (k·F)", wald),
            ("Valor p", P(pvalue)),
            ("Quiebre más probable", str(when)),
        ]
    )
    out.alert(
        "warning" if pvalue < alpha else "success",
        (
            f"Hay evidencia de un cambio estructural; el punto más probable es la observación {int(positions[best]) + 1} ({when}), p = {pvalue:.4f}."
            if pvalue < alpha
            else f"No hay evidencia de un cambio estructural de ubicación desconocida (p = {pvalue:.4f})."
        ),
    )
    before, after = (
        _segment(_cumulative_moments(x, ts.y), 0, int(positions[best])),
        _segment(_cumulative_moments(x, ts.y), int(positions[best]), ts.n),
    )
    out.table(
        "Coeficientes antes y después del quiebre más probable",
        ["Término", "Antes", "Después", "Diferencia"],
        [[names[i], float(before[1][i]), float(after[1][i]), float(after[1][i] - before[1][i])] for i in range(k)],
    )
    out.table(
        "Prueba",
        ["sup-F", "sup-Wald", "Crítico 5 % (simulado)", "Valor p (simulado)"],
        [[float(f[best]), wald, critical, P(pvalue)]],
        note="El valor p y el crítico se simulan a partir de la distribución asintótica de Andrews (1993), que depende solo del número de coeficientes y del recorte.",
    )
    labels = np.asarray(ts.labels, dtype=object)
    out.chart(
        line_chart(
            "Estadístico F de Chow para cada fecha candidata",
            "Tiempo" if ts.dated else "Observación",
            "F",
            [{"name": "F de Chow", "x": labels[positions], "y": f}],
            hlines=[{"value": critical / k, "label": "crítico 5 %", "dash": "dash"}],
        )
    )
    return out.set(
        sup_f=float(f[best]), sup_wald=wald, p=pvalue, break_position=int(positions[best]) + 1, break_label=str(when)
    ).build()


def _qa_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "import numpy as np\nimport statsmodels.api as sm\n\n" + _series_code(
        roles,
        dataset,
        f"n, trim = len(y), {p['trim']}\nx = np.ones((n, 1))\nrss = lambda a, b: sm.OLS(y[a:b], x[a:b]).fit().ssr\nF = [((rss(0, n) - rss(0, s) - rss(s, n)) / 1) / ((rss(0, s) + rss(s, n)) / (n - 2)) for s in range(int(trim * n), int((1 - trim) * n))]\nprint(max(F))\n",
    )


def bai_perron(x: np.ndarray, y: np.ndarray, max_breaks: int, min_size: int) -> dict[int, tuple[float, list[int]]]:
    """Dynamic programming search for up to ``max_breaks`` breaks: {m: (SSR, break positions)} (break = first obs. of a regime)."""
    n = len(y)
    moments = _cumulative_moments(x, y)
    ssr = np.full((n + 1, n + 1), np.inf)
    for i in range(n - min_size + 1):
        for j in range(i + min_size, n + 1):
            ssr[i, j] = _segment(moments, i, j)[0]
    best = {0: np.full(n + 1, np.inf)}
    best[0] = ssr[0].copy()
    choice: dict[int, np.ndarray] = {}
    previous = best[0]
    for m in range(1, max_breaks + 1):
        current = np.full(n + 1, np.inf)
        argmin = np.zeros(n + 1, dtype=int)
        for end in range((m + 1) * min_size, n + 1):
            candidates = (
                previous[np.arange(m * min_size, end - min_size + 1)]
                + ssr[np.arange(m * min_size, end - min_size + 1), end]
            )
            pick = int(np.argmin(candidates))
            current[end], argmin[end] = candidates[pick], m * min_size + pick
        best[m], choice[m] = current, argmin
        previous = current
    result: dict[int, tuple[float, list[int]]] = {0: (float(ssr[0, n]), [])}
    for m in range(1, max_breaks + 1):
        if not np.isfinite(best[m][n]):
            continue
        breaks, end = [], n
        for level in range(m, 0, -1):
            start = int(choice[level][end])
            breaks.append(start)
            end = start
        result[m] = (float(best[m][n]), sorted(breaks))
    return result


def _bai_perron(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts, x, names = _design_for_breaks(frame, roles, 30)
    if ts.n > 600:
        raise InvalidInputError(
            "La búsqueda de Bai-Perron admite hasta 600 observaciones (agregue o recorte la serie)."
        )
    k, trim, max_breaks = x.shape[1], float(p["trim"]), int(p["max_breaks"])
    min_size = max(int(np.ceil(trim * ts.n)), k + 2)
    if (max_breaks + 1) * min_size > ts.n:
        raise InvalidInputError(
            "Con ese recorte no caben tantos regímenes: reduzca el número de quiebres o el recorte."
        )
    found = bai_perron(x, ts.y, max_breaks, min_size)
    n = ts.n
    criteria = {m: n * np.log(ssr / n) + ((m + 1) * k + m) * np.log(n) for m, (ssr, _) in found.items()}
    lwz = {
        m: np.log(ssr / (n - (m + 1) * k)) + 0.299 * ((m + 1) * k + m) * (np.log(n)) ** 2.1 / n
        for m, (ssr, _) in found.items()
    }
    chosen = min(criteria, key=criteria.get) if p["criterion"] == "bic" else min(lwz, key=lwz.get)
    out = Output(
        f"Bai-Perron: quiebres múltiples en {ts.name}",
        f"hasta {max_breaks} quiebre(s) · recorte {100 * trim:g} % (mínimo {min_size} observaciones por régimen) · n = {n}",
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Quiebres elegidos", chosen),
            ("Regímenes", chosen + 1),
            ("SCR final", found[chosen][0]),
            ("Criterio", p["criterion"].upper()),
        ]
    )
    out.table(
        "Modelos con 0 a m quiebres",
        ["Quiebres", "SCR", "BIC", "LWZ", "Elegido"],
        [[m, found[m][0], float(criteria[m]), float(lwz[m]), good("●") if m == chosen else ""] for m in sorted(found)],
        note="La mejor partición de cada tamaño se halla por programación dinámica (óptima); el número de quiebres se elige minimizando el criterio.",
    )
    boundaries = [0, *found[chosen][1], n]
    moments = _cumulative_moments(x, ts.y)
    rows, segments = [], []
    for r in range(len(boundaries) - 1):
        start, end = boundaries[r], boundaries[r + 1]
        ssr, beta = _segment(moments, start, end)
        sigma2 = ssr / max(end - start - k, 1)
        covariance = sigma2 * np.linalg.pinv(moments[0][end] - moments[0][start])
        se = np.sqrt(np.diag(covariance))
        rows.append(
            [
                r + 1,
                ts.labels[start],
                ts.labels[end - 1],
                end - start,
                *[f"{beta[i]:.4g} ({se[i]:.3g})" for i in range(k)],
            ]
        )
        segments.append((start, end, beta))
    out.table("Regímenes (coeficiente y error estándar)", ["Régimen", "Desde", "Hasta", "Observaciones", *names], rows)
    if chosen:
        out.alert(
            "warning",
            "Fechas de los quiebres (primera observación del nuevo régimen): "
            + ", ".join(str(ts.labels[b]) for b in found[chosen][1])
            + ".",
        )
    else:
        out.alert("success", "El criterio no encuentra ningún quiebre: un solo régimen describe bien la serie.")
    labels = np.asarray(ts.labels, dtype=object)
    fitted = np.concatenate([x[s:e] @ b for s, e, b in segments])
    out.chart(
        line_chart(
            "Serie y ajuste por regímenes",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            [
                {"name": "Datos", "x": labels, "y": ts.y},
                {"name": "Ajuste por régimen", "x": labels, "y": fitted, "dash": "dash"},
            ],
            vlines=[{"value": labels[b], "dash": "dot"} for b in found[chosen][1]],
        )
    )
    return out.set(
        breaks=[int(b) + 1 for b in found[chosen][1]],
        break_labels=[str(ts.labels[b]) for b in found[chosen][1]],
        chosen=chosen,
        ssr={str(m): v[0] for m, v in found.items()},
    ).build()


def _bp_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "# pip install ruptures; búsqueda óptima de cambios en la media por programación dinámica\nimport ruptures as rpt\n\n"
        + _series_code(
            roles,
            dataset,
            f"result = rpt.Dynp(model='l2', min_size={max(int(float(p['trim']) * 100), 5)}).fit(y).predict(n_bkps={int(p['max_breaks'])})\nprint(result)\n",
        )
    )


# ============================================================================ Markov regime switching
def _markov(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, minimum=60)
    regimes, order = int(p["regimes"]), int(p["order"])
    np.random.seed(int(p["seed"]))  # the starting-value search draws from the global generator
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if order:
                model = MarkovAutoregression(
                    ts.y,
                    k_regimes=regimes,
                    order=order,
                    switching_ar=bool(p["switch_ar"]),
                    switching_variance=bool(p["switch_variance"]),
                )
            else:
                model = MarkovRegression(
                    ts.y, k_regimes=regimes, trend="c", switching_variance=bool(p["switch_variance"])
                )
            result = model.fit(search_reps=int(p["restarts"]), disp=False, maxiter=300)
    except FIT_ERRORS + (RuntimeError,) as error:
        raise ComputationError(f"No se pudo ajustar el modelo de cambio de régimen: {error}") from None
    smoothed = np.asarray(result.smoothed_marginal_probabilities, float)
    smoothed = smoothed.reshape(ts.n - order, regimes) if smoothed.ndim == 2 else smoothed
    labels = np.asarray(ts.labels, dtype=object)[order:]
    assignment = smoothed.argmax(axis=1)
    params = dict(zip(result.model.param_names, np.asarray(result.params, float), strict=True))
    means = (
        [float(params[f"const[{r}]"]) for r in range(regimes)]
        if f"const[{regimes - 1}]" in result.model.param_names
        else [float(params["const[0]"])] * regimes
    )
    out = Output(
        f"Cambio de régimen de Markov ({regimes} regímenes): {ts.name}",
        ("autorregresivo de orden " + str(order) if order else "regresión con media cambiante") + f" · n = {ts.n}",
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Regímenes", regimes),
            ("Log-verosimilitud", float(result.llf)),
            ("AIC", float(result.aic)),
            ("BIC", float(result.bic)),
        ]
    )
    out.table(
        "Parámetros",
        ["Parámetro", "Estimación", "Error est.", "z", "Valor p"],
        [
            [name, float(result.params[i]), float(result.bse[i]), float(result.tvalues[i]), P(float(result.pvalues[i]))]
            for i, name in enumerate(result.model.param_names)
        ],
    )
    transition = np.asarray(result.regime_transition, float)
    transition = transition[..., 0] if transition.ndim == 3 else transition
    out.table(
        "Matriz de transición (fila: régimen actual, columna: siguiente)",
        ["De \\ A", *[f"Régimen {j + 1}" for j in range(regimes)]],
        [[f"Régimen {i + 1}", *(float(transition[j, i]) for j in range(regimes))] for i in range(regimes)],
        note="Probabilidad de pasar de un régimen a otro de un periodo al siguiente.",
    )
    durations = np.asarray(result.expected_durations, float)
    out.table(
        "Regímenes",
        ["Régimen", "Media (constante)", "Duración esperada (periodos)", "Observaciones asignadas", "% del tiempo"],
        [
            [
                f"Régimen {r + 1}",
                means[r],
                float(durations[r]),
                int((assignment == r).sum()),
                100 * float((assignment == r).mean()),
            ]
            for r in range(regimes)
        ],
    )
    out.chart(
        line_chart(
            "Probabilidades suavizadas de cada régimen",
            "Tiempo" if ts.dated else "Observación",
            "Probabilidad",
            [{"name": f"Régimen {r + 1}", "x": labels, "y": smoothed[:, r]} for r in range(regimes)],
        )
    )
    out.chart(
        line_chart(
            "Serie por régimen más probable",
            "Tiempo" if ts.dated else "Observación",
            ts.name,
            [
                {"name": "Serie", "x": np.asarray(ts.labels, dtype=object), "y": ts.y},
                {"name": "Media del régimen", "x": labels, "y": np.array(means)[assignment], "dash": "dash"},
            ],
        )
    )
    stamps = pd.Series(labels)
    classification = pd.DataFrame({"tiempo": stamps.to_numpy(), ts.name: ts.y[order:], "regimen": assignment + 1})
    for r in range(regimes):
        classification[f"prob_regimen_{r + 1}"] = smoothed[:, r]
    out.dataset("regimenes", "Régimen de cada periodo", classification)
    return out.set(
        llf=float(result.llf),
        aic=float(result.aic),
        durations=durations.tolist(),
        means=means,
        transition=transition.tolist(),
    ).build()


def _markov_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    model = (
        f"MarkovAutoregression(y, k_regimes={int(p['regimes'])}, order={int(p['order'])})"
        if p["order"]
        else f"MarkovRegression(y, k_regimes={int(p['regimes'])}, trend='c', switching_variance={bool(p['switch_variance'])})"
    )
    return (
        "from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression\nfrom statsmodels.tsa.regime_switching.markov_regression import MarkovRegression\n\n"
        + _series_code(
            roles, dataset, f"res = {model}.fit(search_reps=20)\nprint(res.summary())\nprint(res.expected_durations)\n"
        )
    )


# ============================================================================ unobserved components (state space)
_STRUCTURES = (
    ("llevel", "Nivel local (paseo aleatorio + ruido)"),
    ("rwdrift", "Paseo aleatorio con deriva"),
    ("lltrend", "Tendencia lineal local"),
    ("strend", "Tendencia suave (pendiente estocástica, nivel fijo)"),
)


def _ucm(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ts = load_series(frame, roles, period=p["period"], minimum=30)
    h, alpha, structure = int(p["horizon"]), p["alpha"], p["structure"]
    seasonal = ts.period if p["seasonal"] and ts.period > 1 and ts.n >= 2 * ts.period else None
    if p["seasonal"] and not seasonal:
        raise InvalidInputError("Para el componente estacional indique el periodo y use al menos dos ciclos.")
    model = UnobservedComponents(
        ts.y,
        level=structure,
        seasonal=seasonal,
        cycle=bool(p["cycle"]),
        stochastic_cycle=bool(p["cycle"]),
        autoregressive=int(p["ar"]) or None,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False, maxiter=300)
    except FIT_ERRORS + (RuntimeError,) as error:
        raise ComputationError(f"No se pudo ajustar el modelo de componentes no observados: {error}") from None
    out = Output(
        f"Componentes no observados: {ts.name}",
        f"{dict(_STRUCTURES)[structure].split(' (')[0].lower()}"
        + (f" + estacional ({seasonal})" if seasonal else "")
        + (" + ciclo" if p["cycle"] else "")
        + f" · n = {ts.n}",
    )
    note_trimmed(out, ts)
    out.tiles(
        [
            ("Observaciones", ts.n),
            ("Log-verosimilitud", float(result.llf)),
            ("AIC", float(result.aic)),
            ("BIC", float(result.bic)),
        ]
    )
    out.table(
        "Parámetros (varianzas de los componentes)",
        ["Parámetro", "Estimación", "Error est.", "Valor p"],
        [
            [
                name,
                float(result.params[i]),
                float(result.bse[i]) if np.isfinite(result.bse[i]) else None,
                P(float(result.pvalues[i])) if np.isfinite(result.pvalues[i]) else None,
            ]
            for i, name in enumerate(result.param_names)
        ],
        note="Una varianza cercana a 0 significa que ese componente es prácticamente determinista (constante en el tiempo).",
    )
    x = np.asarray(ts.labels, dtype=object)
    comp: dict[str, np.ndarray] = {"nivel": np.asarray(result.smoothed_state[0], float)}
    if structure in ("lltrend", "strend", "rwdrift"):
        comp["pendiente"] = np.asarray(result.smoothed_state[1], float)
    if seasonal:
        comp["estacional"] = np.asarray(result.seasonal.smoothed, float)
    if p["cycle"]:
        comp["ciclo"] = np.asarray(result.cycle.smoothed, float)
    panels = [
        {
            **line_chart(
                "Serie y nivel suavizado",
                None,
                None,
                [{"name": "Datos", "x": x, "y": ts.y}, {"name": "Nivel", "x": x, "y": comp["nivel"], "dash": "dash"}],
                height=240,
            ),
            "title": "Serie y nivel suavizado",
        }
    ]
    for name, values in comp.items():
        if name != "nivel":
            panels.append(
                {
                    **line_chart(name.capitalize(), None, None, [{"name": name, "x": x, "y": values}], height=220),
                    "title": name.capitalize(),
                }
            )
    out.chart(
        panels_chart(
            f"Componentes estimados con el filtro de Kalman de {ts.name}",
            panels,
            columns=1,
            height=250 * len(panels) + 40,
        )
    )
    if h:
        forecast = result.get_forecast(h).summary_frame(alpha=alpha)
        mean, lower, upper = (
            np.asarray(forecast["mean"], float),
            np.asarray(forecast["mean_ci_lower"], float),
            np.asarray(forecast["mean_ci_upper"], float),
        )
        from sigma.domain.analysis.ts_core import forecast_rows

        out.table(
            "Pronósticos",
            ["Periodo", "Pronóstico", f"IP {100 * (1 - alpha):g} % inf.", "IP sup."],
            forecast_rows(future_labels(ts, h), mean, lower, upper),
        )
        out.chart(history_and_forecast(ts, np.asarray(result.fittedvalues, float), mean, lower, upper, alpha))
    stamps = pd.Series(pd.DatetimeIndex(ts.index)) if ts.dated else pd.Series(np.arange(1, ts.n + 1))
    out.dataset("componentes", "Componentes estimados", pd.DataFrame({"tiempo": stamps, ts.name: ts.y, **comp}))
    return out.set(
        llf=float(result.llf),
        aic=float(result.aic),
        variances={n: float(result.params[i]) for i, n in enumerate(result.param_names)},
    ).build()


def _ucm_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    seasonal = f", seasonal={int(p['period'] or 12)}" if p["seasonal"] else ""
    return "from statsmodels.tsa.statespace.structural import UnobservedComponents\n\n" + _series_code(
        roles,
        dataset,
        f"res = UnobservedComponents(y, level={p['structure']!r}{seasonal}).fit()\nprint(res.summary())\n",
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


HORIZON = FieldSpec("horizon", "Periodos a pronosticar (0 = no)", "integer", 10, minimum=0, maximum=500)

SPECS = (
    _spec("garch", "GARCH, EGARCH, GJR y volatilidad", GROUP_VOL, "Volatilidad condicional con asimetría y colas pesadas: parámetros, persistencia, diagnóstico, valor en riesgo con prueba de Kupiec y pronóstico.", _garch, heavy=True, roles=(RoleSpec("series", "Rendimientos (o precios)", "numeric"), TIME), params=(FieldSpec("variant", "Modelo de volatilidad", "select", "gjr", _VARIANTS), FieldSpec("p", "p (rezagos de σ²)", "integer", 1, minimum=0, maximum=4, visible_if=("variant", ("garch", "gjr", "egarch", "tarch", "aparch"))), FieldSpec("o", "o (términos asimétricos)", "integer", 1, minimum=1, maximum=3, visible_if=("variant", ("gjr", "egarch", "tarch", "aparch"))), FieldSpec("q", "q (rezagos de ε²)", "integer", 1, minimum=1, maximum=4), FieldSpec("mean", "Modelo de la media", "select", "constant", (("constant", "Constante"), ("zero", "Cero"), ("ar", "Autorregresivo"))), FieldSpec("ar_lags", "Rezagos AR de la media", "integer", 1, minimum=1, maximum=5, visible_if=("mean", ("ar",))), FieldSpec("distribution", "Distribución de los errores", "select", "t", _DISTRIBUTIONS), FieldSpec("input", "Los datos son", "select", "returns", (("returns", "Rendimientos"), ("prices", "Precios (se calculan rendimientos logarítmicos)"))), FieldSpec("percent", "Multiplicar por 100 (rendimientos en %)", "boolean", True, help="Mejora la estabilidad numérica de la optimización."), FieldSpec("var_level", "Cola del valor en riesgo", "number", 0.05, minimum=0.001, maximum=0.2), FieldSpec("horizon", "Periodos a pronosticar (0 = no)", "integer", 10, minimum=0, maximum=250), FieldSpec("alpha", "Nivel de significancia", "number", 0.05, minimum=0.001, maximum=0.5)), outputs=(OutputSpec("volatilidad", "Volatilidad condicional"),), code=_garch_code, example={"roles": {"series": "ret", "time": "t"}, "params": {"variant": "garch", "horizon": 5}}),
    _spec("markov", "Cambio de régimen de Markov", GROUP_VOL, "Detecta regímenes (expansión/recesión, baja/alta volatilidad) con probabilidades de transición, duraciones y la clasificación de cada periodo.", _markov, heavy=True, roles=(SERIES, TIME), params=(FieldSpec("regimes", "Número de regímenes", "integer", 2, minimum=2, maximum=3), FieldSpec("order", "Orden autorregresivo (0 = solo media cambiante)", "integer", 0, minimum=0, maximum=4), FieldSpec("switch_variance", "La varianza también cambia de régimen", "boolean", True), FieldSpec("switch_ar", "Los coeficientes AR cambian de régimen", "boolean", False, visible_if=("order", ("1", "2", "3", "4"))), FieldSpec("restarts", "Reinicios de la optimización", "integer", 10, minimum=0, maximum=100), FieldSpec("seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1)), outputs=(OutputSpec("regimenes", "Régimen de cada periodo"),), code=_markov_code, example={"roles": {"series": "rg", "time": "t"}, "params": {"restarts": 5}}),
    _spec("ardl", "ARDL y prueba de límites (bounds test)", GROUP_DYN, "Relación dinámica entre una variable y sus determinantes aunque mezclen órdenes de integración I(0) e I(1): multiplicadores de largo plazo y prueba de cointegración.", _ardl, roles=(RoleSpec("y", "Variable dependiente", "numeric"), RoleSpec("x", "Variables explicativas (1 a 4)", "numeric", multiple=True, max_count=4), TIME), params=(FieldSpec("max_lag_y", "Rezagos máximos de Y", "integer", 4, minimum=1, maximum=12), FieldSpec("max_lag_x", "Rezagos máximos de cada X", "integer", 4, minimum=0, maximum=12), FieldSpec("ic", "Criterio de selección", "select", "aic", (("aic", "AIC"), ("bic", "BIC"), ("hqic", "HQIC"))), FieldSpec("case", "Términos deterministas (casos de Pesaran)", "select", "3", _CASES), ALPHA), code=_ardl_code, example={"roles": {"y": "yp", "x": ["x1"], "time": "t"}, "params": {"max_lag_y": 2, "max_lag_x": 2}}),
    _spec("ucm", "Componentes no observados (Kalman)", GROUP_DYN, "Descompone la serie en nivel, pendiente, estacionalidad y ciclo con modelos de espacio de estados; pronostica con intervalos.", _ucm, heavy=True, roles=(SERIES, TIME), params=(FieldSpec("structure", "Tendencia", "select", "lltrend", _STRUCTURES), FieldSpec("seasonal", "Componente estacional", "boolean", False), FieldSpec("cycle", "Componente cíclico estocástico", "boolean", False), FieldSpec("ar", "Orden autorregresivo del residuo (0 = ninguno)", "integer", 0, minimum=0, maximum=3), HORIZON, PERIOD, ALPHA), outputs=(OutputSpec("componentes", "Componentes estimados"),), code=_ucm_code, example={"roles": {"series": "s", "time": "t"}, "params": {"structure": "lltrend", "seasonal": True, "horizon": 6}}),
    _spec("panel", "Datos de panel: efectos fijos y aleatorios", GROUP_PANEL, "MCO agrupado, efectos fijos (individuos y tiempo) y efectos aleatorios con errores agrupados o de Driscoll-Kraay; pruebas F, Breusch-Pagan y Hausman.", _panel, roles=(RoleSpec("response", "Respuesta", "numeric"), RoleSpec("predictors", "Regresores", "any", multiple=True), RoleSpec("entity", "Individuo (entidad)", "any"), RoleSpec("time", "Periodo", "any")), params=(FieldSpec("cov", "Errores estándar", "select", "clustered", _PANEL_COV), FieldSpec("time_effects", "Agregar efectos de tiempo al modelo de efectos fijos", "boolean", False), ALPHA), outputs=(OutputSpec("efectos", "Efectos fijos por individuo"),), code=_panel_code, example={"roles": {"response": "yp", "predictors": ["x1"], "entity": "ent", "time": "per"}, "params": {}}),
    _spec("qa_break", "Cambio estructural de fecha desconocida (Quandt-Andrews)", GROUP_BREAKS, "Busca el punto de cambio más probable en la media o en una regresión con la prueba sup-F y valor p simulado.", _qa_break, roles=(SERIES, TIME, RoleSpec("regressors", "Regresores (opcional; vacío = cambio en la media)", "numeric", multiple=True, optional=True)), params=(FieldSpec("trim", "Recorte de los extremos", "number", 0.15, minimum=0.05, maximum=0.4), ALPHA), code=_qa_code, example={"roles": {"series": "rg", "time": "t"}, "params": {}}),
    _spec("bai_perron", "Quiebres múltiples (Bai-Perron)", GROUP_BREAKS, "Cuántos cambios estructurales hay y dónde están, por programación dinámica y criterio BIC/LWZ; coeficientes de cada régimen.", _bai_perron, heavy=True, roles=(SERIES, TIME, RoleSpec("regressors", "Regresores (opcional; vacío = cambios en la media)", "numeric", multiple=True, optional=True)), params=(FieldSpec("max_breaks", "Máximo de quiebres", "integer", 3, minimum=1, maximum=5), FieldSpec("trim", "Tamaño mínimo de un régimen (fracción de n)", "number", 0.1, minimum=0.05, maximum=0.4), FieldSpec("criterion", "Criterio", "select", "bic", (("bic", "BIC"), ("lwz", "LWZ (más conservador)")))), code=_bp_code, example={"roles": {"series": "rg", "time": "t"}, "params": {"max_breaks": 2}}),
)  # fmt: skip

registry.register(*SPECS)
