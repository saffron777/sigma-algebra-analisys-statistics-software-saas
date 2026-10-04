"""Systems of series (EViews VAR/VEC views, Stata var/vec): VAR, Granger causality, cointegration and error-correction models."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import heatmap_chart, line_chart, panels_chart
from sigma.domain.analysis.output import Output, P, bad, good
from sigma.domain.analysis.series_explore import ALPHA, TIME
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line
from sigma.domain.analysis.ts_core import MultiTS, future_labels_for, load_multi
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "econometria"
GROUP_SYSTEMS = "Sistemas de series"
GROUP_COINT = "Cointegración"
VARS = RoleSpec("vars", "Series (2 a 8)", "numeric", multiple=True, min_count=2, max_count=8)
FIT_ERRORS = (ValueError, np.linalg.LinAlgError, FloatingPointError, KeyError)
MAX_IRF_PANELS = 5


def _code_head(roles: dict[str, Any], dataset: str, extra: str = "") -> str:
    sort = f"df = df.sort_values({roles['time']!r})\n" if roles.get("time") else ""
    return load_line(dataset) + sort + f"data = df[{roles['vars']!r}].dropna()\n" + extra


def _require_rows(mts: MultiTS, needed: int, what: str) -> None:
    if mts.n < needed:
        raise InvalidInputError(f"{what} necesita al menos {needed} observaciones comunes (hay {mts.n}).")


# ============================================================================ VAR
_IC = (("aic", "AIC"), ("bic", "BIC"), ("hqic", "HQIC"), ("fpe", "FPE"))


def _var_term(term: str) -> str:
    """``L2.x1`` → ``x1 (rezago 2)``; ``const`` → ``Constante``."""
    if term == "const":
        return "Constante"
    if term.startswith("L") and "." in term:
        lag, name = term[1:].split(".", 1)
        if lag.isdigit():
            return f"{name} (rezago {lag})"
    return term


def _var(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    mts = load_multi(frame, roles, minimum=30)
    names, k, alpha, horizon = mts.names, mts.k, p["alpha"], int(p["horizon"])
    data = mts.frame()
    trend = p["trend"]
    max_lags = int(p["max_lags"])
    _require_rows(mts, max(30, (max_lags + 2) * k + 10), "El VAR con esos rezagos")
    model = VAR(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = model.select_order(max_lags, trend=trend)
    if p["lags"]:
        lags = int(p["lags"])
        chosen_by = "fijados por usted"
    else:
        lags = int(selection.selected_orders[p["criterion"]]) or 1
        chosen_by = f"elegidos por {dict(_IC)[p['criterion']]}"
    try:
        result = model.fit(lags, trend=trend)
    except FIT_ERRORS as error:
        raise ComputationError(f"No se pudo ajustar el VAR: {error}") from None
    out = Output(
        "Vector autorregresivo (VAR)", f"{k} series · {lags} rezago(s) {chosen_by} · n efectivo = {result.nobs}"
    )
    out.tiles(
        [
            ("Rezagos", lags),
            ("Observaciones efectivas", int(result.nobs)),
            ("AIC", float(result.aic)),
            ("¿Estable?", "sí" if result.is_stable() else "no"),
        ]
    )
    ics = selection.ics
    out.table(
        "Selección de rezagos",
        ["Rezagos", *[label for _, label in _IC]],
        [[lag, *(float(ics[key][lag]) for key, _ in _IC)] for lag in range(len(ics["aic"]))],
        note="El rezago que minimiza cada criterio es el sugerido: "
        + ", ".join(f"{label} {selection.selected_orders[key]}" for key, label in _IC)
        + ".",
    )
    moduli = np.sort(np.abs(result.roots))
    stable = bool(result.is_stable())
    out.alert(
        "success" if stable else "warning",
        f"Todas las raíces del polinomio característico están fuera del círculo unitario (mínimo {moduli[0]:.3f}): el VAR es estable."
        if stable
        else f"El VAR no es estable (raíz con módulo {moduli[0]:.3f} ≤ 1): las series probablemente no son estacionarias; use diferencias o un VECM.",
    )
    rows = []
    for eq in names:
        for term in result.params.index:
            coef, se, t, pv = (
                float(result.params.loc[term, eq]),
                float(result.stderr.loc[term, eq]),
                float(result.tvalues.loc[term, eq]),
                float(result.pvalues.loc[term, eq]),
            )
            rows.append([eq, _var_term(str(term)), coef, se, t, P(pv)])
    out.table(
        "Coeficientes por ecuación",
        ["Ecuación (dependiente)", "Regresor", "Coeficiente", "Error est.", "t", "Valor p"],
        rows,
    )
    resid = np.asarray(result.resid)
    r2 = [
        1
        - (resid[:, i] ** 2).sum()
        / ((data[names[i]].to_numpy()[lags:] - data[names[i]].to_numpy()[lags:].mean()) ** 2).sum()
        for i in range(k)
    ]
    out.table(
        "Ajuste de cada ecuación",
        ["Ecuación", "R²", "Error estándar residual"],
        [[names[i], float(r2[i]), float(resid[:, i].std(ddof=k * lags + 1))] for i in range(k)],
    )
    matrix = np.full((k, k), np.nan)
    for i, j in product(range(k), range(k)):
        if i != j:
            matrix[i, j] = float(result.test_causality(names[i], [names[j]], kind="f").pvalue)
    out.table(
        "Causalidad de Granger (valor p de que la columna ayuda a predecir la fila)",
        ["Variable que se predice", *names],
        [
            [
                names[i],
                *(
                    None if i == j else (bad(P(matrix[i, j])) if matrix[i, j] < alpha else P(matrix[i, j]))
                    for j in range(k)
                ),
            ]
            for i in range(k)
        ],
        note=f"En rojo: p < {alpha:g}, la columna «causa en el sentido de Granger» a la fila (le aporta información predictiva).",
    )
    out.chart(
        heatmap_chart(
            "Valores p de la causalidad de Granger",
            np.where(np.isnan(matrix), 1.0, matrix),
            names,
            names,
            x_title="Causa (columna)",
            y_title="Efecto (fila)",
            annotate=k <= 6,
            height=max(320, 70 * k + 120),
        )
    )
    whiteness = result.test_whiteness(nlags=min(10, max(lags + 2, 4)))
    normality = result.test_normality()
    out.table(
        "Pruebas sobre los residuos",
        ["Prueba", "Estadístico", "gl", "Valor p"],
        [
            [
                "Portmanteau (autocorrelación)",
                float(whiteness.test_statistic),
                int(whiteness.df),
                P(float(whiteness.pvalue)),
            ],
            [
                "Normalidad conjunta (Jarque-Bera)",
                float(normality.test_statistic),
                int(normality.df),
                P(float(normality.pvalue)),
            ],
        ],
        note="Valores p pequeños: queda autocorrelación (agregue rezagos) o los residuos no son normales (afecta los intervalos).",
    )
    irf = result.irf(horizon)
    orth, se = np.asarray(irf.orth_irfs), np.asarray(irf.stderr(orth=True))
    z = stats.norm.ppf(1 - alpha / 2)
    steps = np.arange(horizon + 1)
    if k <= MAX_IRF_PANELS:
        panels = []
        for response, impulse in product(range(k), range(k)):
            title = f"{names[impulse]} → {names[response]}"
            panels.append(
                {
                    **line_chart(
                        title,
                        None,
                        None,
                        [{"name": "Respuesta", "x": steps, "y": orth[:, response, impulse]}],
                        bands=[
                            {
                                "name": "IC",
                                "x": steps,
                                "lower": orth[:, response, impulse] - z * se[:, response, impulse],
                                "upper": orth[:, response, impulse] + z * se[:, response, impulse],
                            }
                        ],
                        hlines=[{"value": 0.0}],
                        height=230,
                    ),
                    "title": title,
                }
            )
        out.chart(
            panels_chart(
                f"Funciones de impulso-respuesta ortogonalizadas (Cholesky, orden: {', '.join(names)})",
                panels,
                columns=k,
                height=260 * k + 60,
            )
        )
    out.table(
        f"Impulso-respuesta acumulada a {horizon} periodos",
        ["Impulso (choque de 1 desv. est.)", *[f"Respuesta de {n}" for n in names]],
        [[names[i], *(float(orth[:, r, i].sum()) for r in range(k))] for i in range(k)],
        note="Choques ortogonalizados con la descomposición de Cholesky: el orden de las series importa (la primera afecta a todas en el mismo periodo).",
    )
    fevd = np.asarray(result.fevd(horizon).decomp)
    marks = sorted({1, min(4, horizon), min(8, horizon), horizon})
    rows = []
    for i in range(k):
        for h in marks:
            rows.append([names[i], h, *(100 * float(fevd[i, h - 1, j]) for j in range(k))])
    out.table(
        "Descomposición de la varianza del error de pronóstico (%)",
        ["Variable", "Horizonte", *[f"Choque de {n}" for n in names]],
        rows,
    )
    steps_f = int(p["forecast_steps"])
    if steps_f:
        point, lower, upper = result.forecast_interval(data.to_numpy()[-lags:], steps_f, alpha=alpha)
        future = future_labels_for(mts.labels, mts.dated, mts.freq, steps_f)
        table_rows = [[future[s], *(float(point[s, i]) for i in range(k))] for s in range(steps_f)]
        out.table("Pronósticos", ["Periodo", *names], table_rows)
        keep = min(mts.n, 4 * steps_f + 12)
        panels = []
        for i, name in enumerate(names[:6]):
            panels.append(
                {
                    **line_chart(
                        name,
                        None,
                        None,
                        [
                            {"name": "Datos", "x": mts.labels[-keep:], "y": data[name].to_numpy()[-keep:]},
                            {
                                "name": "Pronóstico",
                                "x": [mts.labels[-1], *future],
                                "y": np.r_[data[name].to_numpy()[-1], point[:, i]],
                                "dash": "dash",
                            },
                        ],
                        bands=[{"name": "IP", "x": future, "lower": lower[:, i], "upper": upper[:, i]}],
                        height=240,
                    ),
                    "title": name,
                }
            )
        out.chart(panels_chart("Pronósticos con intervalos de predicción", panels, columns=min(2, len(panels))))
    stamps = pd.Series(mts.labels)
    out.dataset(
        "residuos",
        "Residuos del VAR",
        pd.DataFrame(resid, columns=[f"residuo_{n}" for n in names]).assign(tiempo=stamps.iloc[lags:].to_numpy()),
    )
    return out.set(
        lags=lags,
        stable=stable,
        min_root=float(moduli[0]),
        granger_p={f"{names[j]}->{names[i]}": float(matrix[i, j]) for i, j in product(range(k), range(k)) if i != j},
        aic=float(result.aic),
    ).build()


def _var_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    fit = f".fit({int(p['lags'])})" if p["lags"] else f".fit(maxlags={int(p['max_lags'])}, ic={p['criterion']!r})"
    return "from statsmodels.tsa.api import VAR\n\n" + _code_head(
        roles,
        dataset,
        f"res = VAR(data){fit}\nprint(res.summary())\nprint(res.irf({int(p['horizon'])}).irfs.shape)\nprint(res.fevd({int(p['horizon'])}).summary())\n",
    )


# ============================================================================ pairwise Granger causality
def _granger(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    roles_multi = {"vars": [roles["x"], roles["y"]], "time": roles.get("time")}
    mts = load_multi(frame, roles_multi, minimum=25)
    max_lag, alpha = int(p["max_lag"]), p["alpha"]
    data = mts.frame()
    if p["difference"]:
        data = data.diff().dropna()
    _require_rows(
        MultiTS(mts.names, data.to_numpy(), [], False, None, 1, 0),
        3 * max_lag + 10,
        "La prueba con ese número de rezagos",
    )
    x, y = roles["x"], roles["y"]
    out = Output(
        f"Causalidad de Granger: {x} y {y}",
        f"hasta {max_lag} rezagos · n = {len(data)}" + (" · en primeras diferencias" if p["difference"] else ""),
    )
    rows = []
    best: dict[str, tuple[int, float]] = {}
    for cause, effect in ((x, y), (y, x)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tests = grangercausalitytests(data[[effect, cause]], maxlag=max_lag)
        for lag in range(1, max_lag + 1):
            f, pv, df_den, df_num = tests[lag][0]["ssr_ftest"]
            chi2, pc, _ = tests[lag][0]["ssr_chi2test"]
            rows.append(
                [
                    f"{cause} → {effect}",
                    lag,
                    float(f),
                    f"{int(df_num)}, {int(df_den)}",
                    P(float(pv)),
                    float(chi2),
                    P(float(pc)),
                ]
            )
            if best.get(f"{cause}->{effect}", (0, 2.0))[1] > pv:
                best[f"{cause}->{effect}"] = (lag, float(pv))
            best.setdefault(f"{cause}->{effect}", (lag, float(pv)))
    out.table(
        "Pruebas por rezago",
        ["Hipótesis nula: la primera NO causa a la segunda", "Rezagos", "F", "gl", "Valor p (F)", "χ²", "Valor p (χ²)"],
        rows,
        note="H₀: los rezagos de la causa no mejoran la predicción del efecto. La causalidad de Granger es predictiva, no necesariamente causal.",
    )
    for key, (lag, pv) in best.items():
        cause, effect = key.split("->")
        out.alert(
            "warning" if pv < alpha else "info",
            f"{cause} → {effect}: "
            + (
                f"hay causalidad de Granger (mínimo p = {pv:.4f} con {lag} rezago(s))."
                if pv < alpha
                else f"no hay evidencia de que {cause} anticipe a {effect} (mínimo p = {pv:.4f})."
            ),
        )
    out.tiles(
        [("Observaciones", len(data)), ("Rezagos probados", max_lag)]
        + [(k_, f"p = {v[1]:.4f}") for k_, v in best.items()]
    )
    return out.set(best={k_: {"lag": v[0], "p": v[1]} for k_, v in best.items()}).build()


def _granger_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.tsa.stattools import grangercausalitytests\n\n"
        + load_line(dataset)
        + f"d = df[[{roles['y']!r}, {roles['x']!r}]].dropna()\n"
        + ("d = d.diff().dropna()\n" if p["difference"] else "")
        + f"print(grangercausalitytests(d, maxlag={int(p['max_lag'])}))  # ¿{roles['x']} causa a {roles['y']}?\n"
    )


# ============================================================================ cointegration
_DET = (
    ("-1", "Sin constante ni tendencia"),
    ("0", "Constante en la relación y en los datos"),
    ("1", "Constante y tendencia lineal"),
)


def _cointegration(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    mts = load_multi(frame, roles, minimum=40)
    names, k, alpha, lags = mts.names, mts.k, float(p["alpha"]), int(p["lags"])
    if k > 6:
        raise InvalidInputError("La prueba de Johansen admite hasta 6 series.")
    data = mts.frame()
    det = int(p["det_order"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        johansen = coint_johansen(data, det_order=det, k_ar_diff=lags)
    level_index = {0.10: 0, 0.05: 1, 0.01: 2}
    column = level_index[min(level_index, key=lambda v: abs(v - alpha))]
    trace_critical, max_critical = johansen.cvt[:, column], johansen.cvm[:, column]
    rank = 0
    while rank < k and johansen.lr1[rank] > trace_critical[rank]:
        rank += 1
    out = Output(
        "Cointegración (Engle-Granger y Johansen)",
        f"{k} series · n = {mts.n} · {lags} rezago(s) en diferencias · {dict(_DET)[str(det)].lower()}",
    )
    out.tiles(
        [
            ("Series", k),
            ("Rango de cointegración (traza)", rank),
            ("Observaciones", mts.n),
            ("Nivel", f"{100 * (1 - min(level_index, key=lambda v: abs(v - alpha))):g} %"),
        ]
    )
    if rank == 0:
        out.alert(
            "warning",
            "La prueba de la traza no detecta cointegración: las series no comparten una relación de equilibrio de largo plazo; modele sus diferencias con un VAR.",
        )
    elif rank >= k:
        out.alert("info", "El rango es completo: las series ya son estacionarias; use un VAR en niveles.")
    else:
        out.alert(
            "success",
            f"Hay {rank} relación(es) de cointegración: use un modelo de corrección de errores (VECM) con ese rango.",
        )
    out.table(
        "Prueba de la traza",
        ["Hipótesis nula (rango ≤ r)", "Estadístico", "Valor crítico", "¿Se rechaza?"],
        [
            [
                f"r ≤ {r}",
                float(johansen.lr1[r]),
                float(trace_critical[r]),
                good("sí") if johansen.lr1[r] > trace_critical[r] else bad("no"),
            ]
            for r in range(k)
        ],
        note="Se rechazan hipótesis de menor a mayor rango hasta que una no se rechace: ese es el rango estimado.",
    )
    out.table(
        "Prueba del máximo autovalor",
        ["Hipótesis nula", "Estadístico", "Valor crítico", "¿Se rechaza?"],
        [
            [
                f"r = {r}",
                float(johansen.lr2[r]),
                float(max_critical[r]),
                good("sí") if johansen.lr2[r] > max_critical[r] else bad("no"),
            ]
            for r in range(k)
        ],
    )
    beta = np.real(johansen.evec[:, : max(rank, 1)])
    normalised = beta / beta[0, :]
    out.table(
        "Vectores de cointegración (normalizados con la primera serie)",
        ["Serie", *[f"β{i + 1}" for i in range(max(rank, 1))]],
        [[names[i], *(float(normalised[i, j]) for j in range(normalised.shape[1]))] for i in range(k)],
        note="Cada columna define una combinación estacionaria: β′·y(t) fluctúa alrededor de un nivel estable."
        + ("" if rank else " (Se muestra el primero aunque no se rechace la no cointegración.)"),
    )
    out.table("Autovalores", ["#", "Autovalor"], [[i + 1, float(v)] for i, v in enumerate(johansen.eig)])
    y0, others = data[names[0]], data[names[1:]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, critical = coint(y0, others, trend="c", autolag="aic")
    out.table(
        f"Engle-Granger: {names[0]} frente a {', '.join(names[1:])}",
        ["Estadístico ADF de los residuos", "Valor p", "Crítico 1 %", "Crítico 5 %", "Crítico 10 %"],
        [[float(stat), P(float(pvalue)), *(float(c) for c in critical)]],
        note="H₀: no hay cointegración (los residuos de la regresión de nivel tienen raíz unitaria). El valor p usa las tablas de MacKinnon.",
    )
    spread = normalised[:, 0] @ data.to_numpy().T
    out.chart(
        line_chart(
            "Combinación de largo plazo β′y (debe ser estacionaria si hay cointegración)",
            "Tiempo" if mts.dated else "Observación",
            "β′y",
            [{"name": "β₁′y", "x": mts.labels, "y": spread}],
            hlines=[{"value": float(spread.mean()), "label": "media", "dash": "dash"}],
        )
    )
    return out.set(
        rank=rank,
        trace=[float(v) for v in johansen.lr1],
        trace_critical=[float(v) for v in trace_critical],
        eigenvalues=[float(v) for v in johansen.eig],
        engle_granger_p=float(pvalue),
        beta=normalised.tolist(),
    ).build()


def _cointegration_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.tsa.stattools import coint\nfrom statsmodels.tsa.vector_ar.vecm import coint_johansen\n\n"
        + _code_head(
            roles,
            dataset,
            f"print(coint_johansen(data, det_order={int(p['det_order'])}, k_ar_diff={int(p['lags'])}).lr1)\nprint(coint(data.iloc[:, 0], data.iloc[:, 1:]))\n",
        )
    )


# ============================================================================ VECM
_DETERMINISTIC = (
    ("co", "Constante fuera de la relación (en las diferencias)"),
    ("ci", "Constante dentro de la relación de largo plazo"),
    ("n", "Ninguno"),
    ("lo", "Tendencia fuera"),
    ("li", "Tendencia dentro"),
    ("cili", "Constante y tendencia dentro"),
    ("colo", "Constante y tendencia fuera"),
)


def _vecm(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    mts = load_multi(frame, roles, minimum=40)
    names, k, alpha, lags = mts.names, mts.k, p["alpha"], int(p["lags"])
    data = mts.frame()
    deterministic = p["deterministic"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = select_coint_rank(
            data,
            det_order=-1 if deterministic == "n" else 0 if deterministic in ("co", "ci") else 1,
            k_ar_diff=lags,
            signif=alpha,
        )
    rank = int(p["rank"]) if p["rank"] else int(selection.rank)
    if rank == 0:
        raise InvalidInputError(
            "No se detecta cointegración (rango 0): un VECM no tiene sentido. Use un VAR en diferencias o indique el rango a mano."
        )
    if rank >= k:
        raise InvalidInputError(f"El rango debe ser menor que el número de series ({k}).")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = VECM(data, k_ar_diff=lags, coint_rank=rank, deterministic=deterministic).fit()
    except FIT_ERRORS as error:
        raise ComputationError(f"No se pudo ajustar el VECM: {error}") from None
    out = Output(
        "Modelo de corrección de errores (VECM)",
        f"{k} series · rango {rank} · {lags} rezago(s) en diferencias · n = {mts.n}",
    )
    out.tiles(
        [
            ("Rango de cointegración", rank),
            ("Observaciones", mts.n),
            ("Log-verosimilitud", float(np.real(result.llf))),
            ("Rezagos en diferencias", lags),
        ]
    )
    out.table(
        "Elección del rango (Johansen, traza)",
        ["Rango r", "Estadístico", "Valor crítico", "¿Se rechaza r?"],
        [
            [r, float(stat), float(crit), good("sí") if stat > crit else bad("no")]
            for r, (stat, crit) in enumerate(zip(selection.test_stats, selection.crit_vals, strict=False))
        ],
        note=f"Rango sugerido por la prueba: {selection.rank}" + (f"; usted indicó {rank}." if p["rank"] else "."),
    )
    beta = np.real(np.asarray(result.beta))
    alpha_load = np.real(np.asarray(result.alpha))
    scale = beta[0, :]
    out.table(
        "Relación de largo plazo (β, normalizada con la primera serie)",
        ["Serie", *[f"β{i + 1}" for i in range(rank)]],
        [[names[i], *(float(beta[i, j] / scale[j]) for j in range(rank))] for i in range(k)],
        note="La relación de equilibrio es β′y = 0 (más un término determinista si lo incluyó).",
    )
    alpha_se = np.asarray(result.stderr_alpha)
    out.table(
        "Velocidad de ajuste (α)",
        ["Ecuación", *[f"α{i + 1}" for i in range(rank)], *[f"Valor p α{i + 1}" for i in range(rank)]],
        [
            [
                names[i],
                *(float(alpha_load[i, j] * scale[j]) for j in range(rank)),
                *(P(float(2 * stats.norm.sf(abs(alpha_load[i, j] / alpha_se[i, j])))) for j in range(rank)),
            ]
            for i in range(k)
        ],
        note="α negativo en la ecuación de una serie: esa serie corrige las desviaciones del equilibrio (vuelve hacia él).",
    )
    gamma = np.asarray(result.gamma)
    if gamma.size:
        rows = []
        for i in range(k):
            for j in range(gamma.shape[1]):
                lag, variable = divmod(j, k)
                rows.append(
                    [
                        names[i],
                        f"Δ{names[variable]} (rezago {lag + 1})",
                        float(gamma[i, j]),
                        float(result.stderr_gamma[i, j]),
                        P(float(2 * stats.norm.sf(abs(gamma[i, j] / result.stderr_gamma[i, j])))),
                    ]
                )
        out.table(
            "Dinámica de corto plazo (Γ)", ["Ecuación (Δ de)", "Regresor", "Coeficiente", "Error est.", "Valor p"], rows
        )
    matrix = np.full((k, k), np.nan)
    for i, j in product(range(k), range(k)):
        if i != j:
            matrix[i, j] = float(result.test_granger_causality(caused=names[i], causing=names[j]).pvalue)
    out.table(
        "Causalidad de Granger de corto plazo (valor p)",
        ["Variable que se predice", *names],
        [[names[i], *(None if i == j else P(matrix[i, j]) for j in range(k))] for i in range(k)],
    )
    whiteness = result.test_whiteness(nlags=min(10, max(lags + 3, 5)))
    out.table(
        "Residuos",
        ["Prueba", "Estadístico", "Valor p"],
        [["Portmanteau", float(whiteness.test_statistic), P(float(whiteness.pvalue))]],
    )
    spread = (beta[:, 0] / scale[0]) @ data.to_numpy().T
    out.chart(
        line_chart(
            "Término de corrección de errores (β₁′y)",
            "Tiempo" if mts.dated else "Observación",
            "β₁′y",
            [{"name": "Desvío del equilibrio", "x": mts.labels, "y": spread}],
            hlines=[{"value": float(spread.mean()), "dash": "dash"}],
        )
    )
    steps_f = int(p["forecast_steps"])
    if steps_f:
        point, lower, upper = result.predict(steps=steps_f, alpha=alpha)
        future = future_labels_for(mts.labels, mts.dated, mts.freq, steps_f)
        out.table(
            "Pronósticos",
            ["Periodo", *names],
            [[future[s], *(float(point[s, i]) for i in range(k))] for s in range(steps_f)],
        )
    return out.set(
        rank=rank, llf=float(np.real(result.llf)), alpha=(alpha_load * scale).tolist(), beta=(beta / scale).tolist()
    ).build()


def _vecm_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return "from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank\n\n" + _code_head(
        roles,
        dataset,
        f"print(select_coint_rank(data, det_order=0, k_ar_diff={int(p['lags'])}).summary())\nres = VECM(data, k_ar_diff={int(p['lags'])}, coint_rank={int(p['rank'] or 1)}, deterministic={p['deterministic']!r}).fit()\nprint(res.summary())\n",
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec("var", "VAR: vectores autorregresivos", GROUP_SYSTEMS, "Rezagos óptimos, estabilidad, causalidad de Granger, impulso-respuesta, descomposición de varianza y pronósticos de varias series a la vez.", _var, roles=(VARS, TIME), params=(FieldSpec("criterion", "Criterio de selección de rezagos", "select", "aic", _IC), FieldSpec("lags", "Rezagos fijos (vacío = elegir con el criterio)", "integer", optional=True, minimum=1, maximum=24), FieldSpec("max_lags", "Rezagos máximos a probar", "integer", 8, minimum=1, maximum=24), FieldSpec("trend", "Términos deterministas", "select", "c", (("c", "Constante"), ("ct", "Constante y tendencia"), ("n", "Ninguno"), ("ctt", "Constante, tendencia y tendencia²"))), FieldSpec("horizon", "Horizonte del impulso-respuesta", "integer", 10, minimum=2, maximum=60), FieldSpec("forecast_steps", "Periodos a pronosticar (0 = no)", "integer", 8, minimum=0, maximum=100), ALPHA), outputs=(OutputSpec("residuos", "Residuos del VAR"),), code=_var_code, example={"roles": {"vars": ["x1", "x2", "x3"], "time": "t"}, "params": {"max_lags": 4, "horizon": 6, "forecast_steps": 4}}),
    _spec("granger", "Causalidad de Granger entre dos series", GROUP_SYSTEMS, "¿Los valores pasados de X ayudan a predecir Y (y al revés)? Pruebas F y χ² para cada número de rezagos.", _granger, roles=(RoleSpec("x", "Serie X", "numeric"), RoleSpec("y", "Serie Y", "numeric"), TIME), params=(FieldSpec("max_lag", "Rezagos máximos", "integer", 4, minimum=1, maximum=24), FieldSpec("difference", "Usar primeras diferencias", "boolean", True, help="Las series deben ser estacionarias: use diferencias si tienen tendencia."), ALPHA), code=_granger_code, example={"roles": {"x": "x1", "y": "x2", "time": "t"}, "params": {}}),
    _spec("cointegration", "Cointegración (Engle-Granger y Johansen)", GROUP_COINT, "¿Comparten las series una relación de equilibrio de largo plazo? Traza y máximo autovalor, vectores β y prueba de Engle-Granger.", _cointegration, roles=(RoleSpec("vars", "Series (2 a 6, en niveles)", "numeric", multiple=True, min_count=2, max_count=6), TIME), params=(FieldSpec("det_order", "Términos deterministas", "select", "0", _DET), FieldSpec("lags", "Rezagos en diferencias (k − 1)", "integer", 2, minimum=0, maximum=12), FieldSpec("alpha", "Nivel de significancia", "select", "0.05", (("0.1", "10 %"), ("0.05", "5 %"), ("0.01", "1 %")))), code=_cointegration_code, example={"roles": {"vars": ["s", "y"], "time": "t"}, "params": {}}),
    _spec("vecm", "VECM: modelo de corrección de errores", GROUP_COINT, "Relación de largo plazo (β), velocidad de ajuste (α), dinámica de corto plazo y pronósticos para series cointegradas.", _vecm, roles=(RoleSpec("vars", "Series (2 a 6, en niveles)", "numeric", multiple=True, min_count=2, max_count=6), TIME), params=(FieldSpec("rank", "Rango de cointegración (vacío = según Johansen)", "integer", optional=True, minimum=1, maximum=5), FieldSpec("lags", "Rezagos en diferencias", "integer", 2, minimum=0, maximum=12), FieldSpec("deterministic", "Términos deterministas", "select", "ci", _DETERMINISTIC), FieldSpec("forecast_steps", "Periodos a pronosticar (0 = no)", "integer", 8, minimum=0, maximum=100), FieldSpec("alpha", "Nivel de significancia", "number", 0.05, minimum=0.001, maximum=0.5)), code=_vecm_code, example={"roles": {"vars": ["s", "y"], "time": "t"}, "params": {"rank": 1}}),
)  # fmt: skip

registry.register(*SPECS)
