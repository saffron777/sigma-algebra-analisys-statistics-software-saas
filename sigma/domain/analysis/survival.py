"""Supervivencia y confiabilidad: Kaplan-Meier con comparación de grupos, Cox y modelos paramétricos con censura."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma as _gamma
from statsmodels.duration.hazard_regression import PHReg

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import InvalidInputError

MODULE = "supervivencia"
GROUP_SURV = "Supervivencia"
GROUP_PARAM = "Modelos paramétricos"

TIME = RoleSpec("time", "Tiempo hasta el evento", "numeric", help="Tiempo de seguimiento o de fallo (positivo).")
EVENT = RoleSpec(
    "event", "Evento (1 = ocurrió, 0 = censurado)", "numeric", help="1 si el evento ocurrió; 0 si el seguimiento terminó antes."
)
GROUP = RoleSpec("group", "Grupo (opcional)", "categorical", optional=True)
PREDICTORS = RoleSpec("predictors", "Predictores (covariables)", "numeric", multiple=True, min_count=1, max_count=20)
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
_LOG_TESTS = (
    ("logrank", "Log-rank (pesa igual todo el seguimiento)"),
    ("wilcoxon", "Wilcoxon de Gehan-Breslow (pesa por el número en riesgo)"),
    ("tarone", "Tarone-Ware (pesa por la raíz del número en riesgo)"),
)


def _load_survival(frame: pd.DataFrame, roles: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.DataFrame(
        {
            "time": pd.to_numeric(frame[roles["time"]], errors="coerce"),
            "event": pd.to_numeric(frame[roles["event"]], errors="coerce"),
        }
    )
    if roles.get("group"):
        data["group"] = frame[roles["group"]].astype(str)
    data = data.dropna()
    if len(data) < 2:
        raise InvalidInputError("Se necesitan al menos 2 filas con tiempo y evento válidos.")
    time = data["time"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=float)
    if (time <= 0).any():
        raise InvalidInputError("El tiempo hasta el evento debe ser estrictamente positivo.")
    if not np.isin(event, [0.0, 1.0]).all():
        raise InvalidInputError("El evento debe ser 0 (censurado) o 1 (ocurrió).")
    if int(event.sum()) == 0:
        raise InvalidInputError("No hay eventos observados: todas las filas están censuradas.")
    return data, time, event.astype(int)


def _km(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Kaplan-Meier: (event times, survival, number at risk) for a single group."""
    order = np.argsort(time, kind="stable")
    t, e = time[order], event[order]
    event_times = np.unique(t[e == 1])
    surv, at_risk = [], []
    s = 1.0
    for tj in event_times:
        n_j = int(np.sum(t >= tj))
        d_j = int(np.sum((t == tj) & (e == 1)))
        s *= 1 - d_j / n_j
        surv.append(float(s))
        at_risk.append(n_j)
    return event_times, np.asarray(surv), at_risk


def _step(times: np.ndarray, surv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The step-function coordinates of a survival curve (the drop happens right after the event time)."""
    x, y = [0.0], [1.0]
    prev = 1.0
    for tj, sj in zip(times, surv, strict=True):
        x.extend([float(tj), float(tj)])
        y.extend([prev, float(sj)])
        prev = float(sj)
    return np.asarray(x), np.asarray(y)


def _median(event_times: np.ndarray, surv: np.ndarray) -> float | None:
    idx = int(np.searchsorted(-surv, -0.5))  # first survival <= 0.5
    return float(event_times[idx]) if idx < len(event_times) else None


def _logrank(time: np.ndarray, event: np.ndarray, group: np.ndarray, kind: str):
    """Weighted log-rank for K groups: (statistic, df, p-value, observed, expected) or ``None``."""
    groups = np.unique(group)
    k = len(groups)
    if k < 2:
        return None
    event_times = np.unique(time[event == 1])
    observed = np.zeros(k)
    expected = np.zeros(k)
    cov = np.zeros((k, k))
    for tj in event_times:
        at_risk = time >= tj
        n_j = int(at_risk.sum())
        d_j = int(((time == tj) & (event == 1)).sum())
        if n_j <= 1:
            continue
        w = {"logrank": 1.0, "wilcoxon": float(n_j), "tarone": float(np.sqrt(n_j))}[kind]
        n_g = np.array([int((at_risk & (group == g)).sum()) for g in groups], dtype=float)
        d_g = np.array([int(((time == tj) & (event == 1) & (group == g)).sum()) for g in groups], dtype=float)
        observed += w * d_g
        expected += w * d_j * n_g / n_j
        p = n_g / n_j
        cov += w * w * d_j * (n_j - d_j) / (n_j - 1) * (np.diag(p) - np.outer(p, p))
    diff = observed - expected
    stat = float(diff @ np.linalg.pinv(cov) @ diff)
    return stat, k - 1, float(stats.chi2.sf(stat, k - 1)), observed, expected


# ============================================================================ Kaplan-Meier
def _kaplan_meier(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    data, time, event = _load_survival(frame, roles)
    event_times, surv, at_risk = _km(time, event)
    out = Output("Curva de Kaplan-Meier", f"n = {len(time)} · {int(event.sum())} eventos")
    median = _median(event_times, surv)
    out.tiles(
        [
            ("Observaciones", len(time)),
            ("Eventos", int(event.sum())),
            ("Mediana de supervivencia", median if median is not None else "no alcanzada"),
        ]
    )
    x, y = _step(event_times, surv)
    series = [{"name": "Global", "x": x, "y": y}]
    groups = list(np.unique(data["group"])) if roles.get("group") else []
    for g in groups:
        m = data["group"].to_numpy() == g
        et, s, _ = _km(time[m], event[m])
        xx, yy = _step(et, s)
        series.append({"name": str(g), "x": xx, "y": yy})
    out.chart(line_chart("Supervivencia estimada", "Tiempo", "S(t)", series))
    out.table(
        "Curva de supervivencia (global)",
        ["Tiempo", "S(t)", "En riesgo"],
        [[float(t), float(s), int(n)] for t, s, n in zip(event_times, surv, at_risk, strict=True)][:12],
        note="S(t) es el producto de (1 − d/n) hasta ese tiempo; se muestran los primeros pasos.",
    )
    if len(groups) >= 2:
        result = _logrank(time, event, data["group"].to_numpy(), p["test"])
        if result:
            stat, df, pvalue, observed, expected = result
            out.heading(f"Comparación entre grupos ({dict(_LOG_TESTS)[p['test']].split('(')[0].strip()})")
            out.tiles([("Estadístico χ²", stat), ("Grados de libertad", df), ("Valor p", P(pvalue))])
            significant = pvalue < p["alpha"]
            out.alert(
                "warning" if significant else "info",
                f"Hay diferencias significativas entre los grupos (p = {pvalue:.4f})."
                if significant
                else f"No hay evidencia de diferencias entre los grupos (p = {pvalue:.4f}).",
            )
            out.table(
                "Eventos observados y esperados",
                ["Grupo", "Observados", "Esperados", "O − E"],
                [
                    [str(g), int(observed[i]), float(expected[i]), float(observed[i] - expected[i])]
                    for i, g in enumerate(groups)
                ],
                note="Bajo la hipótesis nula (curvas iguales) lo observado y lo esperado deberían parecerse.",
            )
    return out.set(n=len(time), events=int(event.sum()), median=median).build()


def _kaplan_meier_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nimport pandas as pd\n\n"
        + load_line(dataset)
        + f"time = df[{py_name(roles['time'])}].to_numpy(float)\nevent = df[{py_name(roles['event'])}].to_numpy(int)\n"
        + "order = np.argsort(time)\nt, e = time[order], event[order]\n"
        + "s, rows = 1.0, []\nfor tj in np.unique(t[e == 1]):\n"
        + "    s *= 1 - ((t == tj) & (e == 1)).sum() / (t >= tj).sum()\n    rows.append((tj, s))\n"
        + "print(pd.DataFrame(rows, columns=['tiempo', 'S(t)']))\n"
    )


# ============================================================================ Cox
def _cox(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = list(roles["predictors"])
    cols = [roles["time"], roles["event"], *names]
    data = frame[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    if len(data) < 10:
        raise InvalidInputError("Se necesitan al menos 10 filas completas para el modelo de Cox.")
    time = data[roles["time"]].to_numpy(dtype=float)
    event = data[roles["event"]].to_numpy(dtype=float)
    if (time <= 0).any():
        raise InvalidInputError("El tiempo hasta el evento debe ser estrictamente positivo.")
    if not np.isin(event, [0.0, 1.0]).all():
        raise InvalidInputError("El evento debe ser 0 (censurado) o 1 (ocurrió).")
    if int(event.sum()) == 0:
        raise InvalidInputError("No hay eventos observados: todas las filas están censuradas.")
    result = PHReg(time, data[names].to_numpy(dtype=float), status=event.astype(int)).fit()
    alpha = p["alpha"]
    z = stats.norm.ppf(1 - alpha / 2)
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    hr = np.exp(params)
    low, high = np.exp(params - z * bse), np.exp(params + z * bse)
    out = Output("Regresión de Cox (riesgos proporcionales)", f"n = {len(data)} · {int(event.sum())} eventos")
    out.tiles([("Observaciones", len(data)), ("Eventos", int(event.sum())), ("Log-verosimilitud", float(result.llf))])
    out.table(
        "Razones de riesgo",
        ["Predictor", "Coef. β", "Exp(β)", f"IC {100 * (1 - alpha):g} % inf.", "IC sup.", "Valor p"],
        [
            [name, float(params[i]), float(hr[i]), float(low[i]), float(high[i]), P(float(result.pvalues[i]))]
            for i, name in enumerate(names)
        ],
        note="Exp(β) > 1 aumenta el riesgo (acorta la supervivencia); < 1 lo reduce. Supone riesgos proporcionales.",
    )
    significant = [name for i, name in enumerate(names) if float(result.pvalues[i]) < alpha]
    out.alert(
        "info" if significant else "warning",
        "Predictores significativos: " + (", ".join(significant) if significant else "ninguno") + ".",
    )
    return out.set(
        n=len(data),
        events=int(event.sum()),
        llf=float(result.llf),
        hazard_ratios=[float(v) for v in hr],
        coefficients=[float(v) for v in params],
    ).build()


def _cox_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    names = ", ".join(f"df[{py_name(n)}]" for n in roles["predictors"])
    return (
        "import numpy as np\nfrom statsmodels.duration.hazard_regression import PHReg\n\n"
        + load_line(dataset)
        + f"time = df[{py_name(roles['time'])}].to_numpy(float)\nevent = df[{py_name(roles['event'])}].to_numpy(int)\n"
        + f"X = np.column_stack([{names}])\n"
        + "res = PHReg(time, X, status=event).fit()\nprint(res.params, res.bse, res.pvalues)\n"
    )


# ============================================================================ parametric
_DISTRIBUTIONS = (
    ("weibull", "Weibull"),
    ("exponential", "Exponencial"),
    ("lognormal", "Log-normal"),
    ("loglogistic", "Log-logística"),
)


def _parametric_loglik(theta: np.ndarray, time: np.ndarray, event: np.ndarray, kind: str) -> float:
    """Negative log-likelihood (events + right-censored) of a parametric survival model."""
    t = time
    if kind == "exponential":
        lam = np.exp(theta[0])
        logf = np.log(lam) - lam * t
        logS = -lam * t
    elif kind == "weibull":
        k, lam = np.exp(theta[0]), np.exp(theta[1])
        z = t / lam
        logf = np.log(k) - k * np.log(lam) + (k - 1) * np.log(t) - z**k
        logS = -(z**k)
    elif kind == "lognormal":
        mu, sigma = theta[0], np.exp(theta[1])
        z = (np.log(t) - mu) / sigma
        logf = -np.log(t) - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2
        logS = stats.norm.logsf(z)
    else:  # loglogistic
        alpha, beta = np.exp(theta[0]), np.exp(theta[1])
        z = (t / alpha) ** beta
        logf = np.log(beta) - np.log(alpha) + (beta - 1) * np.log(t / alpha) - 2 * np.log1p(z)
        logS = -np.log1p(z)
    return -float(np.sum(event * logf + (1 - event) * logS))


def _fit_parametric(time: np.ndarray, event: np.ndarray, kind: str):
    if kind == "exponential":
        x0 = [np.log(1 / max(time.mean(), 1e-9))]
    elif kind == "weibull":
        x0 = [0.0, np.log(time.mean())]
    elif kind == "lognormal":
        logt = np.log(time)
        x0 = [float(logt.mean()), float(np.log(max(logt.std(ddof=1), 1e-3)))]
    else:
        x0 = [np.log(time.mean()), 0.0]
    result = minimize(_parametric_loglik, x0, args=(time, event, kind), method="Nelder-Mead")
    if not result.success:
        result = minimize(_parametric_loglik, result.x, args=(time, event, kind), method="L-BFGS-B")
    return result


def _surv_at(kind: str, theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    if kind == "exponential":
        lam = np.exp(theta[0])
        return np.exp(-lam * t)
    if kind == "weibull":
        k, lam = np.exp(theta[0]), np.exp(theta[1])
        return np.exp(-((t / lam) ** k))
    if kind == "lognormal":
        mu, sigma = theta[0], np.exp(theta[1])
        return stats.norm.sf((np.log(t) - mu) / sigma)
    alpha, beta = np.exp(theta[0]), np.exp(theta[1])
    return 1 / (1 + (t / alpha) ** beta)


def _parametric_summary(kind: str, theta: np.ndarray) -> tuple[list[tuple], float, float | None]:
    if kind == "exponential":
        lam = np.exp(theta[0])
        return [("tasa (λ)", float(lam))], float(np.log(2) / lam), float(1 / lam)
    if kind == "weibull":
        k, lam = np.exp(theta[0]), np.exp(theta[1])
        return [("forma (k)", float(k)), ("escala (λ)", float(lam))], float(lam * np.log(2) ** (1 / k)), float(
            lam * _gamma(1 + 1 / k)
        )
    if kind == "lognormal":
        mu, sigma = theta[0], np.exp(theta[1])
        return [("μ", float(mu)), ("σ", float(sigma))], float(np.exp(mu)), float(np.exp(mu + sigma**2 / 2))
    alpha, beta = np.exp(theta[0]), np.exp(theta[1])
    mean = float(alpha * np.pi / beta / np.sin(np.pi / beta)) if beta > 1 else None
    return [("escala (α)", float(alpha)), ("forma (β)", float(beta))], float(alpha), mean


def _parametric(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    _data, time, event = _load_survival(frame, roles)
    kind = p["distribution"]
    result = _fit_parametric(time, event, kind)
    rows, median, mean = _parametric_summary(kind, result.x)
    out = Output(
        f"Supervivencia paramétrica: {dict(_DISTRIBUTIONS)[kind]}",
        f"n = {len(time)} · {int(event.sum())} eventos · máxima verosimilitud con censura",
    )
    out.tiles(
        [
            ("Observaciones", len(time)),
            ("Eventos", int(event.sum())),
            ("Mediana estimada", median),
            ("Media estimada", mean if mean is not None else "no definida"),
        ]
    )
    out.table("Parámetros", ["Parámetro", "Estimación"], rows, note="Estimados maximizando la verosimilitud con censura.")
    grid = np.linspace(1e-6, float(time.max()), 200)
    fitted = _surv_at(kind, result.x, grid)
    x_step, y_step = _step(*_km(time, event)[:2])
    out.chart(
        line_chart(
            "Ajuste paramétrico vs Kaplan-Meier",
            "Tiempo",
            "S(t)",
            [
                {"name": "Kaplan-Meier", "x": x_step, "y": y_step},
                {"name": dict(_DISTRIBUTIONS)[kind], "x": grid, "y": fitted, "dash": "dash"},
            ],
        )
    )
    out.dataset("fitted", "Supervivencia ajustada", pd.DataFrame({"tiempo": grid, "supervivencia": fitted}))
    return out.set(
        n=len(time),
        events=int(event.sum()),
        distribution=kind,
        median=median,
        mean=mean,
        loglikelihood=-float(result.fun),
    ).build()


def _parametric_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\n\n"
        + load_line(dataset)
        + f"time = df[{py_name(roles['time'])}].to_numpy(float)\nevent = df[{py_name(roles['event'])}].to_numpy(int)\n"
        + "# Ajuste por máxima verosimilitud con censura (ejemplo exponencial):\n"
        + "lam = event.sum() / time.sum()\nprint('tasa', lam, 'mediana', np.log(2) / lam)\n"
    )


# ============================================================================ catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "kaplan_meier",
        "Kaplan-Meier y comparación de grupos",
        GROUP_SURV,
        "Estima la función de supervivencia S(t) y compara dos o más grupos con log-rank (o sus variantes ponderadas).",
        _kaplan_meier,
        roles=(TIME, EVENT, GROUP),
        params=(FieldSpec("test", "Prueba de comparación", "select", "logrank", _LOG_TESTS), ALPHA),
        code=_kaplan_meier_code,
        example={"roles": {"time": "tm", "event": "ev", "group": "g"}, "params": {"test": "logrank"}},
    ),
    _spec(
        "cox",
        "Regresión de Cox (riesgos proporcionales)",
        GROUP_SURV,
        "Ajusta un modelo de riesgos proporcionales de Cox y reporta las razones de riesgo con intervalos y valores p.",
        _cox,
        roles=(TIME, EVENT, PREDICTORS),
        params=(ALPHA,),
        code=_cox_code,
        example={"roles": {"time": "tm", "event": "ev", "predictors": ["x1", "x2"]}, "params": {}},
    ),
    _spec(
        "parametric",
        "Supervivencia paramétrica (Weibull, exponencial, log-normal, log-logística)",
        GROUP_PARAM,
        "Ajusta un modelo paramétrico de supervivencia por máxima verosimilitud con censura y lo compara con Kaplan-Meier.",
        _parametric,
        roles=(TIME, EVENT),
        params=(FieldSpec("distribution", "Distribución", "select", "weibull", _DISTRIBUTIONS), ALPHA),
        outputs=(OutputSpec("fitted", "Supervivencia ajustada"),),
        code=_parametric_code,
        example={"roles": {"time": "tm", "event": "ev"}, "params": {"distribution": "weibull"}},
    ),
)

registry.register(*SPECS)
