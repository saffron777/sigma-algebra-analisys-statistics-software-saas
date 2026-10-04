"""Additional hypothesis tests (Minitab *Basic Statistics*/*Nonparametrics*/*Equivalence*, SPSS *Nonparametric Tests*)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats import contingency_tables as ct
from statsmodels.stats import rates
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.inter_rater import aggregate_raters, cohens_kappa, fleiss_kappa

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import interval_chart, line_chart, scatter_chart
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.plots import groups_of, level_order, need, numbers
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import InvalidInputError

MODULE = "pruebas"
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
ALTERNATIVE = FieldSpec(
    "alternative",
    "Hipótesis alternativa",
    "select",
    "two-sided",
    (("two-sided", "Distinta (≠)"), ("greater", "Mayor (>)"), ("less", "Menor (<)")),
)


# ============================================================================ helpers
def pfmt(value: float) -> str:
    return "< 0.0001" if value < 1e-4 else f"{value:.4f}"


def decision(out: Output, p: float, alpha: float, h0: str) -> None:
    reject = p < alpha
    out.alert(
        "warning" if reject else "success",
        f"p {'=' if p >= 1e-4 else ''} {pfmt(p)} {'<' if reject else '≥'} α = {alpha:g}: "
        f"{'se rechaza' if reject else 'no se rechaza'} H₀ ({h0}).",
    )


def two_groups(frame: pd.DataFrame, var: str, group: str) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    parts = groups_of(frame, var, group)
    if len(parts) != 2:
        raise InvalidInputError(f"«{group}» debe tener exactamente 2 categorías con datos (tiene {len(parts)}).")
    (a, x), (b, y) = parts
    return (a, x.to_numpy()), (b, y.to_numpy())


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def binary_levels(series: pd.Series, name: str) -> tuple[Any, Any]:
    """(negative, positive) levels of a column with two values (the larger one in sort order is the positive)."""
    levels = level_order(series.dropna())
    if len(levels) != 2:
        raise InvalidInputError(f"«{name}» debe tener exactamente 2 valores distintos (tiene {len(levels)}).")
    return levels[0], levels[1]


# ============================================================================ normality
_ANDERSON = {
    "norm": (stats.norm, "Normal"),
    "expon": (stats.expon, "Exponencial"),
    "logistic": (stats.logistic, "Logística"),
    "gumbel_r": (stats.gumbel_r, "Gumbel"),
}


def _qq(out: Output, x: np.ndarray, frozen, title: str, name: str) -> None:
    ordered = np.sort(x)
    theoretical = frozen.ppf((np.arange(1, len(x) + 1) - 0.3) / (len(x) + 0.4))
    low, high = float(min(theoretical.min(), ordered.min())), float(max(theoretical.max(), ordered.max()))
    out.chart(
        scatter_chart(
            title,
            "Cuantil teórico",
            name,
            [{"name": "Datos", "x": theoretical, "y": ordered}],
            lines=[{"name": "Ajuste perfecto", "x": np.array([low, high]), "y": np.array([low, high]), "dash": "dash"}],
        )
    )


def _anderson(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = roles["var"]
    x = numbers(frame, name).dropna().to_numpy()
    need(pd.Series(x), name, 8)
    dist, label = _ANDERSON[p["dist"]]
    result = stats.anderson(x, dist=p["dist"], method="interpolate")
    out = Output(f"Prueba de Anderson-Darling de {name}", f"distribución {label}")
    out.tiles([("n", len(x)), ("A²", float(result.statistic)), ("Valor p", P(float(result.pvalue)))])
    decision(out, float(result.pvalue), p["alpha"], f"los datos siguen una distribución {label.lower()}")
    out.table(
        "Resultado",
        ["Estadístico A²", "Valor p", "n"],
        [[float(result.statistic), P(float(result.pvalue)), len(x)]],
        note="El valor p se interpola de tablas con los parámetros estimados; queda entre 0.01 y 0.15 (los extremos indican «menor» o «mayor» que eso).",
    )
    _qq(out, x, dist(*dist.fit(x)), f"Gráfico de probabilidad ({label})", name)
    return out.set(statistic=float(result.statistic), p=float(result.pvalue), n=len(x)).build()


def _anderson_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"print(stats.anderson(df[{py_name(roles['var'])}].dropna(), dist={p['dist']!r}, method='interpolate'))\n"
    )


def _lilliefors(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = roles["var"]
    x = numbers(frame, name).dropna().to_numpy()
    need(pd.Series(x), name, 5)
    if p["dist"] == "exp" and (x < 0).any():
        raise InvalidInputError("La distribución exponencial necesita valores no negativos.")
    statistic, pvalue = lilliefors(x, dist=p["dist"], pvalmethod="table")
    label = "normal" if p["dist"] == "norm" else "exponencial"
    out = Output(f"Prueba de Lilliefors de {name}", f"distribución {label} con parámetros estimados")
    out.tiles([("n", len(x)), ("D de Kolmogorov-Smirnov", float(statistic)), ("Valor p", P(float(pvalue)))])
    decision(out, float(pvalue), p["alpha"], f"los datos siguen una distribución {label}")
    out.table(
        "Resultado",
        ["Estadístico D", "Valor p", "n"],
        [[float(statistic), P(float(pvalue)), len(x)]],
        note="Corrige la prueba de Kolmogorov-Smirnov porque los parámetros se estiman con los mismos datos; el valor p se obtiene de tablas (entre 0.001 y 0.2).",
    )
    xs, ys = ecdf(x)
    grid = np.linspace(x.min(), x.max(), 200)
    fitted = stats.norm(x.mean(), x.std(ddof=1)) if p["dist"] == "norm" else stats.expon(scale=x.mean())
    out.chart(
        line_chart(
            "Función de distribución empírica y ajustada",
            name,
            "Probabilidad acumulada",
            [
                {"name": "Empírica", "x": xs, "y": ys},
                {"name": f"Ajustada ({label})", "x": grid, "y": fitted.cdf(grid), "dash": "dash"},
            ],
        )
    )
    return out.set(statistic=float(statistic), p=float(pvalue), n=len(x)).build()


def _lilliefors_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.stats.diagnostic import lilliefors\n\n"
        + load_line(dataset)
        + f"print(lilliefors(df[{py_name(roles['var'])}].dropna(), dist={p['dist']!r}))\n"
    )


_KS_DISTS = (
    ("norm", "Normal (ubicación = media, escala = desv. estándar)"),
    ("uniform", "Uniforme (ubicación = mínimo, escala = amplitud)"),
    ("expon", "Exponencial (ubicación = 0, escala = media)"),
    ("lognorm", "Lognormal (σ = escala; ubicación = 0)"),
)


def _ks_one(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = roles["var"]
    x = numbers(frame, name).dropna().to_numpy()
    need(pd.Series(x), name, 5)
    frozen = {
        "norm": lambda: stats.norm(p["loc"], p["scale"]),
        "uniform": lambda: stats.uniform(p["loc"], p["scale"]),
        "expon": lambda: stats.expon(p["loc"], p["scale"]),
        "lognorm": lambda: stats.lognorm(p["scale"], 0, np.exp(p["loc"])),
    }[p["dist"]]()
    result = stats.kstest(x, frozen.cdf, alternative=p["alternative"])
    out = Output(f"Kolmogorov-Smirnov de {name}", "frente a una distribución completamente especificada")
    out.tiles([("n", len(x)), ("D", float(result.statistic)), ("Valor p", P(float(result.pvalue)))])
    decision(out, float(result.pvalue), p["alpha"], "los datos provienen de esa distribución")
    out.table(
        "Distribución de referencia",
        ["Distribución", "Ubicación", "Escala"],
        [[dict(_KS_DISTS)[p["dist"]].split(" (")[0], p["loc"], p["scale"]]],
        note="Si los parámetros se estimaron de estos datos, use la prueba de Lilliefors o Anderson-Darling.",
    )
    xs, ys = ecdf(x)
    grid = np.linspace(min(x.min(), frozen.ppf(0.001)), max(x.max(), frozen.ppf(0.999)), 250)
    out.chart(
        line_chart(
            "Empírica frente a teórica",
            name,
            "Probabilidad acumulada",
            [
                {"name": "Empírica", "x": xs, "y": ys},
                {"name": "Teórica", "x": grid, "y": frozen.cdf(grid), "dash": "dash"},
            ],
        )
    )
    return out.set(statistic=float(result.statistic), p=float(result.pvalue)).build()


def _ks_one_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"print(stats.kstest(df[{py_name(roles['var'])}].dropna(), stats.{p['dist']}({p['loc']}, {p['scale']}).cdf))\n"
    )


def _jarque_bera(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = roles["var"]
    x = numbers(frame, name).dropna().to_numpy()
    need(pd.Series(x), name, 8)
    result = stats.jarque_bera(x)
    skew, kurt = float(stats.skew(x)), float(stats.kurtosis(x, fisher=False))
    out = Output(f"Prueba de Jarque-Bera de {name}", "normalidad según asimetría y curtosis")
    out.tiles(
        [
            ("n", len(x)),
            ("JB", float(result.statistic)),
            ("Valor p", P(float(result.pvalue))),
            ("Asimetría", skew),
            ("Curtosis", kurt),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], "asimetría = 0 y curtosis = 3, como la normal")
    out.table(
        "Resultado",
        ["JB", "Valor p", "Asimetría", "Curtosis (normal = 3)"],
        [[float(result.statistic), P(float(result.pvalue)), skew, kurt]],
    )
    return out.set(statistic=float(result.statistic), p=float(result.pvalue), skew=skew, kurtosis=kurt).build()


def _jarque_bera_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"print(stats.jarque_bera(df[{py_name(roles['var'])}].dropna()))\n"
    )


def _ks_two(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    (la, a), (lb, b) = two_groups(frame, roles["var"], roles["group"])
    result = stats.ks_2samp(a, b, alternative=p["alternative"])
    out = Output(f"Kolmogorov-Smirnov de dos muestras: {roles['var']}", f"{la} frente a {lb}")
    out.tiles(
        [
            (f"n ({la})", len(a)),
            (f"n ({lb})", len(b)),
            ("D", float(result.statistic)),
            ("Valor p", P(float(result.pvalue))),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], "las dos muestras provienen de la misma distribución")
    (xa, ya), (xb, yb) = ecdf(a), ecdf(b)
    out.chart(
        line_chart(
            "Funciones de distribución empíricas",
            roles["var"],
            "Probabilidad acumulada",
            [{"name": la, "x": xa, "y": ya}, {"name": lb, "x": xb, "y": yb}],
        )
    )
    out.table(
        "Resumen",
        ["Grupo", "n", "Media", "Mediana", "Desv. est."],
        [
            [la, len(a), float(a.mean()), float(np.median(a)), float(a.std(ddof=1))],
            [lb, len(b), float(b.mean()), float(np.median(b)), float(b.std(ddof=1))],
        ],
    )
    return out.set(statistic=float(result.statistic), p=float(result.pvalue)).build()


def _ks_two_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"a, b = (g[{py_name(roles['var'])}].dropna() for _, g in df.groupby({py_name(roles['group'])}))\nprint(stats.ks_2samp(a, b, alternative={p['alternative']!r}))\n"
    )


# ============================================================================ equal variances
def _variances(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, group, alpha = roles["var"], roles["group"], p["alpha"]
    parts = [(label, values.to_numpy()) for label, values in groups_of(frame, var, group)]
    if len(parts) < 2:
        raise InvalidInputError("Se necesitan al menos 2 grupos con datos.")
    for label, values in parts:
        if len(values) < 2:
            raise InvalidInputError(f"El grupo «{label}» necesita al menos 2 datos.")
    samples = [v for _, v in parts]
    k, n = len(samples), sum(len(v) for v in samples)
    tests = [
        ("Bartlett (supone normalidad)", stats.bartlett(*samples), k - 1),
        ("Levene (centrada en la media)", stats.levene(*samples, center="mean"), (k - 1, n - k)),
        ("Brown-Forsythe / Levene (mediana)", stats.levene(*samples, center="median"), (k - 1, n - k)),
        ("Fligner-Killeen (robusta)", stats.fligner(*samples), k - 1),
    ]
    out = Output(f"Igualdad de varianzas de {var}", f"{k} grupos de «{group}»")
    out.tiles(
        [
            ("Grupos", k),
            ("n total", n),
            (
                "Razón máx/mín de varianzas",
                float(max(v.var(ddof=1) for v in samples) / min(v.var(ddof=1) for v in samples)),
            ),
        ]
    )
    rows = [
        [
            name,
            float(res.statistic),
            str(df if not isinstance(df, tuple) else f"{df[0]}, {df[1]}"),
            P(float(res.pvalue)),
        ]
        for name, res, df in tests
    ]
    out.table(
        "Pruebas de homogeneidad de varianzas",
        ["Prueba", "Estadístico", "gl", "Valor p"],
        rows,
        note="Con datos no normales prefiera Brown-Forsythe o Fligner-Killeen.",
    )
    robust = tests[2][1].pvalue
    decision(out, float(robust), alpha, "todas las varianzas son iguales (Brown-Forsythe)")
    adjusted = alpha / k
    groups, table = [], []
    for label, values in parts:
        df = len(values) - 1
        sd = float(values.std(ddof=1))
        low = sd * np.sqrt(df / stats.chi2.ppf(1 - adjusted / 2, df))
        high = sd * np.sqrt(df / stats.chi2.ppf(adjusted / 2, df))
        groups.append({"name": label, "mean": sd, "lower": float(low), "upper": float(high)})
        table.append([label, len(values), float(values.var(ddof=1)), sd, float(low), float(high)])
    out.chart(
        interval_chart(
            f"IC de Bonferroni del {100 * (1 - alpha):g} % para las desviaciones estándar",
            group,
            "Desviación estándar",
            groups,
        )
    )
    out.table("Desviaciones estándar por grupo", ["Grupo", "n", "Varianza", "Desv. est.", "IC inf.", "IC sup."], table)
    return out.set(
        levene_p=float(tests[1][1].pvalue),
        brown_forsythe_p=float(robust),
        bartlett_p=float(tests[0][1].pvalue),
        fligner_p=float(tests[3][1].pvalue),
    ).build()


def _variances_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"groups = [g[{py_name(roles['var'])}].dropna() for _, g in df.groupby({py_name(roles['group'])})]\n"
        + "print(stats.bartlett(*groups))\nprint(stats.levene(*groups, center='mean'))\nprint(stats.levene(*groups, center='median'))\nprint(stats.fligner(*groups))\n"
    )


# ============================================================================ nonparametric
def _sign_test(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, other = roles["var"], roles.get("var2")
    if other:
        data = pd.DataFrame({"a": numbers(frame, var), "b": numbers(frame, other)}).dropna()
        values = (data["a"] - data["b"]).to_numpy()
        label, target = f"mediana de las diferencias ({var} − {other})", 0.0
    else:
        values = numbers(frame, var).dropna().to_numpy()
        label, target = f"mediana de {var}", float(p["median0"])
    need(pd.Series(values), var, 5)
    centred = values - target
    positive, negative = int((centred > 0).sum()), int((centred < 0).sum())
    n_eff = positive + negative
    if n_eff == 0:
        raise InvalidInputError("Todos los datos coinciden con el valor de referencia: no hay signos que contar.")
    result = stats.binomtest(positive, n_eff, 0.5, alternative=p["alternative"])
    ordered, n = np.sort(values), len(values)
    k = 0
    while k + 1 <= n // 2 and 2 * stats.binom.cdf(k, n, 0.5) <= p["alpha"]:
        k += 1
    out = Output("Prueba de los signos", label)
    out.tiles(
        [
            ("n", n),
            ("Por encima", positive),
            ("Por debajo", negative),
            ("Empates", n - n_eff),
            ("Valor p", P(float(result.pvalue))),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], f"la {label} es {target:g}")
    out.table(
        "Resultado",
        ["Mediana de la muestra", "Valor de referencia", "Por encima", "Por debajo", "Valor p (exacto)"],
        [[float(np.median(values)), target, positive, negative, P(float(result.pvalue))]],
    )
    if k >= 1:
        confidence = 1 - 2 * stats.binom.cdf(k - 1, n, 0.5)
        out.table(
            "Intervalo de confianza para la mediana",
            ["Confianza lograda", "Inferior", "Superior"],
            [[float(confidence), float(ordered[k - 1]), float(ordered[n - k])]],
            note="Basado en estadísticos de orden: la confianza solo puede tomar ciertos valores discretos.",
        )
    return out.set(positive=positive, negative=negative, p=float(result.pvalue)).build()


def _sign_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if roles.get("var2"):
        values = f"(df[{py_name(roles['var'])}] - df[{py_name(roles['var2'])}]).dropna()"
        center = "0"
    else:
        values = f"df[{py_name(roles['var'])}].dropna()"
        center = repr(p["median0"])
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"d = {values} - {center}\nprint(stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5, alternative={p['alternative']!r}))\n"
    )


def _mood_median(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    var, group = roles["var"], roles["group"]
    parts = [(label, v.to_numpy()) for label, v in groups_of(frame, var, group)]
    if len(parts) < 2:
        raise InvalidInputError("Se necesitan al menos 2 grupos con datos.")
    statistic, pvalue, grand, table = stats.median_test(*[v for _, v in parts])
    out = Output(f"Prueba de la mediana de Mood de {var}", f"{len(parts)} grupos de «{group}»")
    out.tiles([("Mediana global", float(grand)), ("χ²", float(statistic)), ("Valor p", P(float(pvalue)))])
    decision(out, float(pvalue), p["alpha"], "todos los grupos tienen la misma mediana")
    out.table(
        "Datos por encima y por debajo de la mediana global",
        ["Grupo", "n", "Mediana", "> mediana global", "≤ mediana global"],
        [
            [label, len(v), float(np.median(v)), int(table[0][i]), int(table[1][i])]
            for i, (label, v) in enumerate(parts)
        ],
    )
    return out.set(statistic=float(statistic), p=float(pvalue), grand_median=float(grand)).build()


def _mood_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"groups = [g[{py_name(roles['var'])}].dropna() for _, g in df.groupby({py_name(roles['group'])})]\nprint(stats.median_test(*groups))\n"
    )


def _friedman(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise InvalidInputError("Se necesitan al menos 3 filas completas (bloques).")
    result = stats.friedmanchisquare(*[data[c].to_numpy() for c in names])
    k, n = len(names), len(data)
    ranks = data.rank(axis=1)
    kendall_w = float(result.statistic / (n * (k - 1)))
    out = Output("Prueba de Friedman", f"{k} tratamientos en {n} bloques (filas completas)")
    out.tiles(
        [
            ("Bloques", n),
            ("Tratamientos", k),
            ("χ²", float(result.statistic)),
            ("Valor p", P(float(result.pvalue))),
            ("W de Kendall", kendall_w),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], "los tratamientos no difieren")
    out.table(
        "Rangos medios por tratamiento",
        ["Tratamiento", "Mediana", "Rango medio", "Suma de rangos"],
        [[c, float(data[c].median()), float(ranks[c].mean()), float(ranks[c].sum())] for c in names],
        note="W de Kendall = concordancia entre bloques (0 a 1).",
    )
    if len(data) < len(frame):
        out.alert("info", f"Se omitieron {len(frame) - len(data)} fila(s) con datos faltantes.")
    return out.set(statistic=float(result.statistic), p=float(result.pvalue), kendall_w=kendall_w).build()


def _friedman_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"d = df[{roles['vars']!r}].dropna()\nprint(stats.friedmanchisquare(*[d[c] for c in d.columns]))\n"
    )


def _runs(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name = roles["var"]
    x = numbers(frame, name).dropna().to_numpy()
    cutoff = {
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "value": float(p["value"] if p["value"] is not None else np.median(x)),
    }[p["cutoff"]]
    kept = x[x != cutoff]
    need(pd.Series(kept), name, 10)
    above = kept > cutoff
    n1, n2 = int(above.sum()), int((~above).sum())
    if n1 == 0 or n2 == 0:
        raise InvalidInputError("Todos los datos quedan del mismo lado del valor de corte: no hay rachas que contar.")
    runs = 1 + int((above[1:] != above[:-1]).sum())
    expected = 2 * n1 * n2 / (n1 + n2) + 1
    z, pvalue = runstest_1samp(x, cutoff=cutoff, correction=False)
    out = Output(f"Prueba de rachas de {name}", f"aleatoriedad respecto a {cutoff:.6g}")
    out.tiles(
        [
            ("Rachas observadas", runs),
            ("Rachas esperadas", float(expected)),
            ("Z", float(z)),
            ("Valor p", P(float(pvalue))),
        ]
    )
    decision(out, float(pvalue), p["alpha"], "la secuencia es aleatoria")
    out.table(
        "Resultado",
        ["Corte", "Por encima", "Por debajo / igual", "Rachas", "Esperadas", "Z", "Valor p"],
        [[cutoff, n1, n2, runs, float(expected), float(z), P(float(pvalue))]],
        note="Pocas rachas indican tendencia o agrupamiento; demasiadas, oscilación sistemática. Los datos iguales al corte se omiten.",
    )
    out.chart(
        line_chart(
            "Datos y valor de corte",
            "Orden",
            name,
            [{"name": name, "x": np.arange(1, len(x) + 1), "y": x}],
            hlines=[{"value": cutoff, "label": "corte"}],
        )
    )
    return out.set(runs=runs, expected=float(expected), z=float(z), p=float(pvalue)).build()


def _runs_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.sandbox.stats.runs import runstest_1samp\n\n"
        + load_line(dataset)
        + f"print(runstest_1samp(df[{py_name(roles['var'])}].dropna(), cutoff={p['cutoff'] if p['cutoff'] != 'value' else p['value']!r}))\n"
    )


def _mcnemar(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    a_name, b_name = roles["a"], roles["b"]
    data = frame[[a_name, b_name]].dropna()
    na, pa = binary_levels(data[a_name], a_name)
    nb, pb = binary_levels(data[b_name], b_name)
    first, second = data[a_name] == pa, data[b_name] == pb
    table = np.array(
        [
            [int((first & second).sum()), int((first & ~second).sum())],
            [int((~first & second).sum()), int((~first & ~second).sum())],
        ]
    )
    discordant = int(table[0, 1] + table[1, 0])
    if discordant == 0:
        raise InvalidInputError("No hay pares discordantes: las dos mediciones coinciden en todos los casos.")
    exact = p["method"] == "exact" or (p["method"] == "auto" and discordant < 25)
    result = ct.mcnemar(table, exact=exact, correction=not exact)
    out = Output("Prueba de McNemar", f"{a_name} frente a {b_name} (positivo = {pa} / {pb})")
    out.tiles(
        [
            ("Pares", int(table.sum())),
            ("Discordantes", discordant),
            ("Valor p", P(float(result.pvalue))),
            ("Método", "exacto (binomial)" if exact else "χ² con corrección"),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], "las dos mediciones tienen la misma proporción de positivos")
    out.table(
        "Tabla de pares",
        ["", f"{b_name} positivo", f"{b_name} negativo"],
        [
            [f"{a_name} positivo", int(table[0, 0]), int(table[0, 1])],
            [f"{a_name} negativo", int(table[1, 0]), int(table[1, 1])],
        ],
    )
    out.table(
        "Proporciones de positivos",
        ["Medición", "Proporción"],
        [[a_name, float(first.mean())], [b_name, float(second.mean())]],
    )
    return out.set(p=float(result.pvalue), discordant=discordant).build()


def _mcnemar_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import pandas as pd\nfrom statsmodels.stats.contingency_tables import mcnemar\n\n"
        + load_line(dataset)
        + f"table = pd.crosstab(df[{py_name(roles['a'])}], df[{py_name(roles['b'])}]).to_numpy()\nprint(mcnemar(table, exact={p['method'] != 'chi2'}))\n"
    )


def _cochran(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["vars"]
    data = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5:
        raise InvalidInputError("Se necesitan al menos 5 filas completas.")
    if not set(np.unique(data.to_numpy())) <= {0, 1}:
        raise InvalidInputError("Las columnas deben contener solo 0 (fracaso) y 1 (éxito).")
    result = ct.cochrans_q(data.to_numpy())
    out = Output("Q de Cochran", f"{len(names)} mediciones binarias en {len(data)} sujetos")
    out.tiles(
        [
            ("Sujetos", len(data)),
            ("Q", float(result.statistic)),
            ("gl", int(result.df)),
            ("Valor p", P(float(result.pvalue))),
        ]
    )
    decision(out, float(result.pvalue), p["alpha"], "la proporción de éxitos es la misma en todas las mediciones")
    out.table(
        "Éxitos por medición",
        ["Medición", "Éxitos", "Proporción"],
        [[c, int(data[c].sum()), float(data[c].mean())] for c in names],
    )
    return out.set(statistic=float(result.statistic), p=float(result.pvalue)).build()


def _cochran_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.stats.contingency_tables import cochrans_q\n\n"
        + load_line(dataset)
        + f"print(cochrans_q(df[{roles['vars']!r}].dropna().to_numpy()))\n"
    )


# ============================================================================ rates and tables
def _poisson(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    alpha, alt = p["alpha"], p["alternative"]
    if p["design"] == "one":
        count, exposure, value = int(p["count1"]), float(p["exposure1"]), float(p["rate0"])
        if value <= 0:
            raise InvalidInputError("La tasa de referencia debe ser mayor que 0.")
        test = rates.test_poisson(count, exposure, value, method=p["method1"], alternative=alt)
        low, high = rates.confint_poisson(count, exposure, method=p["method1"], alpha=alpha)
        out = Output("Tasa de Poisson (una muestra)", f"{count} eventos en {exposure:g} unidades de exposición")
        out.tiles([("Tasa observada", count / exposure), ("Valor p", P(float(test.pvalue)))])
        decision(out, float(test.pvalue), alpha, f"la tasa es {value:g}")
        out.table(
            "Resultado",
            ["Eventos", "Exposición", "Tasa", f"IC {100 * (1 - alpha):g} % inf.", "IC sup.", "Tasa H₀", "Valor p"],
            [[count, exposure, count / exposure, float(low), float(high), value, P(float(test.pvalue))]],
            note=f"Método: {p['method1']}.",
        )
        return out.set(rate=count / exposure, p=float(test.pvalue)).build()
    c1, e1, c2, e2 = int(p["count1"]), float(p["exposure1"]), int(p["count2"]), float(p["exposure2"])
    test = rates.test_poisson_2indep(c1, e1, c2, e2, value=float(p["ratio0"]), method=p["method2"], alternative=alt)
    low, high = rates.confint_poisson_2indep(c1, e1, c2, e2, method="score", alpha=alpha)
    out = Output("Comparación de dos tasas de Poisson", f"{c1}/{e1:g} frente a {c2}/{e2:g}")
    out.tiles(
        [
            ("Tasa 1", c1 / e1),
            ("Tasa 2", c2 / e2),
            ("Razón de tasas", float(test.ratio)),
            ("Valor p", P(float(test.pvalue))),
        ]
    )
    decision(out, float(test.pvalue), alpha, f"la razón de tasas es {p['ratio0']:g}")
    out.table(
        "Resultado", ["Muestra", "Eventos", "Exposición", "Tasa"], [["1", c1, e1, c1 / e1], ["2", c2, e2, c2 / e2]]
    )
    out.table(
        "Razón de tasas (1 / 2)",
        ["Razón", f"IC {100 * (1 - alpha):g} % (score) inf.", "sup.", "Razón H₀", "Valor p"],
        [[float(test.ratio), float(low), float(high), p["ratio0"], P(float(test.pvalue))]],
        note=f"Prueba: {p['method2']}.",
    )
    return out.set(ratio=float(test.ratio), p=float(test.pvalue)).build()


def _poisson_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if p["design"] == "one":
        return f"from statsmodels.stats import rates\n\nprint(rates.test_poisson({int(p['count1'])}, {p['exposure1']}, {p['rate0']}, method={p['method1']!r}))\n"
    return f"from statsmodels.stats import rates\n\nprint(rates.test_poisson_2indep({int(p['count1'])}, {p['exposure1']}, {int(p['count2'])}, {p['exposure2']}, value={p['ratio0']}, method={p['method2']!r}))\n"


def _table(data: pd.DataFrame, exposure: str, outcome: str, e_level: Any, o_level: Any) -> np.ndarray:
    """Counts [[exposed & event, exposed & no event], [unexposed & event, unexposed & no event]]."""
    exposed, event = data[exposure] == e_level, data[outcome] == o_level
    return np.array(
        [
            [int((exposed & event).sum()), int((exposed & ~event).sum())],
            [int((~exposed & event).sum()), int((~exposed & ~event).sum())],
        ]
    )


def _odds_ratio(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    exposure, outcome, alpha = roles["exposure"], roles["outcome"], p["alpha"]
    data = frame[[exposure, outcome]].dropna()
    _, e_level = binary_levels(data[exposure], exposure)
    _, o_level = binary_levels(data[outcome], outcome)
    table = _table(data, exposure, outcome, e_level, o_level)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        raise InvalidInputError("La tabla tiene una fila o columna vacía.")
    t = ct.Table2x2(table, shift_zeros=True)
    or_low, or_high = t.oddsratio_confint(alpha=alpha)
    rr_low, rr_high = t.riskratio_confint(alpha=alpha)
    risk_exposed, risk_unexposed = table[0, 0] / table[0].sum(), table[1, 0] / table[1].sum()
    fisher = stats.fisher_exact(table)
    chi = stats.chi2_contingency(table, correction=False)
    out = Output(
        f"Odds ratio y riesgo relativo: {outcome} según {exposure}", f"expuesto = {e_level} · evento = {o_level}"
    )
    out.tiles(
        [
            ("Odds ratio", float(t.oddsratio)),
            ("Riesgo relativo", float(t.riskratio)),
            ("Diferencia de riesgos", float(risk_exposed - risk_unexposed)),
            ("p (Fisher)", P(float(fisher.pvalue))),
        ]
    )
    decision(out, float(fisher.pvalue), alpha, "no hay asociación entre la exposición y el evento")
    out.table(
        "Tabla 2×2",
        ["", f"{outcome} = {o_level}", "Sin el evento", "Total", "Riesgo"],
        [
            [f"{exposure} = {e_level}", int(table[0, 0]), int(table[0, 1]), int(table[0].sum()), float(risk_exposed)],
            ["No expuestos", int(table[1, 0]), int(table[1, 1]), int(table[1].sum()), float(risk_unexposed)],
        ],
    )
    ci = f"IC {100 * (1 - alpha):g} %"
    out.table(
        "Medidas de asociación",
        ["Medida", "Estimación", f"{ci} inf.", f"{ci} sup."],
        [
            ["Odds ratio", float(t.oddsratio), float(or_low), float(or_high)],
            ["Riesgo relativo", float(t.riskratio), float(rr_low), float(rr_high)],
        ],
        note="Si hay celdas en cero se suma 0.5 a todas (corrección de Haldane).",
    )
    out.table(
        "Pruebas de asociación",
        ["Prueba", "Estadístico", "Valor p"],
        [
            ["Exacta de Fisher (razón de odds)", float(fisher.statistic), P(float(fisher.pvalue))],
            ["χ² de Pearson", float(chi.statistic), P(float(chi.pvalue))],
        ],
    )
    return out.set(odds_ratio=float(t.oddsratio), risk_ratio=float(t.riskratio), fisher_p=float(fisher.pvalue)).build()


def _odds_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import pandas as pd\nfrom statsmodels.stats.contingency_tables import Table2x2\n\n"
        + load_line(dataset)
        + f"table = pd.crosstab(df[{py_name(roles['exposure'])}], df[{py_name(roles['outcome'])}]).iloc[::-1, ::-1].to_numpy()\nt = Table2x2(table)\nprint(t.summary())\n"
    )


def _cmh(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    exposure, outcome, stratum, alpha = roles["exposure"], roles["outcome"], roles["stratum"], p["alpha"]
    data = frame[[exposure, outcome, stratum]].dropna()
    _, e_level = binary_levels(data[exposure], exposure)
    _, o_level = binary_levels(data[outcome], outcome)
    tables, rows = [], []
    for level in level_order(data[stratum]):
        table = _table(data[data[stratum] == level], exposure, outcome, e_level, o_level)
        if table.sum() == 0:
            continue
        tables.append(table)
        rows.append([str(level), int(table.sum()), float(ct.Table2x2(table, shift_zeros=True).oddsratio)])
    if len(tables) < 2:
        raise InvalidInputError("Se necesitan al menos 2 estratos con datos.")
    st = ct.StratifiedTable(tables)
    pooled_low, pooled_high = st.oddsratio_pooled_confint(alpha=alpha)
    cmh = st.test_null_odds(correction=False)
    homogeneity = st.test_equal_odds(adjust=False)
    out = Output(
        f"Cochran-Mantel-Haenszel: {outcome} según {exposure}", f"estratificado por {stratum} ({len(tables)} estratos)"
    )
    out.tiles(
        [
            ("Odds ratio común", float(st.oddsratio_pooled)),
            ("χ² de CMH", float(cmh.statistic)),
            ("p (CMH)", P(float(cmh.pvalue))),
            ("p (Breslow-Day)", P(float(homogeneity.pvalue))),
        ]
    )
    decision(out, float(cmh.pvalue), alpha, "la exposición y el evento son independientes en todos los estratos")
    out.table("Odds ratio por estrato", [stratum, "n", "Odds ratio"], rows)
    ci = f"IC {100 * (1 - alpha):g} %"
    out.table(
        "Resultado combinado",
        ["Medida", "Estimación", f"{ci} inf.", f"{ci} sup.", "Valor p"],
        [
            [
                "Odds ratio común (Mantel-Haenszel)",
                float(st.oddsratio_pooled),
                float(pooled_low),
                float(pooled_high),
                P(float(cmh.pvalue)),
            ]
        ],
    )
    heterogeneous = float(homogeneity.pvalue) < alpha
    out.alert(
        "warning" if heterogeneous else "info",
        f"Homogeneidad de los odds ratio (Breslow-Day): p = {pfmt(float(homogeneity.pvalue))}. "
        + (
            "Los odds ratio difieren entre estratos: el valor combinado puede no ser representativo."
            if heterogeneous
            else "No hay evidencia de que difieran entre estratos."
        ),
    )
    return out.set(
        pooled_or=float(st.oddsratio_pooled), cmh_p=float(cmh.pvalue), breslow_day_p=float(homogeneity.pvalue)
    ).build()


def _cmh_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import pandas as pd\nfrom statsmodels.stats.contingency_tables import StratifiedTable\n\n"
        + load_line(dataset)
        + f"tables = [pd.crosstab(g[{py_name(roles['exposure'])}], g[{py_name(roles['outcome'])}]).iloc[::-1, ::-1].to_numpy()\n          for _, g in df.groupby({py_name(roles['stratum'])})]\n"
        + "st = StratifiedTable(tables)\nprint(st.summary())\n"
    )


# ============================================================================ agreement
def _landis_koch(kappa: float) -> str:
    for limit, label in ((0, "pobre"), (0.2, "leve"), (0.4, "regular"), (0.6, "moderado"), (0.8, "sustancial")):
        if kappa <= limit:
            return label
    return "casi perfecto"


def _kappa(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["raters"]
    data = frame[names].dropna()
    if len(data) < 5:
        raise InvalidInputError("Se necesitan al menos 5 sujetos calificados por todos los evaluadores.")
    labels = [
        str(c) for c in sorted({v for c in names for v in data[c].unique()}, key=lambda v: (isinstance(v, str), v))
    ]
    if len(labels) < 2 or len(labels) > 30:
        raise InvalidInputError("Las calificaciones deben tener entre 2 y 30 categorías distintas.")
    coded = data.apply(lambda col: col.map(lambda v: labels.index(str(v))))
    agreement = float(np.mean([len(set(row)) == 1 for row in coded.to_numpy()]))
    out = Output(
        "Acuerdo entre evaluadores (kappa)",
        f"{len(names)} evaluadores · {len(data)} sujetos · {len(labels)} categorías",
    )
    if len(names) == 2:
        table = (
            pd.crosstab(coded[names[0]], coded[names[1]])
            .reindex(index=range(len(labels)), columns=range(len(labels)), fill_value=0)
            .to_numpy()
        )
        weights = None if p["weights"] == "none" else p["weights"]
        result = cohens_kappa(table, wt=weights, return_results=True)
        kappa, low, high = float(result.kappa), float(result.kappa_low), float(result.kappa_upp)
        name, pvalue = (
            ("Kappa de Cohen" + ("" if weights is None else f" ponderado ({weights})")),
            float(result.pvalue_two_sided),
        )
        out.tiles(
            [
                ("Kappa", kappa),
                ("IC 95 %", f"{low:.3f} a {high:.3f}"),
                ("Acuerdo observado", agreement),
                ("Valor p", P(pvalue)),
            ]
        )
        out.table(
            "Tabla de calificaciones",
            [f"{names[0]} \\ {names[1]}", *labels],
            [[labels[i], *(int(v) for v in table[i])] for i in range(len(labels))],
        )
        if weights:
            out.alert("info", "El kappa ponderado supone categorías ordinales, en el orden en que aparecen arriba.")
    else:
        counts, _ = aggregate_raters(coded.to_numpy(), n_cat=len(labels))
        kappa = float(fleiss_kappa(counts, method="fleiss"))
        name, low, high, pvalue = "Kappa de Fleiss", None, None, None
        out.tiles([("Kappa", kappa), ("Acuerdo total", agreement)])
        out.table(
            "Distribución de calificaciones",
            ["Categoría", "Calificaciones", "%"],
            [
                [labels[i], int(counts[:, i].sum()), 100 * float(counts[:, i].sum() / counts.sum())]
                for i in range(len(labels))
            ],
        )
    out.table("Resultado", ["Medida", "Valor", "Interpretación (Landis y Koch)"], [[name, kappa, _landis_koch(kappa)]])
    return out.set(kappa=kappa, agreement=agreement, n=len(data)).build()


def _kappa_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if len(roles["raters"]) == 2:
        return (
            "import pandas as pd\nfrom statsmodels.stats.inter_rater import cohens_kappa\n\n"
            + load_line(dataset)
            + f"table = pd.crosstab(df[{py_name(roles['raters'][0])}], df[{py_name(roles['raters'][1])}]).to_numpy()\nprint(cohens_kappa(table, wt={None if p['weights'] == 'none' else p['weights']!r}))\n"
        )
    return (
        "from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa\n\n"
        + load_line(dataset)
        + f"counts, _ = aggregate_raters(df[{roles['raters']!r}].dropna().to_numpy())\nprint(fleiss_kappa(counts))\n"
    )


_ICC_FORMS = (
    ("ICC(1,1)", "Un factor aleatorio, una medición"),
    ("ICC(2,1)", "Dos factores aleatorios, una medición (acuerdo absoluto)"),
    ("ICC(3,1)", "Evaluadores fijos, una medición (consistencia)"),
    ("ICC(1,k)", "Un factor aleatorio, promedio de k mediciones"),
    ("ICC(2,k)", "Dos factores aleatorios, promedio de k"),
    ("ICC(3,k)", "Evaluadores fijos, promedio de k"),
)


def icc_table(data: np.ndarray, alpha: float = 0.05) -> list[dict[str, float]]:
    """The six intraclass correlations of Shrout and Fleiss (1979) with their F tests and McGraw-Wong intervals."""
    n, k = data.shape
    grand = data.mean()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    msr, msc = ss_rows / (n - 1), ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_error) / (n * (k - 1))
    f_low = 1 - alpha / 2
    results = []

    def add(label, value, f, df1, df2, low, high):
        results.append(
            {
                "label": label,
                "icc": float(value),
                "f": float(f),
                "df1": float(df1),
                "df2": float(df2),
                "p": float(stats.f.sf(f, df1, df2)),
                "low": float(low),
                "high": float(high),
            }
        )

    # ICC(1,1)
    icc1 = (msr - msw) / (msr + (k - 1) * msw)
    f = msr / msw
    fl, fu = f / stats.f.ppf(f_low, n - 1, n * (k - 1)), f * stats.f.ppf(f_low, n * (k - 1), n - 1)
    add("ICC(1,1)", icc1, f, n - 1, n * (k - 1), (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    # ICC(2,1)
    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    f = msr / mse
    fj = msc / mse
    a = k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2
    vn = (k - 1) * (n - 1) * a**2 / ((n - 1) * k**2 * icc2**2 * fj**2 + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2)
    fl2, fu2 = stats.f.ppf(f_low, n - 1, vn), stats.f.ppf(f_low, vn, n - 1)
    denom = k * msc + (k * n - k - n) * mse
    low2 = n * (msr - fl2 * mse) / (fl2 * denom + n * msr)
    high2 = n * (fu2 * msr - mse) / (denom + n * fu2 * msr)
    add("ICC(2,1)", icc2, f, n - 1, (n - 1) * (k - 1), low2, high2)
    # ICC(3,1)
    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    fl3, fu3 = f / stats.f.ppf(f_low, n - 1, (n - 1) * (k - 1)), f * stats.f.ppf(f_low, (n - 1) * (k - 1), n - 1)
    add("ICC(3,1)", icc3, f, n - 1, (n - 1) * (k - 1), (fl3 - 1) / (fl3 + k - 1), (fu3 - 1) / (fu3 + k - 1))
    # average measures
    single = results[:]
    average = [
        ("ICC(1,k)", (msr - msw) / msr, msr / msw, n - 1, n * (k - 1)),
        ("ICC(2,k)", (msr - mse) / (msr + (msc - mse) / n), msr / mse, n - 1, (n - 1) * (k - 1)),
        ("ICC(3,k)", (msr - mse) / msr, msr / mse, n - 1, (n - 1) * (k - 1)),
    ]
    for (label, value, f, df1, df2), base in zip(average, single, strict=True):
        add(
            label,
            value,
            f,
            df1,
            df2,
            k * base["low"] / (1 + (k - 1) * base["low"]),
            k * base["high"] / (1 + (k - 1) * base["high"]),
        )
    return results


def _icc(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names = roles["raters"]
    data = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise InvalidInputError("Se necesitan al menos 3 sujetos con todas las mediciones.")
    results = icc_table(data.to_numpy(), p["alpha"])
    out = Output("Coeficiente de correlación intraclase", f"{len(names)} evaluadores · {len(data)} sujetos")
    out.tiles([("ICC(2,1)", results[1]["icc"]), ("ICC(3,1)", results[2]["icc"]), ("ICC(2,k)", results[4]["icc"])])
    ci = f"IC {100 * (1 - p['alpha']):g} %"
    out.table(
        "Las seis formas del ICC",
        ["Forma", "Modelo", "ICC", "F", "gl1", "gl2", "Valor p", f"{ci} inf.", f"{ci} sup."],
        [
            [
                r["label"],
                dict(_ICC_FORMS)[r["label"]],
                r["icc"],
                r["f"],
                r["df1"],
                r["df2"],
                P(r["p"]),
                r["low"],
                r["high"],
            ]
            for r in results
        ],
        note="Elija según el diseño: ICC(2,·) si los evaluadores son una muestra de una población de evaluadores, ICC(3,·) si son los únicos de interés. Menos de 0.5 pobre, 0.5–0.75 moderada, 0.75–0.9 buena, más de 0.9 excelente (Koo y Li, 2016).",
    )
    return out.set(icc={r["label"]: r["icc"] for r in results}).build()


def _icc_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import pingouin as pg  # pip install pingouin\n\n"
        + load_line(dataset)
        + f"long = df[{roles['raters']!r}].dropna().rename_axis('sujeto').reset_index().melt(id_vars='sujeto', var_name='evaluador', value_name='nota')\nprint(pg.intraclass_corr(data=long, targets='sujeto', raters='evaluador', ratings='nota'))\n"
    )


# ============================================================================ equivalence (TOST)
def _tost(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    design, low, high, alpha = p["design"], float(p["low"]), float(p["upp"]), p["alpha"]
    if design == "two" and not roles.get("group"):
        raise InvalidInputError("Elija la columna de grupos (2 categorías).")
    if design == "paired" and not roles.get("var2"):
        raise InvalidInputError("Elija la segunda columna de los datos pareados.")
    if low >= high:
        raise InvalidInputError("El límite inferior de equivalencia debe ser menor que el superior.")
    if design == "two":
        (la, a), (lb, b) = two_groups(frame, roles["var"], roles["group"])
        diff = float(a.mean() - b.mean())
        n1, n2 = len(a), len(b)
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        se = float(np.sqrt(v1 / n1 + v2 / n2))
        df = float((v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
        what = f"{la} − {lb}"
    else:
        if design == "paired":
            pair = pd.DataFrame({"a": numbers(frame, roles["var"]), "b": numbers(frame, roles["var2"])}).dropna()
            values, what = (pair["a"] - pair["b"]).to_numpy(), f"{roles['var']} − {roles['var2']}"
        else:
            values, what = (
                numbers(frame, roles["var"]).dropna().to_numpy() - float(p["target"]),
                f"{roles['var']} − {p['target']:g}",
            )
        need(pd.Series(values), roles["var"], 3)
        diff, se, df = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))), float(len(values) - 1)
    if se == 0:
        raise InvalidInputError("No hay variabilidad en los datos: no se puede probar la equivalencia.")
    p_low = float(stats.t.sf((diff - low) / se, df))
    p_high = float(stats.t.cdf((diff - high) / se, df))
    p_tost = max(p_low, p_high)
    margin = stats.t.ppf(1 - alpha, df) * se
    ci_low, ci_high = diff - margin, diff + margin
    equivalent = p_tost < alpha
    out = Output("Prueba de equivalencia (TOST)", f"{what} · límites [{low:g}, {high:g}]")
    out.tiles(
        [
            ("Diferencia", diff),
            ("Valor p de TOST", P(p_tost)),
            ("Decisión", "Equivalentes" if equivalent else "No se demuestra equivalencia"),
        ]
    )
    out.alert(
        "success" if equivalent else "warning",
        f"El IC del {100 * (1 - 2 * alpha):g} % [{ci_low:.4g}, {ci_high:.4g}] "
        + ("queda dentro" if equivalent else "no queda dentro")
        + f" de los límites [{low:g}, {high:g}]: "
        + ("se concluye equivalencia." if equivalent else "no se puede concluir equivalencia."),
    )
    out.table(
        "Dos pruebas unilaterales",
        ["Hipótesis nula", "Estadístico t", "gl", "Valor p"],
        [
            [f"diferencia ≤ {low:g}", (diff - low) / se, df, P(p_low)],
            [f"diferencia ≥ {high:g}", (diff - high) / se, df, P(p_high)],
        ],
        note="La equivalencia se declara si las dos hipótesis nulas se rechazan (el mayor de los dos valores p es menor que α).",
    )
    out.table(
        "Diferencia e intervalo",
        ["Diferencia", "Error estándar", f"IC {100 * (1 - 2 * alpha):g} % inf.", "IC sup."],
        [[diff, se, float(ci_low), float(ci_high)]],
    )
    out.chart(
        scatter_chart(
            "Diferencia e intervalo frente a los límites de equivalencia",
            what,
            "",
            [{"name": "Diferencia estimada", "x": np.array([diff]), "y": np.array([0.0]), "size": 11}],
            lines=[
                {
                    "name": f"IC {100 * (1 - 2 * alpha):g} %",
                    "x": np.array([ci_low, ci_high]),
                    "y": np.array([0.0, 0.0]),
                    "color": "#4f46e5",
                }
            ],
            vlines=[{"value": low, "label": "límite inf."}, {"value": high, "label": "límite sup."}],
            height=260,
        )
    )
    return out.set(difference=diff, p=p_tost, equivalent=bool(equivalent), ci=[float(ci_low), float(ci_high)]).build()


def _tost_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if p["design"] == "two":
        return (
            "from statsmodels.stats.weightstats import ttost_ind\n\n"
            + load_line(dataset)
            + f"a, b = (g[{py_name(roles['var'])}].dropna() for _, g in df.groupby({py_name(roles['group'])}))\nprint(ttost_ind(a, b, {p['low']}, {p['upp']}, usevar='unequal'))\n"
        )
    if p["design"] == "paired":
        return (
            "from statsmodels.stats.weightstats import ttost_paired\n\n"
            + load_line(dataset)
            + f"d = df[[{py_name(roles['var'])}, {py_name(roles['var2'])}]].dropna()\nprint(ttost_paired(d.iloc[:, 0], d.iloc[:, 1], {p['low']}, {p['upp']}))\n"
        )
    return (
        "from statsmodels.stats.weightstats import ttost_paired\nimport numpy as np\n\n"
        + load_line(dataset)
        + f"x = df[{py_name(roles['var'])}].dropna()\nprint(ttost_paired(x, np.full(len(x), {p['target']}), {p['low']}, {p['upp']}))\n"
    )


# ============================================================================ multivariate means and partial correlation
def _hotelling(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, alpha = roles["vars"], p["alpha"]
    pcount = len(names)
    if p["design"] == "one":
        data = frame[names].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        n = len(data)
        try:
            mu0 = (
                np.array([float(v) for v in str(p["mu0"] or "").replace(";", ",").split(",") if v.strip()])
                if p["mu0"]
                else np.zeros(pcount)
            )
        except ValueError:
            raise InvalidInputError("El vector de referencia debe ser números separados por comas.") from None
        if len(mu0) != pcount:
            raise InvalidInputError(f"El vector de referencia necesita {pcount} valores (uno por variable).")
        if n <= pcount:
            raise InvalidInputError(f"Se necesitan más de {pcount} filas completas.")
        cov = np.cov(data, rowvar=False)
        diff = data.mean(axis=0) - mu0
        t2, df1, df2 = n * float(diff @ np.linalg.solve(cov, diff)), pcount, n - pcount
        f = t2 * (n - pcount) / (pcount * (n - 1))
        title, detail = "T² de Hotelling (una muestra)", f"{n} filas · vector de referencia {mu0.tolist()}"
        means = [[c, float(m), float(r)] for c, m, r in zip(names, data.mean(axis=0), mu0, strict=True)]
        head = ["Variable", "Media", "Referencia"]
    else:
        data = frame[[*names, roles["group"]]].dropna()
        levels = level_order(data[roles["group"]])
        if len(levels) != 2:
            raise InvalidInputError(f"«{roles['group']}» debe tener exactamente 2 categorías (tiene {len(levels)}).")
        a = data[data[roles["group"]] == levels[0]][names].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        b = data[data[roles["group"]] == levels[1]][names].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        n1, n2 = len(a), len(b)
        if n1 + n2 - 2 <= pcount:
            raise InvalidInputError(f"Se necesitan más de {pcount + 2} filas en total.")
        pooled = ((n1 - 1) * np.cov(a, rowvar=False) + (n2 - 1) * np.cov(b, rowvar=False)) / (n1 + n2 - 2)
        diff = a.mean(axis=0) - b.mean(axis=0)
        t2 = (n1 * n2 / (n1 + n2)) * float(diff @ np.linalg.solve(pooled, diff))
        df1, df2 = pcount, n1 + n2 - pcount - 1
        f = t2 * df2 / (pcount * (n1 + n2 - 2))
        title, detail = "T² de Hotelling (dos muestras)", f"{levels[0]} (n = {n1}) frente a {levels[1]} (n = {n2})"
        means = [[c, float(m1), float(m2)] for c, m1, m2 in zip(names, a.mean(axis=0), b.mean(axis=0), strict=True)]
        head = ["Variable", f"Media {levels[0]}", f"Media {levels[1]}"]
    pvalue = float(stats.f.sf(f, df1, df2))
    out = Output(title, detail)
    out.tiles([("T²", float(t2)), ("F", float(f)), ("gl", f"{df1}, {df2}"), ("Valor p", P(pvalue))])
    decision(out, pvalue, alpha, "los vectores de medias son iguales")
    out.table("Medias", head, means)
    return out.set(t2=float(t2), f=float(f), p=pvalue).build()


def _hotelling_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if p["design"] == "two":
        return (
            "from statsmodels.multivariate.manova import MANOVA  # T² de dos muestras = lambda de Wilks con dos grupos\n\n"
            + load_line(dataset)
            + f"fit = MANOVA.from_formula('{' + '.join(roles['vars'])} ~ C({roles['group']})', data=df)\nprint(fit.mv_test())\n"
        )
    return (
        "import numpy as np\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"x = df[{roles['vars']!r}].dropna().to_numpy()\nn, k = x.shape\nd = x.mean(axis=0)  # menos el vector de referencia\nt2 = n * d @ np.linalg.solve(np.cov(x, rowvar=False), d)\nprint(t2, stats.f.sf(t2 * (n - k) / (k * (n - 1)), k, n - k))\n"
    )


def _partial(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    x_name, y_name, covars, method, alpha = roles["x"], roles["y"], roles["covars"], p["method"], p["alpha"]
    names = [x_name, y_name, *covars]
    data = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    k, n = len(covars), len(data)
    if n <= k + 3:
        raise InvalidInputError(f"Se necesitan más de {k + 3} filas completas.")
    work = data.rank() if method == "spearman" else data

    def residual(column: str) -> np.ndarray:
        design = np.column_stack([np.ones(n), work[covars].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, work[column].to_numpy(), rcond=None)
        return work[column].to_numpy() - design @ coefficients

    rx, ry = residual(x_name), residual(y_name)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        raise InvalidInputError("Una de las variables queda sin variación al quitar el efecto de las covariables.")
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / (1 - r**2))
    pvalue = float(2 * stats.t.sf(abs(t), df))
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    low, high = (
        float(np.tanh(z - stats.norm.ppf(1 - alpha / 2) * se)),
        float(np.tanh(z + stats.norm.ppf(1 - alpha / 2) * se)),
    )
    simple = float(np.corrcoef(work[x_name], work[y_name])[0, 1])
    out = Output(
        f"Correlación parcial de {x_name} y {y_name}",
        f"controlando por {', '.join(covars)} · {'Spearman' if method == 'spearman' else 'Pearson'}",
    )
    out.tiles([("Correlación parcial", r), ("Correlación simple", simple), ("Valor p", P(pvalue)), ("n", n)])
    decision(out, pvalue, alpha, "la correlación parcial es 0")
    out.table(
        "Resultado",
        ["Correlación parcial", "gl", "t", "Valor p", f"IC {100 * (1 - alpha):g} % inf.", "IC sup."],
        [[r, df, float(t), P(pvalue), low, high]],
        note="El intervalo usa la transformación z de Fisher con n − 3 − (covariables) grados de libertad.",
    )
    if abs(simple) > abs(r) + 0.1:
        out.alert(
            "info", "La correlación baja al controlar por las covariables: parte de la relación simple se debe a ellas."
        )
    return out.set(partial=r, simple=simple, p=pvalue, n=n).build()


def _partial_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nimport pandas as pd\nimport statsmodels.api as sm\n\n"
        + load_line(dataset)
        + f"d = df[{[roles['x'], roles['y'], *roles['covars']]!r}].dropna()\n"
        + ("d = d.rank()\n" if p["method"] == "spearman" else "")
        + f"Z = sm.add_constant(d[{roles['covars']!r}])\nrx = sm.OLS(d[{py_name(roles['x'])}], Z).fit().resid\nry = sm.OLS(d[{py_name(roles['y'])}], Z).fit().resid\nprint(np.corrcoef(rx, ry)[0, 1])\n"
    )


# ============================================================================ confidence intervals of one variable
def _ci_one(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    name, level = roles["var"], p["confidence"] / 100
    alpha = 1 - level
    x = numbers(frame, name).dropna().to_numpy()
    need(pd.Series(x), name, 5)
    n, mean, sd = len(x), float(x.mean()), float(x.std(ddof=1))
    half = float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n))
    sd_low, sd_high = (
        sd * np.sqrt((n - 1) / stats.chi2.ppf(1 - alpha / 2, n - 1)),
        sd * np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)),
    )
    ordered, k = np.sort(x), 0
    while k + 1 <= n // 2 and 2 * stats.binom.cdf(k, n, 0.5) <= alpha:
        k += 1
    out = Output(f"Intervalos de confianza de {name}", f"nivel {p['confidence']:g} % · n = {n}")
    out.tiles([("Media", mean), ("Mediana", float(np.median(x))), ("Desv. estándar", sd)])
    rows = [
        ["Media (t de Student)", mean, mean - half, mean + half],
        ["Desviación estándar (χ²)", sd, float(sd_low), float(sd_high)],
        ["Varianza (χ²)", sd**2, float(sd_low**2), float(sd_high**2)],
    ]
    if k >= 1:
        rows.append(
            [
                f"Mediana (signos, confianza {100 * (1 - 2 * stats.binom.cdf(k - 1, n, 0.5)):.1f} %)",
                float(np.median(x)),
                float(ordered[k - 1]),
                float(ordered[n - k]),
            ]
        )
    out.table(
        "Intervalos de confianza",
        ["Parámetro", "Estimación", "Límite inf.", "Límite sup."],
        rows,
        note="La media y las desviaciones suponen datos aproximadamente normales; el de la mediana no.",
    )
    return out.set(mean=mean, sd=sd, mean_ci=[mean - half, mean + half]).build()


def _ci_one_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from scipy import stats\n\n"
        + load_line(dataset)
        + f"x = df[{py_name(roles['var'])}].dropna()\nprint(stats.t.interval({p['confidence'] / 100}, len(x) - 1, loc=x.mean(), scale=stats.sem(x)))\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


VAR = RoleSpec("var", "Variable", "numeric")
GROUP = RoleSpec("group", "Grupos", "categorical")
NORMALITY = "Normalidad"
VARIANCES = "Varianzas"
NONPARAM = "No paramétricas y datos pareados"
TABLES = "Tasas, proporciones y tablas"
AGREEMENT = "Acuerdo entre evaluadores"
EQUIV = "Equivalencia y multivariadas"

SPECS = (
    _spec(
        "anderson",
        "Anderson-Darling (normalidad y otras)",
        NORMALITY,
        "Prueba de bondad de ajuste a la normal, exponencial, logística o Gumbel con un gráfico de probabilidad.",
        _anderson,
        roles=(VAR,),
        params=(
            FieldSpec("dist", "Distribución", "select", "norm", tuple((k, v[1]) for k, v in _ANDERSON.items())),
            ALPHA,
        ),
        code=_anderson_code,
        example={"roles": {"var": "y"}, "params": {}},
    ),
    _spec(
        "lilliefors",
        "Lilliefors (Kolmogorov-Smirnov corregida)",
        NORMALITY,
        "Normalidad o exponencialidad con parámetros estimados de los propios datos.",
        _lilliefors,
        roles=(VAR,),
        params=(
            FieldSpec("dist", "Distribución", "select", "norm", (("norm", "Normal"), ("exp", "Exponencial"))),
            ALPHA,
        ),
        code=_lilliefors_code,
        example={"roles": {"var": "y"}, "params": {}},
    ),
    _spec(
        "ks_one",
        "Kolmogorov-Smirnov de una muestra",
        NORMALITY,
        "Compara los datos con una distribución cuyos parámetros usted fija de antemano.",
        _ks_one,
        roles=(VAR,),
        params=(
            FieldSpec("dist", "Distribución", "select", "norm", _KS_DISTS),
            FieldSpec("loc", "Ubicación", "number", 0.0),
            FieldSpec("scale", "Escala", "number", 1.0, minimum=1e-9),
            ALTERNATIVE,
            ALPHA,
        ),
        code=_ks_one_code,
        example={"roles": {"var": "y"}, "params": {"dist": "norm", "loc": 8, "scale": 4.5}},
    ),
    _spec(
        "jarque_bera",
        "Jarque-Bera",
        NORMALITY,
        "Normalidad a partir de la asimetría y la curtosis (muestras grandes).",
        _jarque_bera,
        roles=(VAR,),
        params=(ALPHA,),
        code=_jarque_bera_code,
        example={"roles": {"var": "y"}, "params": {}},
    ),
    _spec(
        "ks_two",
        "Kolmogorov-Smirnov de dos muestras",
        NORMALITY,
        "¿Provienen dos muestras de la misma distribución? Compara sus funciones de distribución empíricas.",
        _ks_two,
        roles=(VAR, RoleSpec("group", "Grupos (2 categorías)", "categorical")),
        params=(ALTERNATIVE, ALPHA),
        code=_ks_two_code,
        example={"roles": {"var": "y", "group": "h"}, "params": {}},
    ),
    _spec(
        "variances",
        "Igualdad de varianzas (Levene, Bartlett, Fligner)",
        VARIANCES,
        "Cuatro pruebas de homogeneidad de varianzas para 2 o más grupos, con intervalos de Bonferroni para cada desviación.",
        _variances,
        roles=(VAR, GROUP),
        params=(ALPHA,),
        code=_variances_code,
        example={"roles": {"var": "y", "group": "g"}, "params": {}},
    ),
    _spec(
        "sign_test",
        "Prueba de los signos (mediana)",
        NONPARAM,
        "Contrasta la mediana con un valor, o la mediana de las diferencias de dos columnas pareadas; con IC para la mediana.",
        _sign_test,
        roles=(VAR, RoleSpec("var2", "Segunda columna (datos pareados, opcional)", "numeric", optional=True)),
        params=(FieldSpec("median0", "Mediana de referencia (una columna)", "number", 0.0), ALTERNATIVE, ALPHA),
        code=_sign_code,
        example={"roles": {"var": "x1"}, "params": {"median0": 10}},
    ),
    _spec(
        "mood_median",
        "Mediana de Mood (k grupos)",
        NONPARAM,
        "Compara las medianas de varios grupos con una tabla de datos por encima y por debajo de la mediana global.",
        _mood_median,
        roles=(VAR, GROUP),
        params=(ALPHA,),
        code=_mood_code,
        example={"roles": {"var": "y", "group": "g"}, "params": {}},
    ),
    _spec(
        "friedman",
        "Friedman (bloques)",
        NONPARAM,
        "Compara 3 o más tratamientos medidos en los mismos sujetos, con rangos; alternativa al ANOVA de medidas repetidas.",
        _friedman,
        roles=(RoleSpec("vars", "Tratamientos (columnas)", "numeric", multiple=True, min_count=3),),
        params=(ALPHA,),
        code=_friedman_code,
        example={"roles": {"vars": ["r1", "r2", "r3"]}, "params": {}},
    ),
    _spec(
        "runs",
        "Prueba de rachas (aleatoriedad)",
        NONPARAM,
        "¿Es aleatoria la secuencia de datos? Cuenta las rachas por encima y por debajo de un valor de corte.",
        _runs,
        roles=(VAR,),
        params=(
            FieldSpec(
                "cutoff",
                "Corte",
                "select",
                "median",
                (("median", "Mediana"), ("mean", "Media"), ("value", "Un valor")),
            ),
            FieldSpec("value", "Valor de corte", "number", visible_if=("cutoff", ("value",))),
            ALPHA,
        ),
        code=_runs_code,
        example={"roles": {"var": "s"}, "params": {}},
    ),
    _spec(
        "mcnemar",
        "McNemar (dos mediciones binarias)",
        NONPARAM,
        "Compara dos mediciones binarias del mismo sujeto (antes/después, dos pruebas diagnósticas).",
        _mcnemar,
        roles=(RoleSpec("a", "Medición A (2 valores)", "any"), RoleSpec("b", "Medición B (2 valores)", "any")),
        params=(
            FieldSpec(
                "method",
                "Método",
                "select",
                "auto",
                (
                    ("auto", "Automático (exacto con pocos discordantes)"),
                    ("exact", "Exacto (binomial)"),
                    ("chi2", "χ² con corrección de continuidad"),
                ),
            ),
            ALPHA,
        ),
        code=_mcnemar_code,
        example={"roles": {"a": "h", "b": "b"}, "params": {}},
    ),
    _spec(
        "cochran_q",
        "Q de Cochran (k mediciones binarias)",
        NONPARAM,
        "Generaliza McNemar a 3 o más mediciones binarias (0/1) de los mismos sujetos.",
        _cochran,
        roles=(RoleSpec("vars", "Mediciones 0/1", "numeric", multiple=True, min_count=3),),
        params=(ALPHA,),
        code=_cochran_code,
        example={"roles": {"vars": ["c1", "c2", "c3"]}, "params": {}},
    ),
    _spec(
        "poisson_rate",
        "Tasas de Poisson (1 y 2 muestras)",
        TABLES,
        "Contrasta una tasa de eventos con un valor, o compara dos tasas, con intervalos exactos. Se escriben los conteos: no necesita dataset.",
        _poisson,
        needs_dataset=False,
        params=(
            FieldSpec("design", "Diseño", "select", "one", (("one", "Una tasa"), ("two", "Dos tasas"))),
            FieldSpec("count1", "Eventos observados (muestra 1)", "integer", 10, minimum=0),
            FieldSpec(
                "exposure1", "Exposición (muestra 1): horas, personas-año, unidades…", "number", 10, minimum=1e-9
            ),
            FieldSpec("rate0", "Tasa de referencia", "number", 1.0, minimum=1e-9, visible_if=("design", ("one",))),
            FieldSpec(
                "method1",
                "Método",
                "select",
                "exact-c",
                (("exact-c", "Exacto"), ("score", "Score"), ("wald", "Wald")),
                visible_if=("design", ("one",)),
            ),
            FieldSpec(
                "count2", "Eventos observados (muestra 2)", "integer", 15, minimum=0, visible_if=("design", ("two",))
            ),
            FieldSpec(
                "exposure2", "Exposición (muestra 2)", "number", 12, minimum=1e-9, visible_if=("design", ("two",))
            ),
            FieldSpec("ratio0", "Razón de tasas bajo H₀", "number", 1.0, minimum=1e-9, visible_if=("design", ("two",))),
            FieldSpec(
                "method2",
                "Prueba",
                "select",
                "exact-cond",
                (
                    ("exact-cond", "Exacta condicional"),
                    ("cond-midp", "Exacta con valor p medio"),
                    ("score", "Score"),
                    ("wald-log", "Wald (logarítmica)"),
                ),
                visible_if=("design", ("two",)),
            ),
            ALTERNATIVE,
            ALPHA,
        ),
        code=_poisson_code,
        example={"roles": {}, "params": {"design": "two"}},
    ),
    _spec(
        "odds_ratio",
        "Odds ratio y riesgo relativo (2×2)",
        TABLES,
        "Mide la asociación entre una exposición y un evento binarios, con intervalos y pruebas exacta de Fisher y χ².",
        _odds_ratio,
        roles=(
            RoleSpec("exposure", "Exposición (2 valores; el mayor es «expuesto»)", "any"),
            RoleSpec("outcome", "Evento (2 valores; el mayor es «evento»)", "any"),
        ),
        params=(ALPHA,),
        code=_odds_code,
        example={"roles": {"exposure": "h", "outcome": "b"}, "params": {}},
    ),
    _spec(
        "cmh",
        "Cochran-Mantel-Haenszel (2×2 estratificadas)",
        TABLES,
        "Odds ratio común y prueba de asociación controlando por una variable de estrato, con prueba de homogeneidad de Breslow-Day.",
        _cmh,
        roles=(
            RoleSpec("exposure", "Exposición (2 valores)", "any"),
            RoleSpec("outcome", "Evento (2 valores)", "any"),
            RoleSpec("stratum", "Estrato", "categorical"),
        ),
        params=(ALPHA,),
        code=_cmh_code,
        example={"roles": {"exposure": "h", "outcome": "b", "stratum": "g"}, "params": {}},
    ),
    _spec(
        "kappa",
        "Kappa de Cohen y de Fleiss (acuerdo)",
        AGREEMENT,
        "Acuerdo entre 2 evaluadores (Cohen, simple o ponderado) o más (Fleiss), corregido por el azar.",
        _kappa,
        roles=(RoleSpec("raters", "Evaluadores (una columna cada uno)", "any", multiple=True, min_count=2),),
        params=(
            FieldSpec(
                "weights",
                "Ponderación (2 evaluadores, categorías ordenadas)",
                "select",
                "none",
                (("none", "Sin ponderar"), ("linear", "Lineal"), ("quadratic", "Cuadrática")),
            ),
        ),
        code=_kappa_code,
        example={"roles": {"raters": ["r1", "r2"]}, "params": {"weights": "quadratic"}},
    ),
    _spec(
        "icc",
        "Correlación intraclase (CCI)",
        AGREEMENT,
        "Fiabilidad de mediciones continuas de varios evaluadores: las seis formas de Shrout y Fleiss con IC y pruebas F.",
        _icc,
        roles=(
            RoleSpec(
                "raters",
                "Evaluadores (una columna cada uno; una fila por sujeto)",
                "numeric",
                multiple=True,
                min_count=2,
            ),
        ),
        params=(ALPHA,),
        code=_icc_code,
        example={"roles": {"raters": ["r1", "r2", "r3"]}, "params": {}},
    ),
    _spec(
        "tost",
        "Equivalencia de medias (TOST)",
        EQUIV,
        "Demuestra que una diferencia es tan pequeña que no importa: dos pruebas unilaterales contra límites de equivalencia.",
        _tost,
        roles=(
            VAR,
            RoleSpec("var2", "Segunda columna (datos pareados)", "numeric", optional=True),
            RoleSpec("group", "Grupos (2 muestras independientes)", "categorical", optional=True),
        ),
        params=(
            FieldSpec(
                "design",
                "Diseño",
                "select",
                "two",
                (
                    ("one", "Una muestra frente a un valor"),
                    ("two", "Dos muestras independientes"),
                    ("paired", "Datos pareados"),
                ),
            ),
            FieldSpec("target", "Valor de referencia", "number", 0.0, visible_if=("design", ("one",))),
            FieldSpec("low", "Límite inferior de equivalencia (diferencia)", "number", -1.0),
            FieldSpec("upp", "Límite superior de equivalencia (diferencia)", "number", 1.0),
            ALPHA,
        ),
        code=_tost_code,
        example={"roles": {"var": "y", "group": "h"}, "params": {"design": "two", "low": -2, "upp": 2}},
    ),
    _spec(
        "hotelling",
        "T² de Hotelling (vectores de medias)",
        EQUIV,
        "Compara varias medias a la vez: un vector con una referencia, o dos grupos entre sí.",
        _hotelling,
        roles=(
            RoleSpec("vars", "Variables (2 o más)", "numeric", multiple=True, min_count=2),
            RoleSpec("group", "Grupos (2 muestras)", "categorical", optional=True),
        ),
        params=(
            FieldSpec(
                "design",
                "Diseño",
                "select",
                "two",
                (("one", "Una muestra frente a un vector"), ("two", "Dos muestras independientes")),
            ),
            FieldSpec(
                "mu0",
                "Vector de referencia (separado por comas; vacío = ceros)",
                "text",
                optional=True,
                visible_if=("design", ("one",)),
            ),
            ALPHA,
        ),
        code=_hotelling_code,
        example={"roles": {"vars": ["x1", "x2", "x3"], "group": "h"}, "params": {}},
    ),
    _spec(
        "partial_corr",
        "Correlación parcial",
        EQUIV,
        "Correlación entre dos variables quitando el efecto de una o más covariables (Pearson o Spearman).",
        _partial,
        roles=(
            RoleSpec("x", "Variable X", "numeric"),
            RoleSpec("y", "Variable Y", "numeric"),
            RoleSpec("covars", "Covariables a controlar", "numeric", multiple=True),
        ),
        params=(
            FieldSpec("method", "Coeficiente", "select", "pearson", (("pearson", "Pearson"), ("spearman", "Spearman"))),
            ALPHA,
        ),
        code=_partial_code,
        example={"roles": {"x": "x1", "y": "y", "covars": ["x2"]}, "params": {}},
    ),
    _spec(
        "ci_one",
        "Intervalos de confianza de una variable",
        "Estimación",
        "Media (t), desviación estándar y varianza (χ²) y mediana (signos) con el nivel de confianza que elija.",
        _ci_one,
        roles=(VAR,),
        params=(FieldSpec("confidence", "Nivel de confianza (%)", "number", 95, minimum=50, maximum=99.9),),
        code=_ci_one_code,
        example={"roles": {"var": "y"}, "params": {}},
    ),
)

registry.register(*SPECS)
