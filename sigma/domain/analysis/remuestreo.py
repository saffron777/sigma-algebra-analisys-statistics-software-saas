"""Resampling and simulation (Minitab *Bootstrapping*/*Random Data*, SPSS *Bootstrap*/*Monte Carlo*)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import hist_chart
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.plots import groups_of, need, numbers
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import RESULT_KEY, load_line, py_name
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "remuestreo"
MAX_CELLS = 60_000_000  # resamples × sample size a single analysis may draw
SEED = FieldSpec("seed", "Semilla (para repetir el resultado)", "integer", 12345, minimum=0, maximum=2**31 - 1)


def _draws(resamples: int, size: int) -> None:
    if resamples * max(size, 1) > MAX_CELLS:
        raise InvalidInputError(
            f"{resamples:,} réplicas de {size:,} datos son demasiados cálculos; reduzca las réplicas.".replace(",", " ")
        )


# ============================================================================ statistics (vectorised over the last axis)
def _corr(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    xm, ym = x - x.mean(axis=axis, keepdims=True), y - y.mean(axis=axis, keepdims=True)
    return (xm * ym).sum(axis=axis) / np.sqrt((xm**2).sum(axis=axis) * (ym**2).sum(axis=axis))


_ONE = {
    "mean": ("Media", lambda x, axis=-1: np.mean(x, axis=axis)),
    "median": ("Mediana", lambda x, axis=-1: np.median(x, axis=axis)),
    "std": ("Desviación estándar", lambda x, axis=-1: np.std(x, axis=axis, ddof=1)),
    "trimmed": ("Media recortada 10 %", lambda x, axis=-1: stats.trim_mean(x, 0.1, axis=axis)),
    "iqr": ("Rango intercuartílico", lambda x, axis=-1: stats.iqr(x, axis=axis)),
}
_STATISTICS = (
    *((key, label) for key, (label, _) in _ONE.items()),
    ("percentile", "Percentil (indique cuál)"),
    ("correlation", "Correlación de Pearson (dos columnas)"),
    ("mean_diff", "Diferencia de medias (dos grupos)"),
    ("median_diff", "Diferencia de medianas (dos grupos)"),
)
_METHODS = (("BCa", "BCa (corregido por sesgo y asimetría)"), ("percentile", "Percentil"), ("basic", "Básico"))


# ============================================================================ bootstrap
def _bootstrap(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    kind, resamples, confidence = p["statistic"], int(p["resamples"]), p["confidence"] / 100
    var = roles["var"]
    rng = np.random.default_rng(int(p["seed"]))
    if kind == "correlation":
        if not roles.get("var2"):
            raise InvalidInputError("Elija la segunda columna para la correlación.")
        pair = pd.DataFrame({"x": numbers(frame, var), "y": numbers(frame, roles["var2"])}).dropna()
        need(pair["x"], "los pares", 5)
        data, statistic, paired = (pair["x"].to_numpy(), pair["y"].to_numpy()), _corr, True
        point, label, size = float(_corr(*data)), f"correlación de {var} y {roles['var2']}", len(pair)
    elif kind in ("mean_diff", "median_diff"):
        if not roles.get("group"):
            raise InvalidInputError("Elija la columna de grupos (2 categorías).")
        parts = groups_of(frame, var, roles["group"])
        if len(parts) != 2:
            raise InvalidInputError(
                f"«{roles['group']}» debe tener exactamente 2 categorías con datos (tiene {len(parts)})."
            )
        (la, a), (lb, b) = parts
        for label_, values in ((la, a), (lb, b)):
            need(values, label_, 3)
        reducer = np.mean if kind == "mean_diff" else np.median
        data = (a.to_numpy(), b.to_numpy())

        def statistic(x, y, axis=-1):
            return reducer(x, axis=axis) - reducer(y, axis=axis)

        paired, size = False, len(a) + len(b)
        point = float(statistic(*data))
        label = f"{'media' if kind == 'mean_diff' else 'mediana'} de {la} − {lb}"
    else:
        x = numbers(frame, var).dropna().to_numpy()
        need(pd.Series(x), var, 5)
        if kind == "percentile":
            q = float(p["percentile"])

            def statistic(v, axis=-1):
                return np.percentile(v, q, axis=axis)

            label = f"percentil {q:g} de {var}"
        else:
            label, statistic = _ONE[kind][0] + f" de {var}", _ONE[kind][1]
        data, paired, size, point = (x,), False, len(x), float(statistic(x))
    _draws(resamples, size)
    try:
        result = stats.bootstrap(
            data,
            statistic,
            n_resamples=resamples,
            confidence_level=confidence,
            method=p["method"],
            vectorized=True,
            paired=paired,
            random_state=rng,
        )
    except (ValueError, FloatingPointError) as error:
        raise ComputationError(f"No se pudo calcular el intervalo: {error}") from None
    distribution = np.asarray(result.bootstrap_distribution, dtype=float)
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    out = Output(
        f"Bootstrap: {label}",
        f"{resamples:,} réplicas · IC {p['confidence']:g} % ({p['method']}) · semilla {int(p['seed'])}".replace(
            ",", " "
        ),
    )
    out.tiles(
        [
            ("Estimación", point),
            ("Error estándar bootstrap", float(result.standard_error)),
            ("Límite inferior", low),
            ("Límite superior", high),
        ]
    )
    sesgo = float(distribution.mean() - point)
    rows = [[label, point, float(result.standard_error), sesgo, low, high]]
    out.table(
        "Intervalo de confianza bootstrap",
        ["Estadístico", "Estimación", "Error estándar", "Sesgo", "Límite inf.", "Límite sup."],
        rows,
        note="Sesgo = media de las réplicas − estimación original.",
    )
    if kind == "mean" and len(data[0]) > 2:
        x = data[0]
        half = stats.t.ppf(0.5 + confidence / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        out.table(
            "Comparación con el método paramétrico",
            ["Método", "Límite inf.", "Límite sup."],
            [["Bootstrap", low, high], ["t de Student", float(x.mean() - half), float(x.mean() + half)]],
        )
    out.chart(
        hist_chart(
            "Distribución bootstrap",
            label,
            distribution,
            norm="density",
            vlines=[
                {"value": point, "label": "estimación", "dash": "solid"},
                {"value": low, "label": "IC"},
                {"value": high},
            ],
        )
    )
    return out.set(estimate=point, se=float(result.standard_error), low=low, high=high, bias=sesgo).build()


def _bootstrap_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    var = py_name(roles["var"])
    if p["statistic"] in ("correlation", "mean_diff", "median_diff"):
        return "# ver scipy.stats.bootstrap con varias muestras (paired=True para la correlación)\n" + load_line(
            dataset
        )
    func = {
        "mean": "np.mean",
        "median": "np.median",
        "std": "lambda v, axis: np.std(v, axis=axis, ddof=1)",
        "trimmed": "lambda v, axis: stats.trim_mean(v, 0.1, axis=axis)",
        "iqr": "stats.iqr",
        "percentile": f"lambda v, axis: np.percentile(v, {p['percentile']}, axis=axis)",
    }[p["statistic"]]
    return (
        "import numpy as np\nfrom scipy import stats\n\n"
        + load_line(dataset)
        + f"x = df[{var}].dropna().to_numpy()\n"
        + f"res = stats.bootstrap((x,), {func}, n_resamples={int(p['resamples'])}, confidence_level={p['confidence'] / 100}, method={p['method']!r}, vectorized=True, random_state={int(p['seed'])})\n"
        + "print(res.confidence_interval, res.standard_error)\n"
    )


# ============================================================================ permutation tests
_PERM = (
    ("mean_diff", "Diferencia de medias (dos grupos)"),
    ("median_diff", "Diferencia de medianas (dos grupos)"),
    ("paired", "Diferencia media de datos pareados (dos columnas)"),
    ("correlation", "Correlación de Pearson (dos columnas)"),
)


def _permutation(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    kind, resamples, alternative = p["design"], int(p["resamples"]), p["alternative"]
    var = roles["var"]
    rng = np.random.default_rng(int(p["seed"]))
    if kind in ("paired", "correlation"):
        if not roles.get("var2"):
            raise InvalidInputError("Elija la segunda columna.")
        pair = pd.DataFrame({"x": numbers(frame, var), "y": numbers(frame, roles["var2"])}).dropna()
        need(pair["x"], "los pares", 5)
        data = (pair["x"].to_numpy(), pair["y"].to_numpy())
        if kind == "paired":

            def statistic(x, y, axis=-1):
                return np.mean(x - y, axis=axis)

            kind_perm, label = "samples", f"media de {var} − {roles['var2']}"
        else:
            statistic, kind_perm, label = _corr, "pairings", f"correlación de {var} y {roles['var2']}"
        size = len(pair)
    else:
        if not roles.get("group"):
            raise InvalidInputError("Elija la columna de grupos (2 categorías).")
        parts = groups_of(frame, var, roles["group"])
        if len(parts) != 2:
            raise InvalidInputError(
                f"«{roles['group']}» debe tener exactamente 2 categorías con datos (tiene {len(parts)})."
            )
        (la, a), (lb, b) = parts
        reducer = np.mean if kind == "mean_diff" else np.median

        def statistic(x, y, axis=-1):
            return reducer(x, axis=axis) - reducer(y, axis=axis)

        data, kind_perm, size = (a.to_numpy(), b.to_numpy()), "independent", len(a) + len(b)
        label = f"{'media' if kind == 'mean_diff' else 'mediana'} de {la} − {lb}"
        for name, values in ((la, a), (lb, b)):
            need(values, name, 2)
    _draws(resamples, size)
    result = stats.permutation_test(
        data,
        statistic,
        permutation_type=kind_perm,
        vectorized=True,
        n_resamples=resamples,
        alternative=alternative,
        random_state=rng,
    )
    observed, pvalue = float(result.statistic), float(result.pvalue)
    null = np.asarray(result.null_distribution, dtype=float)
    out = Output(
        f"Prueba de permutación: {label}", f"{len(null):,} permutaciones · semilla {int(p['seed'])}".replace(",", " ")
    )
    out.tiles([("Estadístico observado", observed), ("Valor p", P(pvalue)), ("Permutaciones", len(null))])
    reject = pvalue < p["alpha"]
    out.alert(
        "warning" if reject else "success",
        f"p = {pvalue:.4f} {'<' if reject else '≥'} α = {p['alpha']:g}: {'se rechaza' if reject else 'no se rechaza'} la hipótesis de que el estadístico no difiere de lo esperado por azar.",
    )
    out.table(
        "Resultado",
        ["Estadístico", "Observado", "Media bajo H₀", "Desv. est. bajo H₀", "Valor p"],
        [[label, observed, float(null.mean()), float(null.std(ddof=1)), P(pvalue)]],
        note="El valor p es la proporción de permutaciones con un estadístico tan o más extremo que el observado (incluye la observada).",
    )
    out.chart(
        hist_chart(
            "Distribución del estadístico bajo la hipótesis nula",
            label,
            null,
            vlines=[{"value": observed, "label": "observado", "dash": "solid"}],
        )
    )
    return out.set(observed=observed, p=pvalue, permutations=len(null)).build()


def _permutation_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "# scipy.stats.permutation_test((x, y), statistic, permutation_type='independent', vectorized=True,\n"
        f"#                              n_resamples={int(p['resamples'])}, random_state={int(p['seed'])})\n"
        + load_line(dataset)
    )


# ============================================================================ Monte Carlo simulation
_DISTRIBUTIONS = (
    ("norm", "Normal (a = media, b = desviación estándar)"),
    ("uniform", "Uniforme (a = mínimo, b = máximo)"),
    ("expon", "Exponencial (a = media)"),
    ("lognorm", "Lognormal (a = μ del log, b = σ del log)"),
    ("t", "t de Student (a = grados de libertad)"),
    ("chi2", "Chi cuadrado (a = grados de libertad)"),
    ("gamma", "Gamma (a = forma, b = escala)"),
    ("poisson", "Poisson (a = media)"),
    ("binom", "Binomial (a = ensayos, b = probabilidad)"),
    ("bernoulli", "Bernoulli (a = probabilidad de éxito)"),
    ("randint", "Entero uniforme (a = mínimo, b = máximo, incluidos)"),
)


def _frozen(name: str, a: float, b: float | None):
    """A scipy distribution from the two generic parameters, or an error that says what is wrong."""
    b = 1.0 if b is None else b
    try:
        if name == "norm":
            assert b > 0
            return stats.norm(a, b)
        if name == "uniform":
            assert b > a
            return stats.uniform(a, b - a)
        if name == "expon":
            assert a > 0
            return stats.expon(scale=a)
        if name == "lognorm":
            assert b > 0
            return stats.lognorm(b, scale=np.exp(a))
        if name in ("t", "chi2"):
            assert a > 0
            return getattr(stats, name)(a)
        if name == "gamma":
            assert a > 0 and b > 0
            return stats.gamma(a, scale=b)
        if name == "poisson":
            assert a > 0
            return stats.poisson(a)
        if name == "binom":
            assert a >= 1 and a == int(a) and 0 < b < 1
            return stats.binom(int(a), b)
        if name == "bernoulli":
            assert 0 < a < 1
            return stats.bernoulli(a)
        assert b >= a and a == int(a) and b == int(b)
        return stats.randint(int(a), int(b) + 1)
    except AssertionError:
        raise InvalidInputError(
            f"Los parámetros no son válidos para la distribución «{dict(_DISTRIBUTIONS)[name].split(' (')[0]}»."
        ) from None


_SAMPLING = {
    "mean": ("Media", lambda x: x.mean(axis=1)),
    "median": ("Mediana", lambda x: np.median(x, axis=1)),
    "std": ("Desviación estándar", lambda x: x.std(axis=1, ddof=1)),
    "variance": ("Varianza", lambda x: x.var(axis=1, ddof=1)),
    "min": ("Mínimo", lambda x: x.min(axis=1)),
    "max": ("Máximo", lambda x: x.max(axis=1)),
    "range": ("Rango", lambda x: x.max(axis=1) - x.min(axis=1)),
}


def _monte_carlo(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    dist = _frozen(p["distribution"], p["a"], p["b"])
    n, reps = int(p["n"]), int(p["replications"])
    _draws(reps, n)
    rng = np.random.default_rng(int(p["seed"]))
    samples = dist.rvs(size=(reps, n), random_state=rng)
    name, function = _SAMPLING[p["statistic"]]
    values = np.asarray(function(samples), dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    quantiles = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
    out = Output(
        f"Simulación Monte Carlo: {name.lower()} muestral",
        f"{dict(_DISTRIBUTIONS)[p['distribution']].split(' (')[0]} · n = {n} · {reps:,} réplicas · semilla {int(p['seed'])}".replace(
            ",", " "
        ),
    )
    out.tiles(
        [
            ("Media simulada", mean),
            ("Error estándar simulado", sd),
            ("Percentil 2.5 %", float(quantiles[0])),
            ("Percentil 97.5 %", float(quantiles[4])),
        ]
    )
    rows = [[name, mean, sd, *(float(q) for q in quantiles), float(stats.skew(values))]]
    out.table(
        "Distribución muestral simulada",
        [
            "Estadístico",
            "Media",
            "Desv. est. (error estándar)",
            "2.5 %",
            "25 %",
            "Mediana",
            "75 %",
            "97.5 %",
            "Asimetría",
        ],
        rows,
    )
    curves = []
    if p["statistic"] == "mean":
        theory_mean, theory_sd = float(dist.mean()), float(dist.std() / np.sqrt(n))
        spread = max(theory_sd, 1e-12)
        grid = np.linspace(theory_mean - 4.5 * spread, theory_mean + 4.5 * spread, 200)
        curves = [
            {
                "name": "Normal del teorema central del límite",
                "x": grid,
                "y": stats.norm.pdf(grid, theory_mean, theory_sd),
            }
        ]
        out.table(
            "Contraste con la teoría",
            ["", "Teórico", "Simulado"],
            [["Media de la media muestral", theory_mean, mean], ["Error estándar (σ/√n)", theory_sd, sd]],
            note="La media muestral tiende a una normal al crecer n (teorema central del límite).",
        )
    elif p["statistic"] == "variance" and p["distribution"] == "norm":
        out.table(
            "Contraste con la teoría",
            ["", "Teórico", "Simulado"],
            [["Media de la varianza muestral (σ²)", float(dist.var()), mean]],
        )
    out.chart(hist_chart(f"Distribución muestral de la {name.lower()}", name, values, curves=curves, bins=60))
    return out.set(mean=mean, se=sd, quantiles=[float(q) for q in quantiles]).build()


def _monte_carlo_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nfrom scipy import stats\n\n"
        f"rng = np.random.default_rng({int(p['seed'])})\n"
        f"muestras = rng.normal(size=({int(p['replications'])}, {int(p['n'])}))  # sustituya por la distribución elegida: stats.{p['distribution']}(...).rvs(size=..., random_state=rng)\n"
        f"print(muestras.{'mean' if p['statistic'] == 'mean' else 'std'}(axis=1).std(ddof=1))\n"
    )


# ============================================================================ random data
def _random_data(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    n, columns = int(p["n"]), int(p["columns"])
    if n * columns > 5_000_000:
        raise InvalidInputError("Se admiten como máximo 5 millones de valores.")
    dist = _frozen(p["distribution"], p["a"], p["b"])
    rng = np.random.default_rng(int(p["seed"]))
    prefix = (p["prefix"] or "x").strip() or "x"
    data = pd.DataFrame({f"{prefix}{i + 1}": dist.rvs(size=n, random_state=rng) for i in range(columns)})
    out = Output(
        "Datos aleatorios",
        f"{dict(_DISTRIBUTIONS)[p['distribution']].split(' (')[0]} · {n} filas × {columns} columnas · semilla {int(p['seed'])}",
    )
    out.dataset(RESULT_KEY, "Datos aleatorios", data)
    out.tiles(
        [
            ("Filas", n),
            ("Columnas", columns),
            ("Media teórica", float(dist.mean())),
            ("Media simulada", float(data.to_numpy().mean())),
        ]
    )
    out.table(
        "Primeras filas",
        list(data.columns)[:8],
        [[float(v) if not float(v).is_integer() else int(v) for v in row][:8] for row in data.head(8).to_numpy()],
    )
    out.chart(hist_chart("Distribución de la primera columna", data.columns[0], data.iloc[:, 0].to_numpy()))
    return out.set(n=n, columns=columns).build()


def _random_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return f"import numpy as np\nimport pandas as pd\nfrom scipy import stats\n\nrng = np.random.default_rng({int(p['seed'])})\n# stats.{p['distribution']}(...) con los parámetros elegidos\ndf = pd.DataFrame({{f'x{{i}}': stats.norm.rvs(size={int(p['n'])}, random_state=rng) for i in range(1, {int(p['columns'])} + 1)}})\n"


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "bootstrap", "Bootstrap: intervalo de confianza de un estadístico", "Remuestreo",
        "Intervalo de confianza sin supuestos de distribución para una media, mediana, desviación, percentil, correlación o diferencia entre grupos.",
        _bootstrap, heavy=True,
        roles=(RoleSpec("var", "Variable", "numeric"), RoleSpec("var2", "Segunda columna (correlación)", "numeric", optional=True), RoleSpec("group", "Grupos de 2 categorías (diferencias)", "categorical", optional=True)),
        params=(
            FieldSpec("statistic", "Estadístico", "select", "mean", _STATISTICS),
            FieldSpec("percentile", "Percentil (0 a 100)", "number", 90, minimum=0, maximum=100, visible_if=("statistic", ("percentile",))),
            FieldSpec("method", "Método del intervalo", "select", "BCa", _METHODS),
            FieldSpec("confidence", "Nivel de confianza (%)", "number", 95, minimum=50, maximum=99.9),
            FieldSpec("resamples", "Réplicas bootstrap", "integer", 10000, minimum=500, maximum=200000), SEED,
        ),
        code=_bootstrap_code, example={"roles": {"var": "y"}, "params": {"statistic": "median", "resamples": 2000}},
    ),
    _spec(
        "permutation", "Prueba de permutación", "Remuestreo",
        "Valor p exacto o aproximado reordenando los datos al azar: diferencia de medias o medianas entre grupos, datos pareados o correlación.",
        _permutation, heavy=True,
        roles=(RoleSpec("var", "Variable", "numeric"), RoleSpec("var2", "Segunda columna (pareados / correlación)", "numeric", optional=True), RoleSpec("group", "Grupos de 2 categorías (diferencias)", "categorical", optional=True)),
        params=(
            FieldSpec("design", "Qué se compara", "select", "mean_diff", _PERM),
            FieldSpec("alternative", "Hipótesis alternativa", "select", "two-sided", (("two-sided", "Distinta (≠)"), ("greater", "Mayor (>)"), ("less", "Menor (<)"))),
            FieldSpec("resamples", "Permutaciones", "integer", 9999, minimum=999, maximum=200000),
            FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5), SEED,
        ),
        code=_permutation_code, example={"roles": {"var": "y", "group": "h"}, "params": {"resamples": 2000}},
    ),
    _spec(
        "monte_carlo", "Simulación Monte Carlo de una distribución muestral", "Simulación",
        "Simula miles de muestras de una distribución y muestra cómo se comporta un estadístico (teorema central del límite, sesgo, dispersión). No necesita dataset.",
        _monte_carlo, needs_dataset=False,
        params=(
            FieldSpec("distribution", "Distribución de la población", "select", "expon", _DISTRIBUTIONS),
            FieldSpec("a", "Parámetro a", "number", 1.0), FieldSpec("b", "Parámetro b (si aplica)", "number", 1.0, optional=True),
            FieldSpec("n", "Tamaño de cada muestra", "integer", 30, minimum=2, maximum=100000),
            FieldSpec("statistic", "Estadístico", "select", "mean", tuple((k, v[0]) for k, v in _SAMPLING.items())),
            FieldSpec("replications", "Réplicas", "integer", 10000, minimum=100, maximum=500000), SEED,
        ),
        code=_monte_carlo_code, example={"roles": {}, "params": {"distribution": "expon", "a": 2.0, "n": 25, "replications": 4000}},
    ),
    _spec(
        "random_data", "Generar datos aleatorios", "Simulación",
        "Crea un dataset con columnas aleatorias de una distribución (para ensayar métodos o simular escenarios).",
        _random_data, needs_dataset=False, outputs=(OutputSpec(RESULT_KEY, "Datos aleatorios", required=True),),
        params=(
            FieldSpec("distribution", "Distribución", "select", "norm", _DISTRIBUTIONS),
            FieldSpec("a", "Parámetro a", "number", 0.0), FieldSpec("b", "Parámetro b (si aplica)", "number", 1.0, optional=True),
            FieldSpec("n", "Filas", "integer", 100, minimum=1, maximum=1000000),
            FieldSpec("columns", "Columnas", "integer", 1, minimum=1, maximum=100),
            FieldSpec("prefix", "Prefijo del nombre de las columnas", "text", "x"), SEED,
        ),
        code=_random_code, example={"roles": {}, "params": {"distribution": "norm", "a": 10, "b": 2, "n": 50, "columns": 3}},
    ),
)  # fmt: skip

registry.register(*SPECS)
