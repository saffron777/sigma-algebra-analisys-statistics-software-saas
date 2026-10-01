"""Catalogue of the most common probability distributions."""

from __future__ import annotations

import math

from scipy import stats

from sigma.domain.distributions.spec import CONTINUOUS, DISCRETE, DistributionSpec, ParamSpec
from sigma.domain.errors import InvalidInputError


def _p(name, label, default, minimum=None, maximum=None, integer=False, exclusive_min=False):
    return ParamSpec(name, label, default, minimum, maximum, integer, exclusive_min)


_SPECS: tuple[DistributionSpec, ...] = (
    DistributionSpec(
        "normal",
        "Normal",
        CONTINUOUS,
        (_p("mu", "Media (μ)", 0.0), _p("sigma", "Desviación estándar (σ)", 1.0, 0, exclusive_min=True)),
        lambda mu, sigma: stats.norm(loc=mu, scale=sigma),
        "Distribución simétrica en forma de campana; base de las pruebas z.",
    ),
    DistributionSpec(
        "t",
        "t de Student",
        CONTINUOUS,
        (_p("df", "Grados de libertad", 10.0, 0, exclusive_min=True),),
        lambda df: stats.t(df),
        "Pruebas sobre medias con σ desconocida y coeficientes de regresión.",
    ),
    DistributionSpec(
        "chi2",
        "Chi-cuadrado (χ²)",
        CONTINUOUS,
        (_p("df", "Grados de libertad", 5.0, 0, exclusive_min=True),),
        lambda df: stats.chi2(df),
        "Pruebas de independencia, bondad de ajuste y varianza.",
    ),
    DistributionSpec(
        "f",
        "F de Fisher",
        CONTINUOUS,
        (
            _p("dfn", "gl numerador", 4.0, 0, exclusive_min=True),
            _p("dfd", "gl denominador", 20.0, 0, exclusive_min=True),
        ),
        lambda dfn, dfd: stats.f(dfn, dfd),
        "ANOVA, significancia global de la regresión y cociente de varianzas.",
    ),
    DistributionSpec(
        "binomial",
        "Binomial",
        DISCRETE,
        (_p("n", "Ensayos (n)", 10, 1, integer=True), _p("p", "Probabilidad de éxito (p)", 0.5, 0, 1)),
        lambda n, p: stats.binom(n, p),
        "Número de éxitos en n ensayos independientes.",
    ),
    DistributionSpec(
        "poisson",
        "Poisson",
        DISCRETE,
        (_p("mu", "Media (λ)", 3.0, 0, exclusive_min=True),),
        lambda mu: stats.poisson(mu),
        "Conteo de eventos raros en un intervalo.",
    ),
    DistributionSpec(
        "exponential",
        "Exponencial",
        CONTINUOUS,
        (_p("rate", "Tasa (λ)", 1.0, 0, exclusive_min=True),),
        lambda rate: stats.expon(scale=1.0 / rate),
        "Tiempo entre eventos de un proceso de Poisson.",
    ),
    DistributionSpec(
        "uniform",
        "Uniforme continua",
        CONTINUOUS,
        (_p("a", "Mínimo (a)", 0.0), _p("b", "Máximo (b)", 1.0)),
        lambda a, b: _uniform(a, b),
        "Todos los valores del intervalo [a, b] son equiprobables.",
    ),
    DistributionSpec(
        "gamma",
        "Gamma",
        CONTINUOUS,
        (_p("shape", "Forma (k)", 2.0, 0, exclusive_min=True), _p("scale", "Escala (θ)", 1.0, 0, exclusive_min=True)),
        lambda shape, scale: stats.gamma(shape, scale=scale),
        "Suma de variables exponenciales; tiempos de espera.",
    ),
    DistributionSpec(
        "beta",
        "Beta",
        CONTINUOUS,
        (_p("alpha", "α", 2.0, 0, exclusive_min=True), _p("beta", "β", 5.0, 0, exclusive_min=True)),
        lambda alpha, beta: stats.beta(alpha, beta),
        "Proporciones y probabilidades en el intervalo [0, 1].",
    ),
    DistributionSpec(
        "weibull",
        "Weibull",
        CONTINUOUS,
        (_p("shape", "Forma (k)", 1.5, 0, exclusive_min=True), _p("scale", "Escala (λ)", 1.0, 0, exclusive_min=True)),
        lambda shape, scale: stats.weibull_min(shape, scale=scale),
        "Tiempos de vida y análisis de confiabilidad.",
    ),
    DistributionSpec(
        "lognormal",
        "Log-normal",
        CONTINUOUS,
        (_p("mu", "μ del logaritmo", 0.0), _p("sigma", "σ del logaritmo", 1.0, 0, exclusive_min=True)),
        lambda mu, sigma: stats.lognorm(sigma, scale=math.exp(mu)),
        "Variables positivas con asimetría (ingresos, precios).",
    ),
    DistributionSpec(
        "geometric",
        "Geométrica",
        DISCRETE,
        (_p("p", "Probabilidad de éxito (p)", 0.3, 0, 1, exclusive_min=True),),
        lambda p: stats.geom(p),
        "Número de ensayos hasta el primer éxito (1, 2, 3, …).",
    ),
    DistributionSpec(
        "hypergeometric",
        "Hipergeométrica",
        DISCRETE,
        (
            _p("population", "Población (M)", 50, 1, integer=True),
            _p("successes", "Éxitos en la población (K)", 10, 0, integer=True),
            _p("draws", "Extracciones (n)", 10, 0, integer=True),
        ),
        lambda population, successes, draws: _hypergeom(population, successes, draws),
        "Éxitos en n extracciones sin reemplazo.",
    ),
    DistributionSpec(
        "negative_binomial",
        "Binomial negativa",
        DISCRETE,
        (
            _p("r", "Éxitos requeridos (r)", 3.0, 0, exclusive_min=True),
            _p("p", "Probabilidad de éxito (p)", 0.5, 0, 1, exclusive_min=True),
        ),
        lambda r, p: stats.nbinom(r, p),
        "Número de fracasos antes de alcanzar r éxitos.",
    ),
)


def _uniform(a: float, b: float):
    if b <= a:
        raise InvalidInputError("En la uniforme, el máximo (b) debe ser mayor que el mínimo (a).")
    return stats.uniform(loc=a, scale=b - a)


def _hypergeom(population: int, successes: int, draws: int):
    if successes > population or draws > population:
        raise InvalidInputError("En la hipergeométrica, K y n no pueden superar a la población M.")
    return stats.hypergeom(population, successes, draws)


REGISTRY: dict[str, DistributionSpec] = {spec.key: spec for spec in _SPECS}


def get_distribution(key: str) -> DistributionSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise InvalidInputError(f"Distribución desconocida: «{key}».") from None


def list_distributions() -> list[DistributionSpec]:
    return list(_SPECS)
