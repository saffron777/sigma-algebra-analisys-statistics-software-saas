"""More power and sample-size designs (Minitab *Power and Sample Size*): proportions, one-way ANOVA, correlation, equivalence."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart
from sigma.domain.analysis.output import Output, strong
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec
from sigma.domain.errors import ComputationError, InvalidInputError

GROUP = "Más diseños"
_SOLVE = (("n", "Tamaño de muestra"), ("power", "Potencia"), ("effect", "Efecto detectable"))
_ALTERNATIVES = (("two-sided", "Bilateral (≠)"), ("larger", "Cola derecha (>)"), ("smaller", "Cola izquierda (<)"))
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.0001, maximum=0.5)
POWER = FieldSpec(
    "power", "Potencia deseada", "number", 0.8, minimum=0.01, maximum=0.999, visible_if=("solve_for", ("n", "effect"))
)
SOLVE = FieldSpec("solve_for", "Calcular", "select", "n", _SOLVE)


def _n_field(label: str, default: float = 30) -> FieldSpec:
    return FieldSpec("n", label, "number", default, minimum=2, visible_if=("solve_for", ("power", "effect")))


def _finite(*values: float) -> None:
    if not all(v is not None and math.isfinite(v) for v in values):
        raise ComputationError("No se pudo despejar el valor pedido: revise el efecto, α y la potencia.")


def _report(
    out: Output, *, unit: str, n_label: str, effect_label: str, solve: str, n: float, power: float, effect: float,
    alpha: float, extra: list[list[Any]], curve: Callable[[float, float], float], curve_label: Callable[[float], str],
    effect_steps: tuple[float, ...] = (0.5, 1.0, 1.5), note: str | None = None,
) -> None:  # fmt: skip
    """Tiles, parameters table and power curves shared by the designs."""
    shown_n = int(math.ceil(n)) if solve == "n" else n
    out.tiles([(n_label, shown_n), ("Potencia", power), (effect_label, effect), ("α", alpha)])
    if solve == "n":
        out.alert("info", f"Se necesitan {shown_n:g} {unit} para alcanzar una potencia de {power:.4f}.")
    out.table(
        "Parámetros",
        ["Parámetro", "Valor"],
        [
            [n_label, strong(shown_n) if solve == "n" else n],
            ["Potencia", strong(power) if solve == "power" else power],
            [effect_label, strong(effect) if solve == "effect" else effect],
            ["α", alpha],
            *extra,
        ],
        note=note,
    )
    top = max(10, int(math.ceil(shown_n * 2.2)))
    grid = np.unique(np.linspace(3, top, 50))
    curves = [
        {
            "name": curve_label(effect * factor),
            "x": grid,
            "y": [curve(effect * factor, float(m)) for m in grid],
            "dash": dash,
        }
        for factor, dash in zip(effect_steps, ("dot", "solid", "dash"), strict=False)
    ]
    out.chart(
        line_chart(
            "Curvas de potencia",
            n_label,
            "Potencia",
            curves,
            hlines=[{"value": 0.8, "label": "0.80"}],
            vlines=[{"value": shown_n, "label": f"n = {shown_n:g}"}],
        )
    )


def _bracket(power_of: Callable[[float], float], target: float, low: float, high: float) -> float:
    """The argument in [low, high] where an increasing function reaches ``target``."""
    if power_of(high) < target:
        raise ComputationError(
            "Con esos datos no se alcanza la potencia pedida: aumente el tamaño de muestra o el efecto."
        )
    if power_of(low) >= target:
        return low
    return float(optimize.brentq(lambda v: power_of(v) - target, low, high, xtol=1e-9))


# ============================================================================ proportions
def _prop_power(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    design, solve, alpha, alt = p["design"], p["solve_for"], p["alpha"], p["alternative"]
    p0 = p["p0"]
    ratio = p["ratio"] if design == "two" else 0.0
    solver = NormalIndPower()

    def power_at(p1: float, n: float) -> float:
        h = abs(float(proportion_effectsize(p1, p0)))
        return float(
            solver.power(
                effect_size=h,
                nobs1=n,
                alpha=alpha,
                ratio=ratio or 0.0,
                alternative="two-sided" if alt == "two-sided" else "larger",
            )
        )

    p1, n, power = p.get("p1"), p.get("n"), p.get("power")
    direction = -1 if alt == "smaller" else 1
    if solve in ("n", "power") and (p1 is None or p1 == p0):
        raise InvalidInputError("La proporción alternativa debe ser distinta de la de referencia.")
    if not 0 < p0 < 1:
        raise InvalidInputError("La proporción de referencia debe estar entre 0 y 1.")
    try:
        if solve == "n":
            h = abs(float(proportion_effectsize(p1, p0)))
            n = float(
                solver.solve_power(
                    effect_size=h,
                    nobs1=None,
                    alpha=alpha,
                    power=power,
                    ratio=ratio or 0.0,
                    alternative="two-sided" if alt == "two-sided" else "larger",
                )
            )
        elif solve == "power":
            power = power_at(p1, n)
        else:
            h = float(
                solver.solve_power(
                    effect_size=None,
                    nobs1=n,
                    alpha=alpha,
                    power=power,
                    ratio=ratio or 0.0,
                    alternative="two-sided" if alt == "two-sided" else "larger",
                )
            )
            angle = math.asin(math.sqrt(p0)) + (h / 2 if direction == 1 or alt == "two-sided" else -h / 2)
            if not 0 < angle < math.pi / 2:
                raise ComputationError(
                    "Con ese tamaño de muestra no se puede detectar ninguna diferencia (la proporción saldría fuera de [0, 1])."
                )
            p1 = math.sin(angle) ** 2
    except (ValueError, RuntimeError) as error:
        raise ComputationError(f"No se pudo despejar el valor pedido: {error}") from None
    _finite(n, power, p1)
    n_up = math.ceil(n) if solve == "n" else n
    actual = power_at(p1, n_up) if solve == "n" else power
    out = Output(
        f"Potencia y tamaño de muestra: {'una proporción' if design == 'one' else 'dos proporciones'}",
        f"α = {alpha:g} · {dict(_ALTERNATIVES)[alt]}",
    )
    _report(
        out, unit="observaciones" + (" por grupo" if design == "two" else ""), n_label="Tamaño de muestra" + (" por grupo" if design == "two" else ""),
        effect_label="Proporción alternativa", solve=solve, n=n_up, power=actual, effect=float(p1), alpha=alpha,
        extra=[["Proporción de referencia", p0], ["Diferencia", float(p1 - p0)], ["Efecto h de Cohen", abs(float(proportion_effectsize(p1, p0)))]],
        curve=lambda value, m: power_at(min(max(value, 1e-6), 1 - 1e-6), m),
        curve_label=lambda value: f"p = {value:.3g}", effect_steps=(1.0,), note="Usa la aproximación normal con la transformación arcoseno (h de Cohen).",
    )  # fmt: skip
    return out.set(n=n_up, power=actual, p1=float(p1), p0=p0).build()


def _prop_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.stats.power import NormalIndPower\nfrom statsmodels.stats.proportion import proportion_effectsize\n\n"
        f"h = proportion_effectsize({p.get('p1')!r}, {p['p0']!r})\n"
        f"print(NormalIndPower().solve_power(effect_size=h, nobs1={p.get('n')!r}, alpha={p['alpha']!r}, power={p.get('power')!r}, ratio={0.0 if p['design'] == 'one' else p['ratio']!r}))\n"
    )


# ============================================================================ one-way ANOVA
def _anova_power(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    k, solve, alpha = int(p["k"]), p["solve_for"], p["alpha"]
    solver = FTestAnovaPower()
    f, n, power = p.get("effect"), p.get("n"), p.get("power")
    if solve != "effect" and not f:
        raise InvalidInputError("El efecto f debe ser mayor que 0.")

    def power_at(effect: float, per_group: float) -> float:
        return float(solver.power(effect_size=effect, nobs=per_group * k, alpha=alpha, k_groups=k))

    try:
        if solve == "n":
            n = float(solver.solve_power(effect_size=f, nobs=None, alpha=alpha, power=power, k_groups=k)) / k
        elif solve == "power":
            power = power_at(f, n)
        else:
            f = float(solver.solve_power(effect_size=None, nobs=n * k, alpha=alpha, power=power, k_groups=k))
    except (ValueError, RuntimeError) as error:
        raise ComputationError(f"No se pudo despejar el valor pedido: {error}") from None
    _finite(n, power, f)
    n_up = math.ceil(n) if solve == "n" else n
    actual = power_at(f, n_up) if solve == "n" else power
    out = Output(f"Potencia y tamaño de muestra: ANOVA de un factor ({k} grupos)", f"α = {alpha:g}")
    _report(
        out, unit="observaciones por grupo", n_label="Tamaño de muestra por grupo", effect_label="Efecto f de Cohen", solve=solve, n=n_up,
        power=actual, effect=float(f), alpha=alpha,
        extra=[["Grupos", k], ["Tamaño total", float(n_up * k)]],
        curve=lambda value, m: power_at(value, m), curve_label=lambda value: f"f = {value:.3g}",
        note="f = desviación estándar de las medias de los grupos / desviación estándar dentro de los grupos: 0.1 pequeño · 0.25 mediano · 0.4 grande.",
    )  # fmt: skip
    return out.set(n_per_group=n_up, total=n_up * k, power=actual, effect_f=float(f)).build()


def _anova_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return f"from statsmodels.stats.power import FTestAnovaPower\n\nprint(FTestAnovaPower().solve_power(effect_size={p.get('effect')!r}, nobs=None, alpha={p['alpha']!r}, power={p.get('power')!r}, k_groups={int(p['k'])}))  # tamaño TOTAL\n"


# ============================================================================ correlation
def _corr_power(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    solve, alpha, alt = p["solve_for"], p["alpha"], p["alternative"]
    za = stats.norm.ppf(1 - alpha / (2 if alt == "two-sided" else 1))
    r, n, power = p.get("r"), p.get("n"), p.get("power")

    def power_at(rho: float, size: float) -> float:
        shift = abs(math.atanh(rho)) * math.sqrt(size - 3)
        beta = stats.norm.cdf(shift - za)
        return float(beta + (stats.norm.cdf(-shift - za) if alt == "two-sided" else 0.0))

    if solve != "effect" and (r is None or r == 0 or abs(r) >= 1):
        raise InvalidInputError("La correlación r debe estar entre −1 y 1 y ser distinta de 0.")
    if solve == "n":
        n = ((za + stats.norm.ppf(power)) / abs(math.atanh(r))) ** 2 + 3
    elif solve == "power":
        if n <= 3:
            raise InvalidInputError("El tamaño de muestra debe ser mayor que 3.")
        power = power_at(r, n)
    else:
        if n <= 3:
            raise InvalidInputError("El tamaño de muestra debe ser mayor que 3.")
        r = math.tanh(_bracket(lambda z: power_at(math.tanh(z), n), power, 1e-6, 6.0))
    _finite(n, power, r)
    n_up = math.ceil(n) if solve == "n" else n
    actual = power_at(r, n_up) if solve == "n" else power
    out = Output("Potencia y tamaño de muestra: correlación", f"α = {alpha:g} · {dict(_ALTERNATIVES)[alt]}")
    _report(
        out, unit="pares de observaciones", n_label="Tamaño de muestra", effect_label="Correlación r", solve=solve, n=n_up, power=actual,
        effect=float(abs(r)), alpha=alpha, extra=[["Transformación z de Fisher", math.atanh(abs(r))]],
        curve=lambda value, m: power_at(min(max(value, 1e-6), 0.9999), m), curve_label=lambda value: f"r = {value:.3g}",
        note="Usa la transformación z de Fisher: n = ((z_α + z_β) / atanh r)² + 3.",
    )  # fmt: skip
    return out.set(n=n_up, power=actual, r=float(abs(r))).build()


def _corr_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nfrom scipy import stats\n\n"
        f"r, alpha, power = {p.get('r')!r}, {p['alpha']!r}, {p.get('power')!r}\n"
        f"n = ((stats.norm.ppf(1 - alpha / {2 if p['alternative'] == 'two-sided' else 1}) + stats.norm.ppf(power)) / np.arctanh(r)) ** 2 + 3\nprint(np.ceil(n))\n"
    )


# ============================================================================ equivalence (TOST) of two means
def tost_power(difference: float, margin: float, sd: float, n: float, alpha: float) -> float:
    """Exact power of the two one-sided t tests for two groups of ``n`` (equal variances), by integrating over s."""
    df = 2 * n - 2
    a = math.sqrt(2 / n)
    t_crit = stats.t.ppf(1 - alpha, df)

    def integrand(u: float) -> float:
        s = sd * math.sqrt(u / df)  # the sample standard deviation
        upper = (margin - t_crit * s * a - difference) / (sd * a)
        lower = (-margin + t_crit * s * a - difference) / (sd * a)
        inside = stats.norm.cdf(upper) - stats.norm.cdf(lower)
        return max(inside, 0.0) * stats.chi2.pdf(u, df)

    low, high = stats.chi2.ppf(1e-10, df), stats.chi2.ppf(1 - 1e-10, df)  # where the mass of s² is, however many df
    value, _ = integrate.quad(integrand, low, high, points=[df], limit=400)
    return float(min(max(value, 0.0), 1.0))


def _tost_power(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    solve, alpha, margin, sd, delta = p["solve_for"], p["alpha"], p["margin"], p["sd"], p["difference"]
    n, power = p.get("n"), p.get("power")
    if abs(delta) >= margin:
        raise InvalidInputError(
            "La diferencia verdadera debe estar dentro de los límites de equivalencia (|diferencia| < margen)."
        )
    if solve == "n":
        n = _bracket(lambda m: tost_power(delta, margin, sd, m, alpha), power, 3, 100000)
    elif solve == "power":
        if n < 3:
            raise InvalidInputError("Se necesitan al menos 3 observaciones por grupo.")
        power = tost_power(delta, margin, sd, n, alpha)
    else:
        raise InvalidInputError("Esta prueba despeja el tamaño de muestra o la potencia.")
    n_up = math.ceil(n) if solve == "n" else n
    actual = tost_power(delta, margin, sd, n_up, alpha) if solve == "n" else power
    out = Output(
        "Potencia y tamaño de muestra: equivalencia de dos medias (TOST)",
        f"α = {alpha:g} · límites ±{margin:g} · σ = {sd:g}",
    )
    out.tiles(
        [
            ("Tamaño de muestra por grupo", int(n_up) if solve == "n" else n),
            ("Potencia", actual),
            ("Diferencia verdadera", delta),
            ("Margen", margin),
        ]
    )
    if solve == "n":
        out.alert(
            "info",
            f"Se necesitan {n_up:g} observaciones por grupo para demostrar equivalencia con potencia {actual:.4f}.",
        )
    out.table(
        "Parámetros",
        ["Parámetro", "Valor"],
        [
            ["Tamaño por grupo", strong(n_up) if solve == "n" else n],
            ["Potencia", strong(actual) if solve == "power" else actual],
            ["Diferencia verdadera entre medias", delta],
            ["Límite de equivalencia (±)", margin],
            ["Desviación estándar σ", sd],
            ["α", alpha],
        ],
        note="Potencia exacta de las dos pruebas t unilaterales con varianzas iguales.",
    )
    top = max(10, int(math.ceil(n_up * 2.2)))
    grid = np.unique(np.linspace(3, top, 30))
    curves = [
        {
            "name": f"diferencia = {d:.3g}",
            "x": grid,
            "y": [tost_power(d, margin, sd, float(m), alpha) for m in grid],
            "dash": dash,
        }
        for d, dash in ((0.0, "dot"), (delta, "solid"), (min(abs(delta) + 0.25 * margin, 0.95 * margin), "dash"))
    ]
    out.chart(
        line_chart(
            "Curvas de potencia",
            "Tamaño de muestra por grupo",
            "Potencia",
            curves,
            hlines=[{"value": 0.8, "label": "0.80"}],
            vlines=[{"value": n_up, "label": f"n = {n_up:g}"}],
        )
    )
    return out.set(n=n_up, power=actual).build()


def _tost_power_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "# Potencia exacta del TOST de dos muestras (varianzas iguales), integrando sobre la desviación muestral\n"
        "import numpy as np\nfrom scipy import integrate, stats\n\n"
        f"diff, margin, sd, n, alpha = {p['difference']!r}, {p['margin']!r}, {p['sd']!r}, {p.get('n') or 30!r}, {p['alpha']!r}\n"
        "df, a = 2 * n - 2, np.sqrt(2 / n)\ntc = stats.t.ppf(1 - alpha, df)\n"
        "f = lambda u: max(stats.norm.cdf((margin - tc * np.sqrt(u / df) * a - diff) / (sd * a))\n"
        "                  - stats.norm.cdf((-margin + tc * np.sqrt(u / df) * a - diff) / (sd * a)), 0) * stats.chi2.pdf(u, df)\n"
        "print(integrate.quad(f, stats.chi2.ppf(1e-10, df), stats.chi2.ppf(1 - 1e-10, df), points=[df], limit=400)[0])\n"
    )


SPECS = (
    AnalysisSpec(
        "prop_power", "Proporciones: potencia y tamaño de muestra", GROUP, "potencia",
        "Una proporción frente a un valor, o dos proporciones: despeje el tamaño de muestra, la potencia o la proporción detectable.",
        compute=_prop_power, needs_dataset=False,
        params=(
            FieldSpec("design", "Diseño", "select", "two", (("one", "Una proporción frente a un valor"), ("two", "Dos proporciones independientes"))),
            SOLVE,
            FieldSpec("p0", "Proporción de referencia (H₀ o grupo 2)", "number", 0.5, minimum=0.0001, maximum=0.9999),
            FieldSpec("p1", "Proporción alternativa a detectar", "number", 0.6, minimum=0.0001, maximum=0.9999, visible_if=("solve_for", ("n", "power"))),
            _n_field("Tamaño de muestra (por grupo)"), POWER, ALPHA,
            FieldSpec("alternative", "Hipótesis alternativa", "select", "two-sided", _ALTERNATIVES),
            FieldSpec("ratio", "Razón n₂ / n₁", "number", 1.0, minimum=0.1, maximum=20, visible_if=("design", ("two",))),
        ),
        code=_prop_code, example={"roles": {}, "params": {"design": "one", "p0": 0.5, "p1": 0.6}},
    ),
    AnalysisSpec(
        "anova_power", "ANOVA de un factor: potencia y tamaño de muestra", GROUP, "potencia",
        "Despeje el tamaño por grupo, la potencia o el efecto f detectable al comparar varias medias.",
        compute=_anova_power, needs_dataset=False,
        params=(
            FieldSpec("k", "Número de grupos", "integer", 4, minimum=2, maximum=50), SOLVE,
            FieldSpec("effect", "Efecto f de Cohen", "number", 0.25, minimum=0.001, help="0.1 pequeño · 0.25 mediano · 0.4 grande.", visible_if=("solve_for", ("n", "power"))),
            _n_field("Tamaño de muestra por grupo"), POWER, ALPHA,
        ),
        code=_anova_code, example={"roles": {}, "params": {"k": 4, "effect": 0.25}},
    ),
    AnalysisSpec(
        "corr_power", "Correlación: potencia y tamaño de muestra", GROUP, "potencia",
        "Cuántos pares de observaciones hacen falta para detectar una correlación r (transformación z de Fisher).",
        compute=_corr_power, needs_dataset=False,
        params=(
            SOLVE, FieldSpec("r", "Correlación r a detectar", "number", 0.3, minimum=-0.999, maximum=0.999, visible_if=("solve_for", ("n", "power"))),
            _n_field("Tamaño de muestra (pares)", 50), POWER, ALPHA,
            FieldSpec("alternative", "Hipótesis alternativa", "select", "two-sided", (("two-sided", "Bilateral (≠ 0)"), ("larger", "Unilateral"))),
        ),
        code=_corr_code, example={"roles": {}, "params": {"r": 0.3}},
    ),
    AnalysisSpec(
        "tost_power", "Equivalencia (TOST) de dos medias: potencia y tamaño de muestra", GROUP, "potencia",
        "Tamaño de muestra o potencia para demostrar que dos medias son equivalentes dentro de un margen.",
        compute=_tost_power, needs_dataset=False,
        params=(
            FieldSpec("solve_for", "Calcular", "select", "n", (("n", "Tamaño de muestra"), ("power", "Potencia"))),
            FieldSpec("margin", "Límite de equivalencia (±)", "number", 2.0, minimum=1e-9),
            FieldSpec("difference", "Diferencia verdadera esperada entre medias", "number", 0.0),
            FieldSpec("sd", "Desviación estándar σ", "number", 3.0, minimum=1e-9),
            _n_field("Tamaño de muestra por grupo"), POWER, ALPHA,
        ),
        heavy=False, code=_tost_power_code, example={"roles": {}, "params": {"margin": 2.0, "difference": 0.5, "sd": 3.0}},
    ),
)  # fmt: skip

registry.register(*SPECS)
