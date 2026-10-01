"""Report for the distribution calculator."""

from __future__ import annotations

import re

from sigma.domain.distributions import DistributionRef, get_distribution
from sigma.domain.distributions.spec import DISCRETE
from sigma.domain.results import DistributionCalcResult
from sigma.presentation.web.charts.distribution_plot import distribution_figure
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import num, pval

TAIL_LABELS = {"two-sided": "bilateral", "left": "cola izquierda", "right": "cola derecha"}
_SYMBOL = re.compile(r"\(([^)]+)\)")


def params_text(key: str, params: dict[str, float]) -> str:
    """'μ = 0, σ = 1' built from the short symbols in the parameter labels."""
    parts = []
    for spec in get_distribution(key).params:
        symbol = _SYMBOL.search(spec.label)
        parts.append(f"{symbol.group(1) if symbol else spec.label} = {num(params[spec.name])}")
    return ", ".join(parts)


def build_report(result: DistributionCalcResult) -> b.ReportBuilder:
    spec = get_distribution(result.distribution)
    discrete = spec.kind == DISCRETE
    title = f"{result.label} ({params_text(result.distribution, result.params)})"
    builder = b.ReportBuilder(title, spec.description)
    ref = DistributionRef(result.distribution, result.params)
    values = result.values

    if result.mode == "point":
        x = num(result.marker)
        builder.add(
            b.tiles(
                [
                    (f"{'P(X = x)' if discrete else 'f(x) densidad'}", num(values["density"], 6), f"en x = {x}"),
                    ("P(X ≤ x)", num(values["cdf"], 6), "acumulada"),
                    ("P(X ≥ x)" if discrete else "P(X > x)", num(values["survival_inclusive"], 6), "cola derecha"),
                ]
            )
        )
        figure = distribution_figure(ref, statistic=result.marker, statistic_label="x", left_of=None)
    elif result.mode == "between":
        low, high = result.bounds or (0.0, 0.0)
        builder.add(
            b.tiles(
                [(f"P({num(low)} ≤ X ≤ {num(high)})", num(values["probability"], 6), "probabilidad entre los límites")]
            )
        )
        figure = distribution_figure(ref, interval=(low, high))
    elif result.mode == "quantile":
        builder.add(
            b.tiles(
                [
                    ("Cuantil", num(values["quantile"], 6), f"P(X ≤ cuantil) = {num(values['probability'])}"),
                ]
            )
        )
        figure = distribution_figure(ref, left_of=values["quantile"])
    else:
        p = values["p_value"]
        tail = result.tail or "two-sided"
        tiles = [("Valor p", pval(p), TAIL_LABELS[tail]), ("Estadístico", num(result.marker, 6))]
        if result.alpha is not None and not discrete:
            tiles.append(("α", num(result.alpha)))
        builder.add(b.tiles(tiles))
        if result.critical_values:
            builder.add(
                b.text(
                    "Valores críticos al nivel α = "
                    + num(result.alpha)
                    + ": "
                    + ", ".join(num(c, 6) for c in result.critical_values)
                )
            )
        if result.alpha is not None:
            reject = p < result.alpha
            builder.add(
                b.alert(
                    "success" if reject else "info",
                    f"p = {pval(p)} {'<' if reject else '≥'} α = {num(result.alpha)}: "
                    + (
                        "el estadístico cae en la región de rechazo."
                        if reject
                        else "el estadístico no cae en la región de rechazo."
                    ),
                )
            )
        figure = distribution_figure(
            ref,
            statistic=result.marker,
            tail=tail,
            alpha=result.alpha if not discrete else None,
            show_p_value=True,
            statistic_label="estadístico",
        )

    builder.add_figure(figure, "Distribución con las áreas sombreadas")
    moments = result.moments
    builder.add(
        b.table(
            "Características de la distribución",
            ["Media", "Varianza", "Desv. estándar", "Asimetría", "Curtosis (exceso)"],
            [
                [
                    num(moments["mean"]),
                    num(moments["variance"]),
                    num(moments["std"]),
                    num(moments["skewness"]),
                    num(moments["excess_kurtosis"]),
                ]
            ],
        )
    )
    return builder
