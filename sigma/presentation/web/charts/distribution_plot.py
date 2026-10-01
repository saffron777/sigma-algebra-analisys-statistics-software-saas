"""The reusable probability-distribution chart (density/mass with shaded regions and p-value).

Every module that reports a reference-distribution p-value (hypothesis tests, regression, ANOVA,
descriptive normality, chi-square) draws it through :func:`distribution_figure`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from sigma.domain.distributions import DistributionRef, get_distribution
from sigma.domain.distributions.probability import critical_values, curve, frozen_from_ref, plot_support
from sigma.domain.distributions.spec import DISCRETE
from sigma.presentation.web.charts.common import (
    PALETTE,
    PROBABILITY_FILL,
    PVALUE_FILL,
    REJECTION_FILL,
    figure_json,
    new_figure,
)

Region = tuple[float, float]  # (low, high); ±inf mean "to the edge of the plot"
_KIND_FILL = {"rejection": REJECTION_FILL, "pvalue": PVALUE_FILL, "probability": PROBABILITY_FILL}
_KIND_LABEL = {"rejection": "Región de rechazo", "pvalue": "Valor p", "probability": "Probabilidad"}
_KIND_SOLID = {"rejection": "#ef4444", "pvalue": "#4f46e5", "probability": "#10b981"}
_BAR_BASE = "#a5b4fc"


def _tail_regions(dist: Any, point: float, tail: str, discrete: bool) -> list[Region]:
    """Regions whose total probability equals the p-value of ``point``."""
    if tail == "right":
        return [(point, math.inf)]
    if tail == "left":
        return [(-math.inf, point)]
    left, right = float(dist.cdf(point)), float(dist.sf(point - 1 if discrete else point))
    if right <= left:  # statistic in the upper half: mirror the right tail onto the left
        mirrored = float(dist.ppf(right)) if right > 0 else -math.inf
        if discrete:
            mirrored -= 1 if dist.cdf(mirrored) > right else 0
        return [(point, math.inf), (-math.inf, mirrored)]
    mirrored = float(dist.isf(left)) if left > 0 else math.inf
    return [(-math.inf, point), (mirrored, math.inf)]


def _rejection_regions(dist: Any, alpha: float, tail: str) -> list[Region]:
    crit = critical_values(dist, alpha, tail)
    if tail == "right":
        return [(crit[0], math.inf)]
    if tail == "left":
        return [(-math.inf, crit[0])]
    return [(-math.inf, crit[0]), (crit[1], math.inf)]


def distribution_figure(
    ref: DistributionRef,
    *,
    title: str | None = None,
    x_title: str = "x",
    statistic: float | None = None,
    statistic_label: str = "estadístico",
    tail: str | None = None,
    alpha: float | None = None,
    show_p_value: bool = False,
    interval: tuple[float, float] | None = None,
    left_of: float | None = None,
    extra_markers: Sequence[tuple[float, str]] = (),
    height: int = 360,
) -> dict[str, Any]:
    """Draw ``ref`` and shade the requested regions.

    * ``alpha`` + ``tail``           → rejection region(s)
    * ``show_p_value`` + ``statistic`` + ``tail`` → area equal to the p-value
    * ``interval``                   → P(a ≤ X ≤ b)
    * ``left_of``                    → P(X ≤ x)
    """
    spec = get_distribution(ref.key)
    dist = frozen_from_ref(ref)
    discrete = spec.kind == DISCRETE

    regions: list[tuple[str, Region]] = []
    focus = [m for m in (statistic, left_of) if m is not None] + [m for m, _ in extra_markers]
    if interval:
        focus.extend(interval)
    if alpha and tail:
        regions += [("rejection", r) for r in _rejection_regions(dist, alpha, tail)] if not discrete else []
        focus.extend(c for c in critical_values(dist, alpha, tail))
    if show_p_value and statistic is not None and tail:
        regions += [("pvalue", r) for r in _tail_regions(dist, statistic, tail, discrete)]
    if interval:
        regions.append(("probability", (interval[0], interval[1])))
    if left_of is not None:
        regions.append(("probability", (-math.inf, left_of)))

    support = plot_support(dist, discrete, tuple(float(f) for f in focus if f is not None and math.isfinite(f)))
    x, y = curve(dist, support)
    fig = new_figure(title, x_title, "Probabilidad" if discrete else "Densidad", height=height)

    if discrete:
        colors = [_BAR_BASE] * len(x)
        for kind, (low, high) in regions:
            for i, value in enumerate(x):
                if low <= value <= high:
                    colors[i] = _KIND_SOLID[kind]
        fig.add_trace(
            go.Bar(
                x=x.tolist(),
                y=y.tolist(),
                marker={"color": colors},
                name="Función de masa",
                hovertemplate="x=%{x}<br>P=%{y:.5f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=y.tolist(),
                mode="lines",
                line={"color": PALETTE[0], "width": 2.4},
                name="Densidad",
                hovertemplate="x=%{x:.4g}<br>f(x)=%{y:.4g}<extra></extra>",
            )
        )
        shown: set[str] = set()
        for kind, (low, high) in regions:
            lo, hi = max(low, support.low), min(high, support.high)
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, 160)
            fig.add_trace(
                go.Scatter(
                    x=grid.tolist(),
                    y=np.asarray(dist.pdf(grid)).tolist(),
                    mode="lines",
                    fill="tozeroy",
                    line={"width": 0},
                    fillcolor=_KIND_FILL[kind],
                    name=_KIND_LABEL[kind],
                    legendgroup=kind,
                    showlegend=kind not in shown,
                    hoverinfo="skip",
                )
            )
            shown.add(kind)

    top = float(np.nanmax(y)) * 1.12 if len(y) else 1.0
    markers = [(statistic, statistic_label)] if statistic is not None else []
    markers += [(left_of, "x")] if left_of is not None else []
    markers += list(extra_markers)
    for position, label in markers:
        fig.add_shape(
            type="line", x0=position, x1=position, y0=0, y1=top, line={"color": "#ef4444", "width": 2, "dash": "dash"}
        )
        fig.add_annotation(
            x=position,
            y=top,
            text=f"{label} = {position:.4g}",
            showarrow=False,
            yshift=10,
            font={"color": "#ef4444", "size": 12},
        )
    fig.update_yaxes(range=[0, top * 1.08])
    fig.update_xaxes(range=[support.low - (0.5 if discrete else 0), support.high + (0.5 if discrete else 0)])
    return figure_json(fig)
