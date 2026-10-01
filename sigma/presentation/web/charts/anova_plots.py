"""Charts for the ANOVA module."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from sigma.domain.results import AnovaResult
from sigma.presentation.web.charts.common import PALETTE, as_list, figure_json, new_figure

MAX_POINTS_PER_GROUP = 400


def _points(values: np.ndarray) -> np.ndarray:
    if len(values) <= MAX_POINTS_PER_GROUP:
        return values
    return np.random.default_rng(0).choice(values, MAX_POINTS_PER_GROUP, replace=False)


def groups_figure(result: AnovaResult) -> dict[str, Any]:
    fig = new_figure(f"{result.response} por grupo", " × ".join(result.factors), result.response)
    for i, (group, values) in enumerate(result.plot_data["groups"].items()):
        fig.add_trace(
            go.Box(
                y=as_list(_points(values)),
                name=group,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                boxmean=True,
                marker={"color": PALETTE[i % len(PALETTE)], "size": 4, "opacity": 0.6},
                line={"width": 1.5},
            )
        )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category")
    return figure_json(fig)


def means_figure(result: AnovaResult) -> dict[str, Any]:
    confidence = f"{100 * (1 - result.alpha):g} %"
    fig = new_figure(
        f"Medias con intervalo de confianza del {confidence}", " × ".join(result.factors), f"Media de {result.response}"
    )
    groups = result.groups
    fig.add_trace(
        go.Scatter(
            x=[g.group for g in groups],
            y=[g.mean for g in groups],
            mode="markers",
            marker={"color": PALETTE[0], "size": 11},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [g.ci_high - g.mean for g in groups],
                "arrayminus": [g.mean - g.ci_low for g in groups],
                "color": PALETTE[0],
                "thickness": 2,
                "width": 6,
            },
            hovertemplate="%{x}<br>media = %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category")
    return figure_json(fig)


def interaction_figure(result: AnovaResult) -> dict[str, Any]:
    cells = result.plot_data["cells"]
    factor_a, factor_b = result.factors
    fig = new_figure(f"Gráfico de interacción: {factor_a} × {factor_b}", factor_a, f"Media de {result.response}")
    levels_b = sorted({c["b"] for c in cells})
    for i, level in enumerate(levels_b):
        subset = [c for c in cells if c["b"] == level]
        fig.add_trace(
            go.Scatter(
                x=[c["a"] for c in subset],
                y=[c["mean"] for c in subset],
                mode="lines+markers",
                name=f"{factor_b} = {level}",
                line={"color": PALETTE[i % len(PALETTE)], "width": 2.4},
                marker={"size": 9},
                error_y={"type": "data", "array": [c["sem"] for c in subset], "visible": True},
            )
        )
    fig.update_xaxes(type="category")
    return figure_json(fig)


def residual_qq_figure(result: AnovaResult) -> dict[str, Any]:
    residuals = result.plot_data["residuals"]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fig = new_figure("Q-Q de los residuos del ANOVA", "Cuantiles teóricos", "Cuantiles de los residuos")
    step = max(1, len(osm) // 1500)
    fig.add_trace(
        go.Scatter(
            x=as_list(osm[::step]),
            y=as_list(osr[::step]),
            mode="markers",
            marker={"color": PALETTE[0], "size": 6, "opacity": 0.75},
        )
    )
    ends = np.array([float(np.min(osm)), float(np.max(osm))])
    fig.add_trace(
        go.Scatter(x=as_list(ends), y=as_list(intercept + slope * ends), mode="lines", line={"color": "#ef4444"})
    )
    fig.update_layout(showlegend=False)
    return figure_json(fig)
