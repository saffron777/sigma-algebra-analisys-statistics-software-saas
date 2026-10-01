"""Charts for the survey / market-research module."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from sigma.domain.results import (
    CrosstabResult,
    LikertResult,
    SampleSizeResult,
    StratumAllocation,
    SurveyEstimateResult,
)
from sigma.presentation.web.charts.common import PALETTE, figure_json, new_figure

LIKERT_SCALE = ["#dc2626", "#f97316", "#d1d5db", "#84cc16", "#16a34a"]


def sample_size_curve_figure(result: SampleSizeResult) -> dict[str, Any]:
    proportion = result.kind == "proportion"
    scale = 100 if proportion else 1
    x_title = "Margen de error (%)" if proportion else "Error máximo admitido"
    fig = new_figure("Tamaño de muestra según el margen de error", x_title, "Tamaño de muestra (n)")
    fig.add_trace(
        go.Scatter(
            x=[m * scale for m, _ in result.curve],
            y=[n for _, n in result.curve],
            mode="lines+markers",
            line={"color": PALETTE[0], "width": 2.4},
            marker={"size": 6},
            name="n requerido",
            hovertemplate="margen %{x:.3g}<br>n = %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.margin_of_error * scale],
            y=[result.n_final],
            mode="markers+text",
            marker={"color": "#ef4444", "size": 13},
            text=[f"n = {result.n_final}"],
            textposition="top right",
            name="Su diseño",
        )
    )
    return figure_json(fig)


def allocation_figure(allocation: list[StratumAllocation], method: str | None) -> dict[str, Any]:
    labels = {"proportional": "proporcional", "equal": "igual", "neyman": "de Neyman"}
    fig = new_figure(f"Asignación {labels.get(method or '', '')} por estrato", "Estrato", "Unidades")
    fig.add_trace(
        go.Bar(
            x=[a.stratum for a in allocation],
            y=[a.population for a in allocation],
            name="Población (N_h)",
            marker={"color": "#a5b4fc"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=[a.stratum for a in allocation],
            y=[a.sample_size for a in allocation],
            name="Muestra (n_h)",
            marker={"color": PALETTE[0]},
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    return figure_json(fig)


def estimates_figure(result: SurveyEstimateResult) -> dict[str, Any]:
    proportion = result.kind == "proportion"
    scale = 100 if proportion else 1
    first_method = result.rows[0].method
    rows = [r for r in result.rows if r.method == first_method]
    title = f"Estimaciones con IC del {100 * result.confidence:g} % ({first_method})"
    fig = new_figure(
        title, "Porcentaje (%)" if proportion else result.variable, None, height=max(300, 70 * len(rows) + 120)
    )
    fig.add_trace(
        go.Bar(
            y=[r.label for r in rows],
            x=[r.estimate * scale for r in rows],
            orientation="h",
            marker={"color": PALETTE[0]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [(r.ci_high - r.estimate) * scale for r in rows],
                "arrayminus": [(r.estimate - r.ci_low) * scale for r in rows],
                "color": "#111827",
                "thickness": 2,
            },
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def crosstab_figure(result: CrosstabResult) -> dict[str, Any]:
    fig = new_figure(
        f"{result.row_variable} según {result.col_variable} (% por fila)",
        "% de la fila",
        None,
        height=max(320, 56 * len(result.row_labels) + 140),
    )
    for j, column in enumerate(result.col_labels):
        fig.add_trace(
            go.Bar(
                y=result.row_labels,
                x=[row[j] for row in result.row_percent],
                name=column,
                orientation="h",
                marker={"color": PALETTE[j % len(PALETTE)]},
                hovertemplate="%{y} · " + column + ": %{x:.1f} %<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(type="category", autorange="reversed")
    return figure_json(fig)


def likert_figure(result: LikertResult) -> dict[str, Any]:
    fig = new_figure(
        "Distribución de respuestas por ítem", "% de respuestas", None, height=max(300, 64 * len(result.items) + 150)
    )
    colors = _likert_colors(len(result.levels))
    for j, level in enumerate(result.levels):
        fig.add_trace(
            go.Bar(
                y=result.items,
                x=[row[j] for row in result.percents],
                name=level,
                orientation="h",
                marker={"color": colors[j]},
                hovertemplate="%{y} · " + level + ": %{x:.1f} %<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(type="category", autorange="reversed")
    return figure_json(fig)


def _likert_colors(n: int) -> list[str]:
    if n == len(LIKERT_SCALE):
        return LIKERT_SCALE
    return [LIKERT_SCALE[round(i * (len(LIKERT_SCALE) - 1) / max(n - 1, 1))] for i in range(n)]
