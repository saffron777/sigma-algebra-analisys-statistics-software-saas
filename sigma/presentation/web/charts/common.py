"""Shared Plotly helpers: palette, base layout and JSON conversion.

Figures use transparent backgrounds and no hard-coded text colours: the browser re-applies the
current light/dark theme when rendering (see ``static/js/plot.js``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from sigma.presentation.web.serializers import to_jsonable

PALETTE = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#64748b"]
REJECTION_FILL = "rgba(239,68,68,0.30)"
PVALUE_FILL = "rgba(79,70,229,0.45)"
PROBABILITY_FILL = "rgba(16,185,129,0.38)"
GRID = "rgba(128,128,128,0.22)"
MAX_POINTS = 5000


def as_list(values: Any) -> list:
    return np.asarray(values).tolist()


def new_figure(
    title: str | None = None, x_title: str | None = None, y_title: str | None = None, height: int = 380, **layout: Any
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 15}} if title else None,
        height=height,
        margin={"l": 60, "r": 20, "t": 52 if title else 20, "b": 55},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, 'Segoe UI', sans-serif", "size": 12.5},
        colorway=PALETTE,
        legend={"orientation": "h", "y": -0.22, "x": 0},
        hoverlabel={"font": {"size": 12}},
        xaxis={"title": {"text": x_title}, "gridcolor": GRID, "zerolinecolor": GRID, "automargin": True},
        yaxis={"title": {"text": y_title}, "gridcolor": GRID, "zerolinecolor": GRID, "automargin": True},
        **layout,
    )
    return fig


def figure_json(fig: go.Figure) -> dict[str, Any]:
    """Plain-JSON representation (lists, never typed-array blobs) understood by plotly.js."""
    return to_jsonable(fig.to_plotly_json())


def thin(values: np.ndarray, limit: int = MAX_POINTS) -> np.ndarray:
    """Evenly down-sample long arrays so scatter plots stay responsive."""
    if len(values) <= limit:
        return np.arange(len(values))
    return np.linspace(0, len(values) - 1, limit).astype(int)
