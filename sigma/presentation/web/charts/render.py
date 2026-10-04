"""Draws the chart descriptions of ``domain/analysis/charts.py`` with Plotly, in Sigma's theme."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from sigma.presentation.web.charts.common import GRID, PALETTE, as_list, figure_json, new_figure, thin


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _colour(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def render_chart(spec: dict[str, Any]) -> dict[str, Any]:
    """Plotly JSON for one chart description."""
    kind = spec["kind"]
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown chart kind {kind!r}")
    fig = new_figure(spec.get("title"), spec.get("x_title"), spec.get("y_title"), spec.get("height", 380))
    builder(fig, spec)
    return figure_json(fig)


def _guides(fig: go.Figure, spec: dict[str, Any]) -> None:
    for line in spec.get("hlines") or []:
        fig.add_hline(
            y=line["value"],
            line={"dash": line.get("dash", "dot"), "width": 1.2, "color": "rgba(128,128,128,0.8)"},
            annotation_text=line.get("label"),
            annotation_position="top left",
        )
    for line in spec.get("vlines") or []:
        fig.add_vline(
            x=line["value"],
            line={"dash": line.get("dash", "dot"), "width": 1.2, "color": "rgba(128,128,128,0.8)"},
            annotation_text=line.get("label"),
            annotation_position="top right",
        )


def _line(fig: go.Figure, spec: dict[str, Any]) -> None:
    for index, band in enumerate(spec.get("bands") or []):
        colour = _colour(index)
        x = as_list(band["x"])
        fig.add_trace(
            go.Scatter(
                x=x, y=as_list(band["upper"]), mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=as_list(band["lower"]),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(colour, 0.18),
                name=band.get("name"),
                hoverinfo="skip",
            )
        )
    for index, series in enumerate(spec["series"]):
        x, y = np.asarray(series["x"]), np.asarray(series["y"], dtype=float)
        keep = thin(y)
        fig.add_trace(
            go.Scatter(
                x=as_list(x[keep]),
                y=as_list(y[keep]),
                name=series.get("name"),
                mode="lines+markers" if series.get("markers") else "lines",
                line={
                    "color": series.get("color", _colour(index)),
                    "width": series.get("width", 2),
                    "dash": series.get("dash", "solid"),
                },
                marker={"size": 5},
            )
        )
    _guides(fig, spec)


def _scatter(fig: go.Figure, spec: dict[str, Any]) -> None:
    for index, series in enumerate(spec["series"]):
        x, y = np.asarray(series["x"]), np.asarray(series["y"])
        keep = thin(y)
        marker: dict[str, Any] = {
            "size": series.get("size", 6),
            "color": series.get("color", _colour(index)),
            "opacity": 0.75,
        }
        if series.get("color_values") is not None:
            marker.update(color=as_list(np.asarray(series["color_values"])[keep]), colorscale="Viridis", showscale=True)
        text = series.get("text")
        fig.add_trace(
            go.Scatter(
                x=as_list(x[keep]),
                y=as_list(y[keep]),
                mode="markers+text" if series.get("labels") else "markers",
                name=series.get("name"),
                marker=marker,
                text=as_list(np.asarray(text)[keep]) if text is not None else None,
                textposition="top center",
            )
        )
    for index, line in enumerate(spec.get("lines") or []):
        fig.add_trace(
            go.Scatter(
                x=as_list(line["x"]),
                y=as_list(line["y"]),
                mode="lines",
                name=line.get("name"),
                line={
                    "color": line.get("color", "#ef4444" if index == 0 else _colour(index + 3)),
                    "width": 2,
                    "dash": line.get("dash", "solid"),
                },
            )
        )
    _guides(fig, spec)


def _bar(fig: go.Figure, spec: dict[str, Any]) -> None:
    horizontal = spec.get("horizontal")
    errors = spec.get("errors")
    for index, series in enumerate(spec["series"]):
        values = as_list(series["y"])
        cats = as_list(spec["categories"])
        error = {"type": "data", "array": errors[index], "visible": True} if errors else None
        fig.add_trace(
            go.Bar(
                x=values if horizontal else cats,
                y=cats if horizontal else values,
                name=series.get("name"),
                orientation="h" if horizontal else "v",
                marker={"color": series.get("color", _colour(index))},
                **({"error_x" if horizontal else "error_y": error} if error else {}),
            )
        )
    fig.update_layout(barmode="stack" if spec.get("stacked") else "group", showlegend=len(spec["series"]) > 1)


def _hist(fig: go.Figure, spec: dict[str, Any]) -> None:
    values = np.asarray(spec["values"], dtype=float)
    values = values[np.isfinite(values)]
    bins = spec.get("bins")
    fig.add_trace(
        go.Histogram(
            x=as_list(values[thin(values, 20000)]),
            histnorm="probability density",
            nbinsx=bins or None,
            marker={"color": _rgba(PALETTE[0], 0.55), "line": {"color": PALETTE[0], "width": 1}},
            name="Datos",
        )
    )
    for index, curve in enumerate(spec.get("curves") or []):
        fig.add_trace(
            go.Scatter(
                x=as_list(curve["x"]),
                y=as_list(curve["y"]),
                mode="lines",
                name=curve.get("name"),
                line={"color": _colour(index + 1), "width": 2.2},
            )
        )
    fig.update_layout(bargap=0.03, showlegend=bool(spec.get("curves")))


def _box(fig: go.Figure, spec: dict[str, Any]) -> None:
    for index, group in enumerate(spec["groups"]):
        values = np.asarray(group["values"], dtype=float)
        values = values[np.isfinite(values)]
        common = {
            "y": as_list(values[thin(values, 3000)]),
            "name": group.get("name"),
            "marker": {"color": _colour(index)},
        }
        if spec.get("violin"):
            fig.add_trace(
                go.Violin(
                    **common, box_visible=True, meanline_visible=True, points="all" if spec.get("points") else False
                )
            )
        else:
            fig.add_trace(
                go.Box(**common, boxmean=True, boxpoints="all" if spec.get("points") else "outliers", jitter=0.4)
            )
    fig.update_layout(showlegend=False)


def _heatmap(fig: go.Figure, spec: dict[str, Any]) -> None:
    z = np.asarray(spec["z"], dtype=float)
    text = np.where(np.isfinite(z), np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)), "")
    extra: dict[str, Any] = {}
    if spec.get("midpoint") is not None:
        extra["zmid"] = spec["midpoint"]
    fig.add_trace(
        go.Heatmap(
            z=[[None if math.isnan(v) else v for v in row] for row in z.tolist()],
            x=as_list(spec["x"]),
            y=as_list(spec["y"]),
            colorscale="RdBu_r" if spec.get("midpoint") is not None else "Viridis",
            text=text.tolist() if spec.get("annotate") else None,
            texttemplate="%{text}" if spec.get("annotate") else None,
            hovertemplate="%{y} · %{x}: %{z:.4g}<extra></extra>",
            **extra,
        )
    )
    fig.update_layout(yaxis={"autorange": "reversed", "gridcolor": GRID})


def _stem(fig: go.Figure, spec: dict[str, Any]) -> None:
    x = np.asarray(spec["x"], dtype=float)
    values = np.asarray(spec["values"], dtype=float)
    band = spec.get("band")
    if band is not None:
        width = np.broadcast_to(np.asarray(band, dtype=float), x.shape)
        fig.add_trace(
            go.Scatter(
                x=as_list(x), y=as_list(width), mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=as_list(x),
                y=as_list(-width),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(PALETTE[1], 0.18),
                name="Banda de confianza",
                hoverinfo="skip",
            )
        )
    xs, ys = [], []
    for position, value in zip(x, values, strict=True):
        xs += [position, position, None]
        ys += [0.0, float(value), None]
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", line={"color": PALETTE[0], "width": 2}, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(x), y=as_list(values), mode="markers", marker={"color": PALETTE[0], "size": 7}, name="Valor"
        )
    )
    fig.add_hline(y=0, line={"color": "rgba(128,128,128,0.8)", "width": 1})
    fig.update_layout(showlegend=band is not None)


def _raw(fig: go.Figure, spec: dict[str, Any]) -> None:
    for trace in spec["data"]:
        fig.add_trace(trace if isinstance(trace, go.BaseTraceType) else _trace(trace))
    if spec.get("layout"):
        fig.update_layout(**spec["layout"])


def _trace(definition: dict[str, Any]) -> Any:
    kind = definition.get("type", "scatter")
    cls = {
        "scatter": go.Scatter,
        "bar": go.Bar,
        "heatmap": go.Heatmap,
        "contour": go.Contour,
        "surface": go.Surface,
        "histogram": go.Histogram,
        "box": go.Box,
        "violin": go.Violin,
        "scatter3d": go.Scatter3d,
    }.get(kind)
    if cls is None:
        raise ValueError(f"unknown trace type {kind!r}")
    return cls(**{key: value for key, value in definition.items() if key != "type"})


_BUILDERS = {
    "line": _line,
    "scatter": _scatter,
    "bar": _bar,
    "hist": _hist,
    "box": _box,
    "heatmap": _heatmap,
    "stem": _stem,
    "raw": _raw,
}
