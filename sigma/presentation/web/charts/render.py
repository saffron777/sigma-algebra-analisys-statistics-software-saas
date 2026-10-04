"""Draws the chart descriptions of ``domain/analysis/charts.py`` with Plotly, in Sigma's theme."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots

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
    fig = builder(fig, spec) or fig  # a builder may hand back another figure (subplots)
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
    for band in spec.get("bands") or []:
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
                fillcolor=_rgba("#ef4444", 0.16),
                name=band.get("name"),
                hoverinfo="skip",
            )
        )
    for index, series in enumerate(spec["series"]):
        x, y = np.asarray(series["x"]), np.asarray(series["y"])
        keep = thin(y)
        marker: dict[str, Any] = {
            "size": series.get("size", 6),
            "color": series.get("color", _colour(index)),
            "opacity": 0.75,
        }
        if series.get("sizes") is not None:
            marker.update(
                size=as_list(np.asarray(series["sizes"], dtype=float)[keep]), line={"width": 0.5, "color": "white"}
            )
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
    """Histograms on common bins (so groups are comparable), drawn as bars of known width."""
    groups = [
        (g.get("name"), v[np.isfinite(v)]) for g in spec["groups"] for v in [np.asarray(g["values"], dtype=float)]
    ]
    pooled = np.concatenate([v for _, v in groups]) if groups else np.array([])
    if pooled.size == 0:
        raise ValueError("a histogram needs values")
    edges = np.histogram_bin_edges(pooled, bins=spec.get("bins") or "auto")
    width, centres = float(edges[1] - edges[0]), (edges[:-1] + edges[1:]) / 2
    norm = spec.get("norm", "density")
    overlay = len(groups) > 1
    totals = []
    for index, (name, values) in enumerate(groups):
        counts, _ = np.histogram(values, bins=edges)
        total = int(counts.sum())
        totals.append(total)
        heights = {
            "density": counts / (total * width) if total else counts * 0.0,
            "count": counts,
            "percent": 100 * counts / total if total else counts * 0.0,
        }[norm]
        colour = _colour(index)
        fig.add_trace(
            go.Bar(
                x=as_list(centres),
                y=as_list(heights),
                width=width,
                name=name,
                opacity=0.6 if overlay else 1,
                marker={"color": _rgba(colour, 0.55 if not overlay else 0.7), "line": {"color": colour, "width": 1}},
                hovertemplate="%{x:.4g}: %{y:.4g}<extra>" + str(name or "") + "</extra>",
            )
        )
    scale = {"density": 1.0, "count": width * (totals[0] if totals else 1), "percent": 100 * width}[norm]
    for index, curve in enumerate(spec.get("curves") or []):
        fig.add_trace(
            go.Scatter(
                x=as_list(curve["x"]),
                y=as_list(np.asarray(curve["y"], dtype=float) * scale),
                mode="lines",
                name=curve.get("name"),
                line={"color": _colour(index + len(groups)), "width": 2.2},
            )
        )
    fig.update_layout(barmode="overlay", bargap=0.02, showlegend=overlay or bool(spec.get("curves")))


def _box(fig: go.Figure, spec: dict[str, Any]) -> None:
    for index, group in enumerate(spec["groups"]):
        values = np.asarray(group["values"], dtype=float)
        values = values[np.isfinite(values)]
        common = {
            "y": as_list(values[thin(values, 3000)]),
            "name": group.get("name"),
            "marker": {"color": _colour(index)},
        }
        if spec.get("strip"):
            fig.add_trace(
                go.Box(
                    **{**common, "marker": {"color": _colour(index), "size": 6, "opacity": 0.7}},
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.45,
                    fillcolor="rgba(0,0,0,0)",
                    line={"width": 0},
                    hoveron="points",
                )
            )
        elif spec.get("violin"):
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


def _pie(fig: go.Figure, spec: dict[str, Any]) -> None:
    fig.add_trace(
        go.Pie(
            labels=[str(label) for label in spec["labels"]],
            values=as_list(spec["values"]),
            hole=0.45 if spec.get("donut") else 0,
            marker={"colors": [_colour(i) for i in range(len(spec["labels"]))], "line": {"color": "white", "width": 1}},
            textinfo="label+percent",
            sort=False,
        )
    )
    fig.update_layout(showlegend=len(spec["labels"]) <= 12, margin={"l": 20, "r": 20, "t": 52, "b": 20})


def _interval(fig: go.Figure, spec: dict[str, Any]) -> None:
    for index, group in enumerate(spec["groups"]):
        colour = _colour(index)
        values = group.get("values")
        if values is not None:
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            fig.add_trace(
                go.Box(
                    y=as_list(values[thin(values, 1500)]),
                    x=[group["name"]] * len(values[thin(values, 1500)]),
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.5,
                    fillcolor="rgba(0,0,0,0)",
                    line={"width": 0},
                    marker={"color": _rgba(colour, 0.35), "size": 5},
                    hoveron="points",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[group["name"]],
                y=[group["mean"]],
                mode="markers",
                name=str(group["name"]),
                marker={"color": colour, "size": 11, "symbol": "diamond"},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [group["upper"] - group["mean"]],
                    "arrayminus": [group["mean"] - group["lower"]],
                    "color": colour,
                    "thickness": 2.2,
                    "width": 9,
                },
                showlegend=False,
            )
        )
    fig.update_layout(xaxis={"type": "category", "title": {"text": spec.get("x_title")}, "gridcolor": GRID})


def _panels(fig: go.Figure, spec: dict[str, Any]) -> go.Figure:
    panels = spec["panels"]
    columns = spec["columns"]
    rows = -(-len(panels) // columns)
    grid = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[str(panel.get("title") or "") for panel in panels],
        horizontal_spacing=0.08,
        vertical_spacing=min(0.16, 0.5 / rows),
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        builder = _BUILDERS.get(panel["kind"])
        if builder is None or panel["kind"] == "panels":
            raise ValueError(f"cannot draw a {panel['kind']!r} chart inside a panel")
        inner = new_figure()
        inner = builder(inner, panel) or inner
        for trace in inner.data:
            grid.add_trace(trace, row=row + 1, col=column + 1)
        grid.update_xaxes(title_text=panel.get("x_title"), row=row + 1, col=column + 1)
        grid.update_yaxes(title_text=panel.get("y_title"), row=row + 1, col=column + 1)
    for key in (
        "title",
        "height",
        "margin",
        "paper_bgcolor",
        "plot_bgcolor",
        "font",
        "colorway",
        "legend",
        "hoverlabel",
    ):
        grid.layout[key] = fig.layout[key]
    # each panel is headed by its own title, so a shared legend would only repeat the same entries
    grid.update_layout(showlegend=any(p.get("show_legend") for p in panels))
    grid.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    grid.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    return grid


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
        fig.add_trace(trace if isinstance(trace, BaseTraceType) else _trace(trace))
    if spec.get("layout"):
        fig.update_layout(**_plain(spec["layout"]))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)  # every axis (a splom has one per variable)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)


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
        "splom": go.Splom,
        "pie": go.Pie,
    }.get(kind)
    if cls is None:
        raise ValueError(f"unknown trace type {kind!r}")
    return cls(**{key: _plain(value) for key, value in definition.items() if key != "type"})


def _plain(value: Any) -> Any:
    """Arrays become lists so the figure carries plain JSON (plotly would otherwise ship binary typed arrays)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


_BUILDERS = {
    "line": _line,
    "scatter": _scatter,
    "bar": _bar,
    "hist": _hist,
    "box": _box,
    "heatmap": _heatmap,
    "stem": _stem,
    "pie": _pie,
    "interval": _interval,
    "panels": _panels,
    "raw": _raw,
}
