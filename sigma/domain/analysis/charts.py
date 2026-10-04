"""Chart descriptions (plain data) that the web layer draws with Plotly in Sigma's theme.

Each constructor returns a dict with a ``kind`` and the data; values may be numpy arrays. ``raw_chart`` is the escape
hatch for anything unusual: Plotly ``data`` traces and ``layout`` as plain dictionaries.
"""

from __future__ import annotations

from typing import Any


def _base(kind: str, title: str | None, x_title: str | None, y_title: str | None, height: int) -> dict[str, Any]:
    return {"kind": kind, "title": title, "x_title": x_title, "y_title": y_title, "height": height}


def line_chart(
    title: str | None,
    x_title: str | None,
    y_title: str | None,
    series: list[dict[str, Any]],
    *,
    bands: list[dict[str, Any]] | None = None,
    hlines: list[dict[str, Any]] | None = None,
    vlines: list[dict[str, Any]] | None = None,
    height: int = 380,
) -> dict[str, Any]:
    """``series``: ``{name, x, y, dash?, width?, markers?}``; ``bands``: ``{name, x, lower, upper}``;
    ``hlines``/``vlines``: ``{value, label?, dash?}``."""
    return {
        **_base("line", title, x_title, y_title, height),
        "series": series,
        "bands": bands or [],
        "hlines": hlines or [],
        "vlines": vlines or [],
    }


def scatter_chart(
    title: str | None,
    x_title: str | None,
    y_title: str | None,
    series: list[dict[str, Any]],
    *,
    lines: list[dict[str, Any]] | None = None,
    hlines: list[dict[str, Any]] | None = None,
    vlines: list[dict[str, Any]] | None = None,
    height: int = 400,
) -> dict[str, Any]:
    """``series``: ``{name, x, y, text?, size?, color?}``; ``lines`` are fitted curves ``{name, x, y, dash?}``."""
    return {
        **_base("scatter", title, x_title, y_title, height),
        "series": series,
        "lines": lines or [],
        "hlines": hlines or [],
        "vlines": vlines or [],
    }


def bar_chart(
    title: str | None,
    x_title: str | None,
    y_title: str | None,
    categories: list[Any],
    series: list[dict[str, Any]],
    *,
    stacked: bool = False,
    horizontal: bool = False,
    errors: list[list[float]] | None = None,
    height: int = 380,
) -> dict[str, Any]:
    """``series``: ``{name, y}`` per bar series; ``errors``: one ± list per series (same order)."""
    return {
        **_base("bar", title, x_title, y_title, height),
        "categories": categories,
        "series": series,
        "stacked": stacked,
        "horizontal": horizontal,
        "errors": errors,
    }


def hist_chart(
    title: str | None,
    x_title: str | None,
    values: Any,
    *,
    bins: int | None = None,
    curves: list[dict[str, Any]] | None = None,
    height: int = 380,
) -> dict[str, Any]:
    """A density histogram of ``values`` plus optional curves ``{name, x, y}`` on the same scale."""
    return {**_base("hist", title, x_title, "Densidad", height), "values": values, "bins": bins, "curves": curves or []}


def box_chart(
    title: str | None,
    y_title: str | None,
    groups: list[dict[str, Any]],
    *,
    violin: bool = False,
    points: bool = False,
    height: int = 400,
) -> dict[str, Any]:
    """``groups``: ``{name, values}``."""
    return {**_base("box", title, None, y_title, height), "groups": groups, "violin": violin, "points": points}


def heatmap_chart(
    title: str | None,
    z: Any,
    x: list[Any],
    y: list[Any],
    *,
    x_title: str | None = None,
    y_title: str | None = None,
    annotate: bool = True,
    midpoint: float | None = None,
    height: int = 420,
) -> dict[str, Any]:
    return {
        **_base("heatmap", title, x_title, y_title, height),
        "z": z,
        "x": x,
        "y": y,
        "annotate": annotate,
        "midpoint": midpoint,
    }


def stem_chart(
    title: str | None,
    x_title: str | None,
    y_title: str | None,
    x: Any,
    values: Any,
    *,
    band: float | list[float] | None = None,
    height: int = 320,
) -> dict[str, Any]:
    """Lollipop bars from zero (autocorrelations); ``band`` draws ± limits (a number or one per lag)."""
    return {**_base("stem", title, x_title, y_title, height), "x": x, "values": values, "band": band}


def raw_chart(
    data: list[dict[str, Any]], layout: dict[str, Any] | None = None, title: str | None = None, height: int = 400
) -> dict[str, Any]:
    return {**_base("raw", title, None, None, height), "data": data, "layout": layout or {}}
