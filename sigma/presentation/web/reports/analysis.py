"""Turns an ``AnalysisOutput`` (blocks as plain data) into the report the pages show, export and save."""

from __future__ import annotations

from typing import Any

from sigma.domain.analysis.output import AnalysisOutput, Cell, P
from sigma.presentation.web.charts.render import render_chart
from sigma.presentation.web.reports import blocks as b
from sigma.presentation.web.reports.fmt import num, pval


def _text(value: Any) -> Any:
    """A table cell or tile value, formatted like the rest of the reports."""
    if isinstance(value, Cell):
        return b.cell(_text(value.value), value.tone)
    if isinstance(value, P):
        return pval(float(value))
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) or hasattr(value, "dtype"):
        return num(value)
    if value is None:
        return "—"
    return value if isinstance(value, str | dict) else str(value)


def build_report(output: AnalysisOutput, subtitle: str | None = None) -> b.ReportBuilder:
    builder = b.ReportBuilder(output.title, subtitle or output.subtitle)
    for block in output.blocks:
        kind = block["type"]
        if kind == "heading":
            builder.add(b.heading(block["text"]))
        elif kind == "tiles":
            builder.add(b.tiles([(item[0], _plain(_text(item[1])), *item[2:]) for item in block["items"]]))
        elif kind == "table":
            rows = [[_text(value) for value in row] for row in block["rows"]]
            builder.add(b.table(block["title"], block["columns"], rows, block["note"]))
        elif kind == "text":
            builder.add(b.text(*block["paragraphs"]))
        elif kind == "alert":
            builder.add(b.alert(block["kind"], block["message"]))
        elif kind == "chart":
            builder.add_figure(render_chart(block["spec"]), block.get("caption"))
        else:
            raise ValueError(f"unknown block type {kind!r}")
    return builder


def _plain(value: Any) -> str:
    return value["text"] if isinstance(value, dict) else str(value)
