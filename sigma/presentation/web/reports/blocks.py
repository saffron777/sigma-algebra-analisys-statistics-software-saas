"""Constructors for the generic report blocks rendered by ``static/js/report.js``.

A report is ``{"title", "subtitle", "blocks": [...]}``; block types: heading, tiles, table, text, alert, figure.
Table cells are strings/numbers or ``{"text", "tone"}`` with tone in strong|good|bad|muted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Cell = Any


def cell(text: Any, tone: str | None = None) -> Cell:
    return {"text": str(text), "tone": tone} if tone else text


def heading(text: str) -> dict[str, Any]:
    return {"type": "heading", "text": text}


def tiles(items: list[tuple[str, Any] | tuple[str, Any, str]]) -> dict[str, Any]:
    return {
        "type": "tiles",
        "items": [
            {"label": item[0], "value": str(item[1]), **({"hint": item[2]} if len(item) > 2 else {})} for item in items
        ],
    }


def table(title: str | None, columns: list[str], rows: list[list[Cell]], note: str | None = None) -> dict[str, Any]:
    return {"type": "table", "title": title, "columns": columns, "rows": rows, "note": note}


def text(*paragraphs: str) -> dict[str, Any]:
    return {"type": "text", "paragraphs": list(paragraphs)}


def alert(kind: str, message: str) -> dict[str, Any]:
    """kind: info | success | warning | danger."""
    return {"type": "alert", "kind": kind, "message": message}


def figure(index: int, caption: str | None = None) -> dict[str, Any]:
    """Places figure number ``index`` of the payload at this position of the report."""
    return {"type": "figure", "index": index, "caption": caption}


@dataclass
class ReportBuilder:
    """Collects blocks and Plotly figures; ``figure()`` returns the block referencing the new figure."""

    title: str
    subtitle: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *blocks: dict[str, Any] | None) -> ReportBuilder:
        self.blocks.extend(b for b in blocks if b)
        return self

    def add_figure(self, fig: dict[str, Any], caption: str | None = None) -> ReportBuilder:
        self.figures.append(fig)
        self.blocks.append(figure(len(self.figures) - 1, caption))
        return self

    def report(self) -> dict[str, Any]:
        return {"title": self.title, "subtitle": self.subtitle, "blocks": self.blocks}
