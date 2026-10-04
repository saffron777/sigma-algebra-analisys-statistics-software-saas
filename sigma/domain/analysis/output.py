"""What an analysis returns: report blocks as plain data, tables it can leave as datasets, and its key numbers.

Domain code never touches Plotly or HTML. It builds an :class:`Output` (headings, tiles, tables, notes, charts) and the
web layer turns the blocks into the report the page shows, exports and saves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class P(float):
    """A p-value: shown as ``0.0123`` or ``< 0.0001``."""


@dataclass(frozen=True)
class Cell:
    """A table cell with emphasis: ``tone`` is strong | good | bad | muted."""

    value: Any
    tone: str | None = None


def strong(value: Any) -> Cell:
    return Cell(value, "strong")


def good(value: Any) -> Cell:
    return Cell(value, "good")


def bad(value: Any) -> Cell:
    return Cell(value, "bad")


def muted(value: Any) -> Cell:
    return Cell(value, "muted")


@dataclass
class DatasetOffer:
    label: str
    frame: pd.DataFrame


@dataclass
class AnalysisOutput:
    title: str
    subtitle: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    datasets: dict[str, DatasetOffer] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)  # the key numbers (saved with the result, read by the AI)


class Output:
    """Fluent builder: ``Output("ACF").tiles([...]).table(...).chart(line_chart(...))``."""

    def __init__(self, title: str, subtitle: str | None = None) -> None:
        self._out = AnalysisOutput(title, subtitle)

    def heading(self, text: str) -> Output:
        self._out.blocks.append({"type": "heading", "text": text})
        return self

    def tiles(self, items: list[tuple]) -> Output:
        """``(label, value)`` or ``(label, value, hint)``; floats are formatted by the report."""
        self._out.blocks.append({"type": "tiles", "items": items})
        return self

    def table(self, title: str | None, columns: list[str], rows: list[list[Any]], note: str | None = None) -> Output:
        self._out.blocks.append({"type": "table", "title": title, "columns": columns, "rows": rows, "note": note})
        return self

    def text(self, *paragraphs: str) -> Output:
        self._out.blocks.append({"type": "text", "paragraphs": list(paragraphs)})
        return self

    def alert(self, kind: str, message: str) -> Output:
        """kind: info | success | warning | danger."""
        self._out.blocks.append({"type": "alert", "kind": kind, "message": message})
        return self

    def chart(self, spec: dict[str, Any], caption: str | None = None) -> Output:
        self._out.blocks.append({"type": "chart", "spec": spec, "caption": caption})
        return self

    def dataset(self, key: str, label: str, frame: pd.DataFrame) -> Output:
        self._out.datasets[key] = DatasetOffer(label, frame)
        return self

    def set(self, **values: Any) -> Output:
        self._out.result.update(values)
        return self

    def build(self) -> AnalysisOutput:
        return self._out
