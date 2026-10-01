"""A notebook as a report section: text cells, code, and the outputs the kernel produced (text, tables, charts)."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from sigma.domain.documents import ReportSection
from sigma.domain.markdown_blocks import markdown_to_blocks
from sigma.domain.notebooks import NotebookDocument

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
MAX_OUTPUT_CHARS = 20_000
MAX_TABLE_ROWS = 400


@dataclass(frozen=True)
class NotebookReportOptions:
    include_text: bool = True
    include_code: bool = True
    include_outputs: bool = True
    number_cells: bool = False
    break_before_h1: bool = False  # a page break before every «#» heading of a text cell
    break_before_h2: bool = False  # …and before every «##» heading


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n… (salida truncada)"


class _TableParser(HTMLParser):
    """Reads the first ``<table>`` of a pandas ``DataFrame._repr_html_`` into columns and rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.header: list[str] = []
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_head = False
        self._depth = 0
        self._done = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._done:
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._in_head:
                self.header = self._row
            else:
                self.rows.append(self._row)
            self._row = None
        elif tag == "thead":
            self._in_head = False
        elif tag == "table":
            self._depth -= 1
            if self._depth <= 0:
                self._done = True

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._done:
            self._cell.append(data)


def html_table(html: str) -> dict[str, Any] | None:
    """A ``table`` block from an HTML table, or ``None`` when there is no usable one."""
    if "<table" not in html.lower():
        return None
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML must never break an export
        return None
    rows = parser.rows
    header = parser.header
    if not header and rows:
        header, rows = rows[0], rows[1:]
    if not header:
        return None
    width = len(header)
    notes = None
    if len(rows) > MAX_TABLE_ROWS:
        notes = f"Se muestran las primeras {MAX_TABLE_ROWS} filas de {len(rows)}."
        rows = rows[:MAX_TABLE_ROWS]
    return {
        "type": "table",
        "title": None,
        "columns": [c or " " for c in header],
        "rows": [(row + [""] * width)[:width] for row in rows],
        "note": notes,
    }


def _output_blocks(output: dict[str, Any], images: list[bytes | None]) -> list[dict[str, Any]]:
    kind = output.get("output_type")
    if kind == "stream":
        text = strip_ansi(_joined(output.get("text")))
        return [{"type": "output", "text": _truncate(text.rstrip("\n"))}] if text.strip() else []
    if kind == "error":
        trace = (
            strip_ansi("\n".join(output.get("traceback") or []))
            or f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        )
        return [{"type": "output", "text": _truncate(trace), "kind": "error"}]
    if kind in ("execute_result", "display_data"):
        data = output.get("data") or {}
        if "image/png" in data:
            try:
                png = base64.b64decode(_joined(data["image/png"]), validate=False)
            except (binascii.Error, ValueError):
                png = b""
            if png.startswith(b"\x89PNG"):
                images.append(png)
                return [{"type": "figure", "index": len(images) - 1, "caption": None}]
        if "text/html" in data:
            table = html_table(_joined(data["text/html"]))
            if table:
                return [table]
        if "text/plain" in data:
            text = strip_ansi(_joined(data["text/plain"]))
            return [{"type": "output", "text": _truncate(text)}] if text.strip() else []
    return []


def notebook_section(
    document: NotebookDocument, options: NotebookReportOptions = NotebookReportOptions()
) -> ReportSection:
    blocks: list[dict[str, Any]] = []
    images: list[bytes | None] = []
    counter = 0
    for cell in document.cells:
        kind = cell.get("cell_type")
        source = _joined(cell.get("source"))
        sigma_meta = (cell.get("metadata") or {}).get("sigma") or {}
        if sigma_meta.get("page_break_before") and blocks:
            blocks.append({"type": "pagebreak"})
        if kind == "markdown":
            if not options.include_text or not source.strip():
                continue
            for block in markdown_to_blocks(source, base_level=2):
                if (
                    block["type"] == "heading"
                    and blocks
                    and (
                        (block["level"] == 2 and options.break_before_h1)
                        or (block["level"] == 3 and options.break_before_h2)
                    )
                ):
                    blocks.append({"type": "pagebreak"})
                blocks.append(block)
        elif kind == "code":
            counter += 1
            label = f"Celda [{cell.get('execution_count') or ' '}]" if options.number_cells else None
            if options.include_code and source.strip():
                blocks.append({"type": "code", "source": source.rstrip("\n"), **({"label": label} if label else {})})
            if options.include_outputs:
                for output in cell.get("outputs") or []:
                    blocks.extend(_output_blocks(output, images))
    return ReportSection(
        title=document.name,
        subtitle=f"Notebook · {len(document.cells)} celdas",
        meta=[f"Actualizado: {document.updated_at[:10]}"],
        blocks=blocks,
        images=images,
    )
