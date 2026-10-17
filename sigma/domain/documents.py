"""The document model shared by every exporter (PDF, Word, Excel): pure data, no libraries.

A document is a list of sections; a section is a title plus *blocks*. Blocks are the ones the screen already renders
(``presentation/web/reports/blocks.py``) plus a few for notebooks and AI text:

    heading   {"text", "level"?}              tiles    {"items": [{"label", "value", "hint"?}]}
    table     {"title", "columns", "rows", "note"}   (cells: str | number | {"text", "tone"})
    text      {"paragraphs": [...]}           alert    {"kind", "message"}
    figure    {"index", "caption"}            -> ``section.images[index]`` (PNG bytes, drawn by the browser)
    list      {"items": [...], "ordered"}     code     {"source", "label"?}
    output    {"text", "kind"?}               pagebreak {}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sigma.domain.markdown_blocks import markdown_to_blocks

PAGE_SIZES = ("A4", "Letter", "Legal")
ORIENTATIONS = ("portrait", "landscape")
TONES = ("strong", "good", "bad", "muted")
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
DEFAULT_PRIMARY = "#4f46e5"
DEFAULT_SECONDARY = "#0f172a"

_NUMBER = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")


@dataclass
class Branding:
    """What makes a report look like the organization's: name, colours, footer, cover text and logo."""

    organization: str = ""
    primary_color: str = DEFAULT_PRIMARY
    secondary_color: str = DEFAULT_SECONDARY
    footer_text: str = ""
    cover_text: str = ""
    logo_png: bytes | None = None


@dataclass
class LayoutOptions:
    page_size: str = "A4"
    orientation: str = "portrait"
    margin_mm: int = 18
    cover: bool = True
    toc: bool = True


@dataclass
class ReportSection:
    title: str
    subtitle: str | None = None
    meta: list[str] = field(default_factory=list)  # e.g. «Dataset: ventas», «Guardado: 2026-10-01»
    blocks: list[dict[str, Any]] = field(default_factory=list)
    images: list[bytes | None] = field(default_factory=list)  # PNG per ``figure`` index (None = not available)


@dataclass
class DataTable:
    """A whole table that only the Excel export carries (e.g. every row of a dataset)."""

    name: str
    columns: list[str]
    rows: list[list[Any]]


@dataclass
class ReportDocument:
    title: str
    sections: list[ReportSection]
    subtitle: str | None = None
    author: str = ""
    created_at: str = ""  # ISO date shown on the cover
    branding: Branding = field(default_factory=Branding)
    layout: LayoutOptions = field(default_factory=LayoutOptions)
    data_tables: list[DataTable] = field(default_factory=list)


# ------------------------------------------------------------------ cells
def cell_text(value: Any) -> str:
    """The text of a table cell (``None`` shows as an em dash, like on screen)."""
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return "—" if value is None else str(value)


def cell_tone(value: Any) -> str | None:
    tone = value.get("tone") if isinstance(value, dict) else None
    return tone if tone in TONES else None


def cell_number(value: Any) -> float | int | None:
    """The numeric value of a cell when it is a plain number (``"12.5"``, ``3``); formatted text stays text."""
    if isinstance(value, dict):
        value = value.get("text")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and _NUMBER.match(value.strip()):
        number = float(value)
        return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number
    return None


def is_numeric_text(text: str) -> bool:
    """Numbers, intervals, percentages and dashes are right-aligned (same rule as the screen)."""
    return bool(re.match(r"^[−+\-<>±≈ ]*[\d.,]+(e[+-]?\d+)?( ?%)?$|^[—∞−]+$|^\[.+\]$", text.strip()))


def normalize_color(value: str | None, fallback: str) -> str:
    return value.lower() if value and HEX_COLOR.match(value) else fallback


def safe_filename(title: str, extension: str, date: str) -> str:
    """«Sigma Algebra - <título> - AAAA-MM-DD.<ext>» with nothing that a file system or a header could object to."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", title)
    cleaned = " ".join(cleaned.split())[:80].strip(" .") or "Informe"
    return f"Sigma Algebra - {cleaned} - {date}.{extension}"


# ------------------------------------------------------------------ sanitising blocks that come from the browser
ALLOWED_BLOCKS = (
    "heading",
    "tiles",
    "table",
    "text",
    "alert",
    "figure",
    "list",
    "code",
    "output",
    "pagebreak",
    "markdown",
)
ALERT_KINDS = ("info", "success", "warning", "danger")
MAX_BLOCKS_PER_SECTION = 3000
MAX_TABLE_ROWS = 5000
MAX_TABLE_COLUMNS = 60
MAX_CODE_CHARS = 200_000


def _str(value: Any, limit: int = 4000) -> str:
    return "" if value is None else str(value)[:limit]


def _cell(value: Any) -> Any:
    if isinstance(value, dict):
        tone = value.get("tone")
        return {"text": _str(value.get("text"), 2000), **({"tone": tone} if tone in TONES else {})}
    if isinstance(value, bool) or value is None:
        return None if value is None else str(value)
    if isinstance(value, (int, float)):
        return value
    return _str(value, 2000)


def clean_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Keep only well-formed, known blocks with bounded sizes (the input comes from the browser, so it is untrusted)."""
    cleaned: list[dict[str, Any]] = []
    for block in (blocks if isinstance(blocks, list) else [])[:MAX_BLOCKS_PER_SECTION]:
        if not isinstance(block, dict) or block.get("type") not in ALLOWED_BLOCKS:
            continue
        kind = block["type"]
        if kind == "heading":
            level = block.get("level")
            cleaned.append(
                {"type": kind, "text": _str(block.get("text"), 300), "level": level if level in (2, 3, 4) else 2}
            )
        elif kind == "tiles":
            items = [i for i in (block.get("items") or []) if isinstance(i, dict)][:60]
            cleaned.append(
                {
                    "type": kind,
                    "items": [
                        {
                            "label": _str(i.get("label"), 200),
                            "value": _str(i.get("value"), 400),
                            **({"hint": _str(i["hint"], 300)} if i.get("hint") else {}),
                        }
                        for i in items
                    ],
                }
            )
        elif kind == "table":
            columns = [_str(c, 200) for c in (block.get("columns") or [])][:MAX_TABLE_COLUMNS]
            rows = [r for r in (block.get("rows") or []) if isinstance(r, list)][:MAX_TABLE_ROWS]
            cleaned.append(
                {
                    "type": kind,
                    "title": _str(block["title"], 300) if block.get("title") else None,
                    "columns": columns,
                    "rows": [[_cell(v) for v in row[: len(columns)]] for row in rows],
                    "note": _str(block["note"], 1000) if block.get("note") else None,
                }
            )
        elif kind == "text":
            paragraphs = [_str(p, 20000) for p in (block.get("paragraphs") or [])][:500]
            cleaned.append(
                {"type": kind, "paragraphs": paragraphs, **({"markdown": True} if block.get("markdown") else {})}
            )
        elif kind == "alert":
            cleaned.append(
                {
                    "type": kind,
                    "kind": block.get("kind") if block.get("kind") in ALERT_KINDS else "info",
                    "message": _str(block.get("message"), 4000),
                }
            )
        elif kind == "figure":
            index = block.get("index")
            cleaned.append(
                {
                    "type": kind,
                    "index": index if isinstance(index, int) and not isinstance(index, bool) and index >= 0 else 0,
                    "caption": _str(block["caption"], 400) if block.get("caption") else None,
                }
            )
        elif kind == "list":
            cleaned.append(
                {
                    "type": kind,
                    "ordered": bool(block.get("ordered")),
                    "items": [_str(i, 1000) for i in (block.get("items") or [])][:500],
                }
            )
        elif kind == "markdown":  # Markdown text (an AI interpretation) → ordinary blocks
            cleaned.extend(clean_blocks(markdown_to_blocks(_str(block.get("text"), 20_000), base_level=3)))
        elif kind in ("code", "output"):
            field = "source" if kind == "code" else "text"
            cleaned.append(
                {
                    "type": kind,
                    field: _str(block.get(field), MAX_CODE_CHARS),
                    **({"label": _str(block["label"], 80)} if block.get("label") else {}),
                }
            )
        else:  # pagebreak
            cleaned.append({"type": kind})
    return cleaned
