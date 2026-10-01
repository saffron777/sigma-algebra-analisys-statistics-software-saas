"""A small Markdown subset → report blocks (notebook text cells and AI interpretations).

Supported: ``#``..``####`` headings, paragraphs, ``-``/``*``/``1.`` lists, fenced code blocks, ``> `` quotes (as
paragraphs), pipe tables, horizontal rules (ignored), and inline ``**bold**`` / ``*italic*`` / `` `code` `` which are
reduced to plain text for blocks and kept as markers for ``inline_runs``. Nothing here executes or interprets HTML.
"""

from __future__ import annotations

import re
from typing import Any

_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_INLINE = re.compile(r"(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")


def plain(text: str) -> str:
    """Inline Markdown reduced to plain text."""
    text = _LINK.sub(r"\1 (\2)", text)
    text = _INLINE.sub(lambda m: m.group(0).strip("*_`"), text)
    return _HTML_TAG.sub("", text).strip()


def inline_runs(text: str) -> list[tuple[str, str]]:
    """Split a paragraph into ``(style, text)`` runs; style is ``""``, ``"b"``, ``"i"`` or ``"c"`` (code)."""
    text = _HTML_TAG.sub("", _LINK.sub(r"\1 (\2)", text))
    runs: list[tuple[str, str]] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            runs.append(("", text[position : match.start()]))
        token = match.group(0)
        if token.startswith(("**", "__")):
            runs.append(("b", token[2:-2]))
        elif token.startswith("`"):
            runs.append(("c", token[1:-1]))
        else:
            runs.append(("i", token[1:-1]))
        position = match.end()
    if position < len(text):
        runs.append(("", text[position:]))
    return runs or [("", text)]


def markdown_to_blocks(source: str, base_level: int = 2) -> list[dict[str, Any]]:
    """Turn Markdown into report blocks; headings are shifted so ``#`` becomes ``base_level``."""
    blocks: list[dict[str, Any]] = []
    lines = source.replace("\r\n", "\n").split("\n")
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append({"type": "text", "paragraphs": [" ".join(p.strip() for p in paragraph)], "markdown": True})
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
        elif stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1  # closing fence
            blocks.append({"type": "code", "source": "\n".join(code)})
        elif (heading := _HEADING.match(stripped)) is not None:
            flush_paragraph()
            level = min(len(heading.group(1)) - 1 + base_level, 4)
            blocks.append({"type": "heading", "text": plain(heading.group(2)), "level": level})
            index += 1
        elif _BULLET.match(line) or _ORDERED.match(line):
            flush_paragraph()
            ordered = bool(_ORDERED.match(line))
            pattern = _ORDERED if ordered else _BULLET
            items: list[str] = []
            while index < len(lines) and (m := pattern.match(lines[index])):
                items.append(m.group(1).strip())
                index += 1
            blocks.append({"type": "list", "ordered": ordered, "items": items})
        elif "|" in stripped and index + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[index + 1]):
            flush_paragraph()
            header = _split_row(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_split_row(lines[index]))
                index += 1
            width = len(header)
            blocks.append(
                {
                    "type": "table",
                    "title": None,
                    "columns": [plain(c) for c in header],
                    "rows": [[plain(c) for c in (row + [""] * width)[:width]] for row in rows],
                    "note": None,
                }
            )
        elif re.match(r"^\s*([-*_])\1{2,}\s*$", stripped):
            flush_paragraph()
            index += 1
        else:
            paragraph.append(stripped[1:].strip() if stripped.startswith(">") else stripped)
            index += 1
    flush_paragraph()
    return blocks


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
