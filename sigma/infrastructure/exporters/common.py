"""Helpers shared by the three exporters."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

from PIL import Image, UnidentifiedImageError

from sigma.domain.documents import ReportDocument, ReportSection, normalize_color
from sigma.domain.markdown_blocks import inline_runs

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = normalize_color(color, "#000000").lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def tint(color: str, amount: float) -> str:
    """Mix ``color`` with white (``amount`` 0 = unchanged, 1 = white); used for soft backgrounds."""
    r, g, b = hex_to_rgb(color)
    mixed = [round(c + (255 - c) * amount) for c in (r, g, b)]
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def image_size(png: bytes) -> tuple[int, int] | None:
    """Width and height of a PNG, or ``None`` when the bytes are not a readable image."""
    try:
        with Image.open(io.BytesIO(png)) as image:
            return image.size
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def flatten_png(png: bytes) -> bytes:
    """Re-encode as an opaque PNG (some viewers draw transparent figures on black)."""
    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, "white")
                background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
                image = background
            out = io.BytesIO()
            image.convert("RGB").save(out, format="PNG", optimize=True)
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return png


def iter_blocks(section: ReportSection) -> Iterator[dict[str, Any]]:
    yield from section.blocks


def figure_image(section: ReportSection, block: dict[str, Any]) -> bytes | None:
    index = block.get("index")
    if isinstance(index, int) and 0 <= index < len(section.images):
        return section.images[index]
    return None


def paragraph_runs(block: dict[str, Any], paragraph: str) -> list[tuple[str, str]]:
    """Runs of a text paragraph: Markdown blocks keep bold/italic/code, plain blocks are a single run."""
    return inline_runs(paragraph) if block.get("markdown") else [("", paragraph)]


def document_date(document: ReportDocument) -> str:
    return (document.created_at or "")[:10]


def table_blocks(document: ReportDocument) -> Iterator[tuple[ReportSection, dict[str, Any]]]:
    for section in document.sections:
        for block in section.blocks:
            if block.get("type") == "table":
                yield section, block
