"""PDF export (reportlab): cover, contents page, PDF bookmarks, branded header/footer and «Página n de N».

Text is set in DejaVu (shipped with matplotlib) so Greek letters, ², ≤, ≠, → … print correctly. Every string that
reaches a ``Paragraph`` is XML-escaped first: results and imported data are never interpreted as markup.
"""

from __future__ import annotations

import io
import os
import re
from typing import Any
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape, legal, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from sigma.domain.documents import (
    ReportDocument,
    ReportSection,
    cell_text,
    cell_tone,
    is_numeric_text,
    normalize_color,
)
from sigma.infrastructure.exporters.common import (
    document_date,
    figure_image,
    flatten_png,
    image_size,
    paragraph_runs,
    tint,
)

_PAGES = {"A4": A4, "Letter": letter, "Legal": legal}
_FONT, _BOLD, _ITALIC, _BOLDITALIC, _MONO = (
    "DejaVuSans",
    "DejaVuSans-Bold",
    "DejaVuSans-Oblique",
    "DejaVuSans-BoldOblique",
    "DejaVuSansMono",
)
_TONE = {"strong": "#000000", "good": "#15803d", "bad": "#b91c1c", "muted": "#64748b"}
_ALERT = {
    "info": ("#e0f2fe", "#0284c7"),
    "success": ("#dcfce7", "#16a34a"),
    "warning": ("#fef3c7", "#d97706"),
    "danger": ("#fee2e2", "#dc2626"),
}
_GRAY = "#64748b"
_LONG_TOKEN = re.compile(r"(\S{26})(?=\S)")
_fonts_ready = False


def _register_fonts() -> None:
    global _fonts_ready
    if _fonts_ready:
        return
    folder = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for name, file in (
        (_FONT, "DejaVuSans.ttf"),
        (_BOLD, "DejaVuSans-Bold.ttf"),
        (_ITALIC, "DejaVuSans-Oblique.ttf"),
        (_BOLDITALIC, "DejaVuSans-BoldOblique.ttf"),
        (_MONO, "DejaVuSansMono.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, os.path.join(folder, file)))
    pdfmetrics.registerFontFamily(_FONT, normal=_FONT, bold=_BOLD, italic=_ITALIC, boldItalic=_BOLDITALIC)
    _fonts_ready = True


def _markup(text: Any) -> str:
    """XML-safe text for a Paragraph; absurdly long unbroken tokens get break opportunities."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))
    return escape(_LONG_TOKEN.sub(r"\1 ", cleaned))


class _Canvas(pdf_canvas.Canvas):
    """Two-pass canvas: knows the total page count, so every page can say «Página n de N»."""

    def __init__(self, *args: Any, settings: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._settings = settings
        self._states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._states)
        for state in self._states:
            self.__dict__.update(state)
            if not (self._settings["cover"] and self._pageNumber == 1):
                self.setFont(_FONT, 7.5)
                self.setFillColor(colors.HexColor(_GRAY))
                self.drawRightString(
                    self._settings["right"], self._settings["footer_y"], f"Página {self._pageNumber} de {total}"
                )
            super().showPage()
        super().save()


class _Template(BaseDocTemplate):
    def afterFlowable(self, flowable) -> None:
        entry = getattr(flowable, "_sigma_entry", None)
        if not entry:
            return
        level, title, key = entry
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level, closed=level > 0)
        if level == 0:
            self.notify("TOCEntry", (0, title, self.page, key))


class PdfExporter:
    format = "pdf"
    content_type = "application/pdf"
    extension = "pdf"

    def export(self, document: ReportDocument) -> bytes:
        _register_fonts()
        brand = document.branding
        self._primary = normalize_color(brand.primary_color, "#4f46e5")
        self._secondary = normalize_color(brand.secondary_color, "#0f172a")
        self._brand = brand
        self._document = document
        page = _PAGES.get(document.layout.page_size, A4)
        if document.layout.orientation == "landscape":
            page = landscape(page)
        margin = max(10, min(40, document.layout.margin_mm)) * mm
        self._page_width, self._page_height = page
        self._width = self._page_width - 2 * margin
        self._styles = self._make_styles()
        self._bookmarks = 0

        buffer = io.BytesIO()
        doc = _Template(
            buffer,
            pagesize=page,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
            title=document.title,
            author=document.author or brand.organization or "Sigma",
            subject=document.subtitle or "",
            creator="Sigma",
        )
        header_space, footer_space = 9 * mm, 7 * mm
        body = Frame(
            margin,
            margin + footer_space,
            self._width,
            self._page_height - 2 * margin - header_space - footer_space,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        cover = Frame(
            margin, margin, self._width, self._page_height - 2 * margin, id="cover", leftPadding=0, rightPadding=0
        )
        footer_y = margin - 1 * mm
        self._margin, self._footer_y = margin, footer_y
        templates = [
            PageTemplate(id="cover", frames=[cover], onPage=lambda c, d: None),
            PageTemplate(id="body", frames=[body], onPage=self._decorate_page),
        ]
        doc.addPageTemplates(
            templates if document.layout.cover else templates[::-1]
        )  # the first one starts the document
        story: list[Any] = []
        if document.layout.cover:
            story += self._cover(document)
            story.append(NextPageTemplate("body"))
            story.append(PageBreak())
        if document.layout.toc and document.sections:
            story += self._toc(document)
        for position, section in enumerate(document.sections):
            if position:
                story.append(PageBreak())
            story += self._section(section, position)
        settings = {"cover": document.layout.cover, "right": self._page_width - margin, "footer_y": footer_y}
        doc.multiBuild(story, canvasmaker=lambda *a, **kw: _Canvas(*a, settings=settings, **kw))
        return buffer.getvalue()

    # ------------------------------------------------------------------ page furniture
    def _decorate_page(self, canvas, doc) -> None:
        canvas.saveState()
        top = self._page_height - self._margin
        x0, x1 = self._margin, self._page_width - self._margin
        cursor = x0
        if self._brand.logo_png and (size := image_size(self._brand.logo_png)):
            height = 7 * mm
            width = height * size[0] / size[1]
            canvas.drawImage(
                ImageReader(io.BytesIO(flatten_png(self._brand.logo_png))),
                cursor,
                top - height,
                width=width,
                height=height,
            )
            cursor += width + 2.5 * mm
        if self._brand.organization:
            canvas.setFont(_BOLD, 8.5)
            canvas.setFillColor(colors.HexColor(self._primary))
            canvas.drawString(cursor, top - 5 * mm, self._brand.organization[:60])
        canvas.setFont(_FONT, 7.5)
        canvas.setFillColor(colors.HexColor(_GRAY))
        canvas.drawRightString(x1, top - 5 * mm, self._document.title[:80])
        canvas.setStrokeColor(colors.HexColor(self._primary))
        canvas.setLineWidth(0.8)
        canvas.line(x0, top - 8 * mm, x1, top - 8 * mm)
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.4)
        canvas.line(x0, self._footer_y + 4 * mm, x1, self._footer_y + 4 * mm)
        canvas.setFont(_FONT, 7.5)
        canvas.setFillColor(colors.HexColor(_GRAY))
        canvas.drawString(x0, self._footer_y, (self._brand.footer_text or "Generado con Sigma")[:140])
        canvas.showOutline()
        canvas.restoreState()

    def _make_styles(self) -> dict[str, ParagraphStyle]:
        base = ParagraphStyle("base", fontName=_FONT, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1e293b"))
        return {
            "body": base,
            "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=colors.HexColor(_GRAY)),
            "h1": ParagraphStyle(
                "h1",
                parent=base,
                fontName=_BOLD,
                fontSize=18,
                leading=22,
                textColor=colors.HexColor(self._primary),
                spaceAfter=4,
            ),
            "h2": ParagraphStyle(
                "h2",
                parent=base,
                fontName=_BOLD,
                fontSize=12.5,
                leading=16,
                textColor=colors.HexColor(self._secondary),
                spaceBefore=10,
                spaceAfter=4,
            ),
            "h3": ParagraphStyle(
                "h3",
                parent=base,
                fontName=_BOLD,
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor(self._secondary),
                spaceBefore=8,
                spaceAfter=3,
            ),
            "subtitle": ParagraphStyle(
                "subtitle",
                parent=base,
                fontName=_ITALIC,
                fontSize=10,
                textColor=colors.HexColor("#475569"),
                spaceAfter=2,
            ),
            "table_title": ParagraphStyle(
                "table_title", parent=base, fontName=_BOLD, fontSize=9.5, spaceBefore=6, spaceAfter=3
            ),
            "caption": ParagraphStyle(
                "caption",
                parent=base,
                fontName=_ITALIC,
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.HexColor(_GRAY),
            ),
            "code": ParagraphStyle(
                "code",
                parent=base,
                fontName=_MONO,
                fontSize=7.8,
                leading=10.2,
                backColor=colors.HexColor("#f1f5f9"),
                borderPadding=(4, 5, 4, 5),
                spaceBefore=4,
                spaceAfter=8,
                leftIndent=2,
            ),
            "output": ParagraphStyle(
                "output",
                parent=base,
                fontName=_MONO,
                fontSize=7.8,
                leading=10.2,
                backColor=colors.HexColor("#fafaf9"),
                borderColor=colors.HexColor("#e2e8f0"),
                borderWidth=0.4,
                borderPadding=(4, 5, 4, 5),
                spaceBefore=2,
                spaceAfter=8,
                leftIndent=2,
            ),
        }

    # ------------------------------------------------------------------ cover and contents
    def _cover(self, document: ReportDocument) -> list[Any]:
        styles = self._styles
        story: list[Any] = []
        if self._brand.logo_png and (size := image_size(self._brand.logo_png)):
            height = 26 * mm
            story.append(
                Image(
                    io.BytesIO(flatten_png(self._brand.logo_png)),
                    width=height * size[0] / size[1],
                    height=height,
                    hAlign="LEFT",
                )
            )
        story.append(Spacer(1, 55 * mm))
        story.append(
            Paragraph(
                _markup(document.title),
                ParagraphStyle("cover_title", parent=styles["h1"], fontSize=30, leading=36, spaceAfter=10),
            )
        )
        if document.subtitle:
            story.append(
                Paragraph(
                    _markup(document.subtitle),
                    ParagraphStyle(
                        "cover_sub",
                        parent=styles["body"],
                        fontSize=15,
                        leading=20,
                        textColor=colors.HexColor("#475569"),
                    ),
                )
            )
        story.append(Spacer(1, 8 * mm))
        story.append(
            HRFlowable(width="22%", thickness=3, color=colors.HexColor(self._primary), hAlign="LEFT", spaceAfter=10)
        )
        for line in (self._brand.organization, self._brand.cover_text):
            if line:
                story.append(
                    Paragraph(
                        _markup(line), ParagraphStyle("cover_line", parent=styles["body"], fontSize=12, leading=17)
                    )
                )
        meta = " · ".join(part for part in (document.author, document_date(document)) if part)
        if meta:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(_markup(meta), styles["small"]))
        if len(document.sections) > 1:
            story.append(Paragraph(f"{len(document.sections)} análisis", styles["small"]))
        return story

    def _toc(self, document: ReportDocument) -> list[Any]:
        toc = TableOfContents()
        toc.dotsMinLevel = 0
        toc.levelStyles = [
            ParagraphStyle(
                "toc0",
                parent=self._styles["body"],
                fontSize=10.5,
                leading=18,
                leftIndent=0,
                rightIndent=18,
                firstLineIndent=0,
            )
        ]
        title = Paragraph("Contenido", self._styles["h1"])
        return [title, Spacer(1, 4 * mm), toc, PageBreak()]

    # ------------------------------------------------------------------ sections
    def _entry(self, flowable, level: int, title: str):
        self._bookmarks += 1
        flowable._sigma_entry = (level, title[:120], f"bk{self._bookmarks}")
        return flowable

    def _section(self, section: ReportSection, position: int) -> list[Any]:
        styles = self._styles
        story: list[Any] = [self._entry(Paragraph(_markup(section.title), styles["h1"]), 0, section.title)]
        if section.subtitle:
            story.append(Paragraph(_markup(section.subtitle), styles["subtitle"]))
        if section.meta:
            story.append(Paragraph(_markup("  ·  ".join(section.meta)), styles["small"]))
        story.append(
            HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=8)
        )
        for block in section.blocks:
            handler = getattr(self, f"_block_{block.get('type')}", None)
            if handler:
                story += handler(section, block) or []
        return story

    def _block_heading(self, section, block) -> list[Any]:
        level = min(max(block.get("level") or 2, 2), 4)
        text = str(block.get("text", ""))
        paragraph = Paragraph(_markup(text), self._styles["h2" if level == 2 else "h3"])
        # a heading is never left alone at the bottom of a page (keepWithNext would push a whole long table away)
        return [CondPageBreak(32 * mm), self._entry(paragraph, 1, text) if level == 2 else paragraph]

    def _block_pagebreak(self, section, block) -> list[Any]:
        return [PageBreak()]

    def _runs_markup(self, block: dict[str, Any], text: str) -> str:
        parts = []
        for style, content in paragraph_runs(block, text):
            safe = _markup(content)
            if style == "b":
                safe = f"<b>{safe}</b>"
            elif style == "i":
                safe = f"<i>{safe}</i>"
            elif style == "c":
                safe = f'<font name="{_MONO}" size="8.5">{safe}</font>'
            parts.append(safe)
        return "".join(parts)

    def _block_text(self, section, block) -> list[Any]:
        return [
            Paragraph(self._runs_markup(block, str(t)), self._styles["body"]) for t in block.get("paragraphs", [])
        ] + [Spacer(1, 2 * mm)]

    def _block_list(self, section, block) -> list[Any]:
        ordered = bool(block.get("ordered"))
        out = []
        for number, item in enumerate(block.get("items", []), start=1):
            style = ParagraphStyle("li", parent=self._styles["body"], leftIndent=14, bulletIndent=2, spaceAfter=2)
            out.append(
                Paragraph(
                    self._runs_markup({"markdown": True}, str(item)), style, bulletText=f"{number}." if ordered else "•"
                )
            )
        return out + [Spacer(1, 2 * mm)]

    def _block_alert(self, section, block) -> list[Any]:
        fill, bar = _ALERT.get(block.get("kind"), _ALERT["info"])
        table = Table([[Paragraph(_markup(block.get("message", "")), self._styles["body"])]], colWidths=[self._width])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(bar)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return [table, Spacer(1, 3 * mm)]

    def _block_tiles(self, section, block) -> list[Any]:
        items = block.get("items", [])
        if not items:
            return []
        columns = min(len(items), 5 if self._width > 220 * mm else 4)
        cell_width = self._width / columns
        label = ParagraphStyle("tile_label", parent=self._styles["small"], fontSize=7)
        value = ParagraphStyle("tile_value", parent=self._styles["body"], fontName=_BOLD, fontSize=12.5, leading=15)
        cells = []
        for item in items:
            content = [
                Paragraph(_markup(item.get("label", "")), label),
                Paragraph(_markup(item.get("value", "")), value),
            ]
            if item.get("hint"):
                content.append(
                    Paragraph(_markup(item["hint"]), ParagraphStyle("tile_hint", parent=label, fontSize=6.5))
                )
            cells.append(content)
        rows = [
            cells[i : i + columns] + [""] * (columns - len(cells[i : i + columns]))
            for i in range(0, len(cells), columns)
        ]
        table = Table(rows, colWidths=[cell_width] * columns)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint(self._primary, 0.9))),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.white),
                    ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return [table, Spacer(1, 3 * mm)]

    def _block_table(self, section, block) -> list[Any]:
        columns = [str(c) for c in block.get("columns", [])]
        rows = block.get("rows", [])
        if not columns:
            return []
        size = 8.5 if len(columns) <= 5 else 7.5 if len(columns) <= 8 else 6.5
        head = ParagraphStyle(
            "th", parent=self._styles["body"], fontName=_BOLD, fontSize=size, leading=size + 2, textColor=colors.white
        )
        cell = ParagraphStyle("td", parent=self._styles["body"], fontSize=size, leading=size + 2)
        right = ParagraphStyle("tdr", parent=cell, alignment=TA_RIGHT)
        data: list[list[Any]] = [[Paragraph(_markup(name), head) for name in columns]]
        for row in rows:
            line = []
            for index in range(len(columns)):
                value = row[index] if index < len(row) else None
                text = cell_text(value)
                tone = cell_tone(value)
                style = right if index > 0 and is_numeric_text(text) else cell
                safe = _markup(text)
                if tone:
                    safe = f'<font color="{_TONE[tone]}">{"<b>" + safe + "</b>" if tone == "strong" else safe}</font>'
                line.append(Paragraph(safe, style))
            data.append(line)
        lengths = [
            max([len(columns[i])] + [len(cell_text(r[i])) for r in rows[:80] if i < len(r)])
            for i in range(len(columns))
        ]
        weights = [min(max(n, 4), 38) + 3 for n in lengths]
        widths = [self._width * w / sum(weights) for w in weights]
        table = Table(data, colWidths=widths, repeatRows=1, splitByRow=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self._primary)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        story: list[Any] = [CondPageBreak(32 * mm)]
        if block.get("title"):
            story.append(Paragraph(_markup(block["title"]), self._styles["table_title"]))
        story.append(table)
        if block.get("note"):
            story.append(
                Paragraph(
                    _markup(block["note"]),
                    ParagraphStyle("note", parent=self._styles["small"], fontName=_ITALIC, spaceBefore=2),
                )
            )
        story.append(Spacer(1, 3 * mm))
        return story

    def _block_figure(self, section, block) -> list[Any]:
        png = figure_image(section, block)
        size = image_size(png) if png else None
        if not png or size is None:
            return [Paragraph("(Gráfico no disponible)", self._styles["caption"]), Spacer(1, 2 * mm)]
        max_width, max_height = self._width, self._page_height * 0.58
        scale = min(max_width / size[0], max_height / size[1])
        picture = Image(io.BytesIO(flatten_png(png)), width=size[0] * scale, height=size[1] * scale, hAlign="CENTER")
        parts: list[Any] = [picture]
        if block.get("caption"):
            parts.append(Paragraph(_markup(block["caption"]), self._styles["caption"]))
        return [KeepTogether(parts), Spacer(1, 3 * mm)]

    def _mono(self, style: str, source: str, label: str | None = None) -> list[Any]:
        width = int(self._width / (7.8 * 0.602)) - 4  # characters per line in 7.8 pt DejaVu Sans Mono
        lines: list[str] = []
        for raw in (source or "").replace("\t", "    ").split("\n"):
            while len(raw) > width:
                lines.append(raw[:width])
                raw = raw[width:]
            lines.append(raw)
        body = "<br/>".join(_markup(line).replace(" ", "&nbsp;") for line in lines) or "&nbsp;"
        if label:
            body = f'<font color="{_GRAY}" size="6.5">{_markup(label)}</font><br/>{body}'
        return [Paragraph(body, self._styles[style])]

    def _block_code(self, section, block) -> list[Any]:
        return self._mono("code", str(block.get("source", "")), block.get("label"))

    def _block_output(self, section, block) -> list[Any]:
        return self._mono("output", str(block.get("text", "")))
