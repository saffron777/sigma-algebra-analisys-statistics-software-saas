import io

import pytest
from pypdf import PdfReader

from sigma.domain.documents import LayoutOptions, ReportSection
from sigma.infrastructure.exporters.pdf_exporter import PdfExporter
from tests.unit.infrastructure.exporter_fixtures import GREEK, sample_document


def read(document):
    data = PdfExporter().export(document)
    assert data.startswith(b"%PDF-")
    return PdfReader(io.BytesIO(data))


def text_of(reader, pages=None):
    pages = reader.pages if pages is None else [reader.pages[i] for i in pages]
    return "\n".join(page.extract_text() for page in pages)


def test_cover_contents_and_sections():
    reader = read(sample_document(branded=True))
    cover = reader.pages[0].extract_text()
    assert "Informe de ventas" in cover and "Análisis trimestral" in cover
    assert "Acme Estadística" in cover and "Informe preparado para la gerencia" in cover
    assert "Ana Pérez" in cover and "2026-10-01" in cover
    assert "Contenido" in reader.pages[1].extract_text()
    toc = reader.pages[1].extract_text()
    assert "Regresión lineal múltiple" in toc and "ANOVA de un factor" in toc
    body = text_of(reader, range(2, len(reader.pages)))
    assert "Resumen del modelo" in body and "0.8421" in body and "Tabla ANOVA" in body


def test_greek_and_math_symbols_survive():
    reader = read(sample_document())
    body = text_of(reader)
    for symbol in ("Σ", "μ", "σ²", "χ²", "α", "≤", "±", "≠", "→", "∞"):
        assert symbol in body, symbol
    assert GREEK.split()[0] in body


def test_table_values_and_notes_are_present_and_text_is_not_markup():
    reader = read(sample_document())
    body = text_of(reader)
    for value in ("Intercepto", "12.3456", "<0.001", "[9.9, 14.7]", "Errores estándar clásicos."):
        assert value in body
    assert "=1+1" in body and "@cmd" in body


def test_hostile_markup_in_text_is_printed_literally():
    section = ReportSection(
        title="<b>Título</b> & más",
        blocks=[
            {"type": "text", "paragraphs": ["<script>alert(1)</script> <font color='red'>x</font> &amp; &lt;"]},
            {"type": "table", "title": "<i>t</i>", "columns": ["<a>"], "rows": [["<img src=x onerror=y>"]]},
        ],
    )
    reader = read(sample_document().__class__(title="x", sections=[section]))
    body = text_of(reader)
    assert "<b>Título</b> & más" in body and "<script>alert(1)</script>" in body and "<img src=x onerror=y>" in body


def test_images_bookmarks_and_metadata():
    reader = read(sample_document(branded=True))
    assert sum(len(page.images) for page in reader.pages) >= 4  # 2 charts + the logo on the cover and in the header
    titles = [item.title for item in reader.outline if not isinstance(item, list)]
    assert titles == ["Regresión lineal múltiple: ventas ~ precio + publicidad", "ANOVA de un factor"]
    assert reader.metadata.title == "Informe de ventas" and reader.metadata.author == "Ana Pérez"
    fonts = {name for page in reader.pages for name in (page.get("/Resources", {}).get("/Font", {}) or {})}
    assert fonts  # fonts are embedded (DejaVu), so the file looks the same everywhere


def test_header_footer_and_page_numbers():
    reader = read(sample_document(branded=True))
    total = len(reader.pages)
    assert total >= 4
    assert "Página" not in reader.pages[0].extract_text()  # no furniture on the cover
    second = reader.pages[2].extract_text()
    assert "Confidencial · Acme S.A." in second and "Acme Estadística" in second
    assert f"Página 3 de {total}" in second


def test_sections_start_on_a_new_page_and_tables_repeat_their_header():
    many_rows = [[f"fila {i}", i, i * 2.5] for i in range(160)]
    section = ReportSection(
        title="Tabla larga",
        blocks=[{"type": "table", "title": "Datos", "columns": ["Nombre", "n", "x"], "rows": many_rows}],
    )
    reader = read(
        sample_document().__class__(title="Larga", sections=[section], layout=LayoutOptions(cover=False, toc=False))
    )
    assert len(reader.pages) >= 3
    for page in reader.pages:
        assert "Nombre" in page.extract_text()  # header row repeated on each page


@pytest.mark.parametrize(
    ("size", "orientation", "expected"),
    [
        ("A4", "portrait", (595, 842)),
        ("A4", "landscape", (842, 595)),
        ("Letter", "portrait", (612, 792)),
        ("Legal", "landscape", (1008, 612)),
    ],
)
def test_page_size_and_orientation(size, orientation, expected):
    reader = read(sample_document(layout=LayoutOptions(page_size=size, orientation=orientation)))
    width, height = reader.pages[0].mediabox.width, reader.pages[0].mediabox.height
    assert (round(float(width)), round(float(height))) == expected


def test_without_cover_or_contents_the_report_starts_immediately():
    reader = read(sample_document(layout=LayoutOptions(cover=False, toc=False)))
    first = reader.pages[0].extract_text()
    assert "Regresión lineal múltiple" in first and "Contenido" not in first
    assert "Página 1 de" in first  # numbering starts on the first page when there is no cover


def test_missing_images_and_long_unbroken_text_do_not_break_the_layout():
    section = ReportSection(
        title="Raro",
        blocks=[
            {"type": "figure", "index": 3, "caption": "no existe"},
            {"type": "text", "paragraphs": ["x" * 400]},
            {"type": "table", "columns": ["a", "b"], "rows": [["y" * 300, 1]]},
            {"type": "code", "source": "z" * 500},
        ],
    )
    reader = read(sample_document().__class__(title="Raro", sections=[section]))
    assert "(Gráfico no disponible)" in text_of(reader)


def test_empty_document():
    reader = read(sample_document().__class__(title="Vacío", sections=[]))
    assert "Vacío" in reader.pages[0].extract_text()
