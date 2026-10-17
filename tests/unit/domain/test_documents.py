import base64

import numpy as np
import pandas as pd

from sigma.domain.dataset_report import dataset_data_table, dataset_section
from sigma.domain.datasets import ColumnInfo, DatasetInfo, describe_columns
from sigma.domain.documents import (
    cell_number,
    cell_text,
    cell_tone,
    clean_blocks,
    is_numeric_text,
    normalize_color,
    safe_filename,
)
from sigma.domain.markdown_blocks import inline_runs, markdown_to_blocks, plain
from sigma.domain.notebook_report import NotebookReportOptions, html_table, notebook_section
from sigma.domain.notebooks import NotebookDocument
from tests.unit.infrastructure.exporter_fixtures import make_png


# ------------------------------------------------------------------ cells and names
def test_cell_helpers():
    assert cell_text(None) == "—" and cell_text({"text": "ok", "tone": "good"}) == "ok" and cell_text(3.5) == "3.5"
    assert cell_tone({"text": "x", "tone": "bad"}) == "bad" and cell_tone({"text": "x", "tone": "weird"}) is None
    assert cell_number("12.5") == 12.5 and cell_number("7") == 7 and isinstance(cell_number("7"), int)
    assert cell_number(" -3e2 ") == -300.0 and cell_number({"text": "0.25"}) == 0.25
    for text in ("<0.001", "[1, 2]", "12 %", "1,5", "abc", "", True, None, "—"):
        assert cell_number(text) is None
    assert (
        is_numeric_text("12.5")
        and is_numeric_text("[1.2, 3.4]")
        and is_numeric_text("<0.001")
        and not is_numeric_text("Norte")
    )


def test_colours_and_file_names():
    assert normalize_color("#AbCdEf", "#000000") == "#abcdef"
    assert normalize_color("red", "#111111") == "#111111" and normalize_color(None, "#222222") == "#222222"
    assert (
        safe_filename("Informe: ventas/2024 <final>?", "pdf", "2026-10-01")
        == "Sigma Algebra - Informe ventas 2024 final - 2026-10-01.pdf"
    )
    assert safe_filename("   ", "xlsx", "2026-10-01") == "Sigma Algebra - Informe - 2026-10-01.xlsx"
    assert len(safe_filename("x" * 500, "pdf", "2026-10-01")) < 120
    assert ".." not in safe_filename("../../etc/passwd", "pdf", "d").replace("etc passwd", "")


# ------------------------------------------------------------------ untrusted blocks from the browser
def test_clean_blocks_keeps_only_known_well_formed_blocks():
    dirty = [
        {"type": "heading", "text": "T", "level": 9},
        {"type": "script", "src": "x"},
        "not a block",
        {"type": "tiles", "items": [{"label": "L", "value": 5, "hint": "h"}, "bad", {"label": "x"}]},
        {
            "type": "table",
            "columns": ["a", "b"],
            "rows": [[1, {"text": "x", "tone": "good"}, "extra"], "bad", [None, True]],
        },
        {"type": "alert", "kind": "explosive", "message": "m"},
        {"type": "figure", "index": -4, "caption": "c"},
        {"type": "text", "paragraphs": ["p", 7], "markdown": 1},
        {"type": "code", "source": "x" * 300_000},
        {"type": "pagebreak", "evil": 1},
    ]
    cleaned = clean_blocks(dirty)
    assert [b["type"] for b in cleaned] == ["heading", "tiles", "table", "alert", "figure", "text", "code", "pagebreak"]
    assert cleaned[0]["level"] == 2 and cleaned[1]["items"][0] == {"label": "L", "value": "5", "hint": "h"}
    assert cleaned[2]["rows"] == [
        [1, {"text": "x", "tone": "good"}],
        [None, "True"],
    ]  # cut to the columns; no junk rows
    assert cleaned[3]["kind"] == "info" and cleaned[4]["index"] == 0
    assert cleaned[5]["paragraphs"] == ["p", "7"] and cleaned[5]["markdown"] is True
    assert len(cleaned[6]["source"]) == 200_000 and cleaned[7] == {"type": "pagebreak"}
    assert clean_blocks("nonsense") == [] and clean_blocks(None) == []


def test_clean_blocks_bounds_sizes():
    big = {"type": "table", "columns": [str(i) for i in range(100)], "rows": [[1] * 100 for _ in range(6000)]}
    table = clean_blocks([big])[0]
    assert len(table["columns"]) == 60 and len(table["rows"]) == 5000 and len(table["rows"][0]) == 60
    assert len(clean_blocks([{"type": "pagebreak"}] * 5000)) == 3000


# ------------------------------------------------------------------ markdown
def test_markdown_blocks():
    source = (
        "# Título\n\nUn párrafo con **negrita**, *cursiva* y `código`.\nsegunda línea.\n\n"
        "- uno\n- dos\n\n1. primero\n2. segundo\n\n```python\nx = 1\n```\n\n"
        "| a | b |\n|---|---|\n| 1 | **2** |\n\n> cita\n\n---\n### Sub"
    )
    blocks = markdown_to_blocks(source, base_level=2)
    assert [b["type"] for b in blocks] == ["heading", "text", "list", "list", "code", "table", "text", "heading"]
    assert blocks[0] == {"type": "heading", "text": "Título", "level": 2}
    assert blocks[1]["paragraphs"] == ["Un párrafo con **negrita**, *cursiva* y `código`. segunda línea."]
    assert blocks[2] == {"type": "list", "ordered": False, "items": ["uno", "dos"]}
    assert blocks[3]["ordered"] is True and blocks[4]["source"] == "x = 1"
    assert blocks[5]["columns"] == ["a", "b"] and blocks[5]["rows"] == [["1", "2"]]
    assert blocks[6]["paragraphs"] == ["cita"] and blocks[7]["level"] == 4


def test_inline_runs_and_plain_text_never_keep_html():
    assert inline_runs("a **b** *c* `d` e") == [
        ("", "a "),
        ("b", "b"),
        ("", " "),
        ("i", "c"),
        ("", " "),
        ("c", "d"),
        ("", " e"),
    ]
    assert plain("**x** [enlace](http://a.b) <script>alert(1)</script>") == "x enlace (http://a.b) alert(1)"
    assert ("", "<b>x</b>") not in inline_runs("<b>x</b>")


# ------------------------------------------------------------------ notebooks
def make_notebook(cells):
    return NotebookDocument(
        id="a" * 32, name="Mi análisis", created_at="2026-01-01", updated_at="2026-10-01T00:00:00", cells=cells
    )


def test_notebook_to_blocks_covers_text_code_and_every_output_kind():
    png = base64.b64encode(make_png()).decode()
    table_html = (
        "<table><thead><tr><th></th><th>x</th></tr></thead>"
        "<tbody><tr><th>0</th><td>1.5</td></tr><tr><th>1</th><td>2</td></tr></tbody></table>"
    )
    cells = [
        {"cell_type": "markdown", "source": "# Introducción\nTexto *importante*.", "metadata": {}},
        {
            "cell_type": "code",
            "source": ["x = 1\n", "print(x)"],
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
                {"output_type": "execute_result", "data": {"text/html": table_html, "text/plain": "df"}},
                {"output_type": "display_data", "data": {"image/png": png, "text/plain": "<Figure>"}},
                {
                    "output_type": "error",
                    "ename": "ValueError",
                    "evalue": "mal",
                    "traceback": ["\x1b[31mTraceback\x1b[0m", "ValueError: mal"],
                },
            ],
        },
    ]
    section = notebook_section(make_notebook(cells), NotebookReportOptions(number_cells=True))
    kinds = [b["type"] for b in section.blocks]
    assert kinds == ["heading", "text", "code", "output", "table", "figure", "output"]
    code = section.blocks[2]
    assert code["source"] == "x = 1\nprint(x)" and code["label"] == "Celda [3]"
    assert section.blocks[3]["text"] == "1"
    assert section.blocks[4]["columns"] == [" ", "x"] and section.blocks[4]["rows"] == [["0", "1.5"], ["1", "2"]]
    assert section.blocks[5]["index"] == 0 and section.images[0].startswith(b"\x89PNG")
    assert "\x1b" not in section.blocks[6]["text"] and "ValueError: mal" in section.blocks[6]["text"]
    assert section.title == "Mi análisis" and section.subtitle == "Notebook · 2 celdas"


def test_notebook_options_and_page_breaks():
    cells = [
        {"cell_type": "markdown", "source": "# Uno\ntexto", "metadata": {}},
        {"cell_type": "code", "source": "a = 1", "outputs": [{"output_type": "stream", "text": "ok"}], "metadata": {}},
        {"cell_type": "markdown", "source": "## Dos", "metadata": {}},
        {"cell_type": "code", "source": "b = 2", "outputs": [], "metadata": {"sigma": {"page_break_before": True}}},
        {"cell_type": "markdown", "source": "# Tres", "metadata": {}},
    ]
    notebook = make_notebook(cells)
    full = notebook_section(notebook, NotebookReportOptions(break_before_h1=True, break_before_h2=True))
    assert [b["type"] for b in full.blocks].count("pagebreak") == 3  # before «Dos», the manual one and before «Tres»
    only_text = notebook_section(notebook, NotebookReportOptions(include_code=False, include_outputs=False))
    assert {b["type"] for b in only_text.blocks} <= {"heading", "text", "pagebreak"}
    only_code = notebook_section(notebook, NotebookReportOptions(include_text=False, include_outputs=False))
    assert [b["type"] for b in only_code.blocks if b["type"] != "pagebreak"] == ["code", "code"]


def test_html_table_parser_is_defensive():
    assert html_table("<p>nada</p>") is None
    assert html_table("<table></table>") is None
    assert html_table("<table><tr><td>a</td><td>b</td></tr><tr><td>1</td></tr></table>")["rows"] == [["1", ""]]
    huge = "<table><thead><tr><th>x</th></tr></thead>" + "<tr><td>1</td></tr>" * 1000 + "</table>"
    parsed = html_table(huge)
    assert len(parsed["rows"]) == 400 and "primeras 400" in parsed["note"]
    assert html_table("<table><tr><td><script>alert(1)</script></td></tr></table>")["columns"] == ["alert(1)"]


# ------------------------------------------------------------------ dataset ficha
def test_dataset_ficha_has_summary_columns_statistics_and_rows():
    frame = pd.DataFrame(
        {
            "ventas": [10.0, 20.0, np.nan, 40.0],
            "region": ["N", "S", "N", None],
            "fecha": pd.to_datetime(["2024-01-01", "2024-02-01", None, "2024-04-01"]),
        }
    )
    info = DatasetInfo(
        id="a" * 32,
        name="ventas",
        source_type="csv",
        n_rows=4,
        n_cols=3,
        columns=describe_columns(frame),
        created_at="2026-10-01T10:00:00",
        source_meta={},
    )
    section = dataset_section(info, frame)
    assert section.title == "Ficha del dataset «ventas»"
    tables = {b.get("title"): b for b in section.blocks if b["type"] == "table"}
    assert set(tables) >= {
        "Columnas",
        "Resumen de las columnas numéricas",
        "Frecuencias de «region»",
        "Primeras 4 filas",
    }
    assert [r[0] for r in tables["Columnas"]["rows"]] == ["ventas", "region", "fecha"]
    numeric = tables["Resumen de las columnas numéricas"]["rows"][0]
    assert numeric[0] == "ventas" and numeric[1] == 3 and numeric[2] == "23.3333"
    assert any(b["type"] == "alert" for b in section.blocks)  # >20 % missing in some column
    head = tables["Primeras 4 filas"]
    assert head["rows"][2][0] is None and isinstance(head["rows"][0][2], str)  # NaN → empty, dates → text
    data = dataset_data_table(info, frame)
    assert data.columns == ["ventas", "region", "fecha"] and len(data.rows) == 4 and data.rows[2][0] is None
    assert hasattr(data.rows[0][2], "year")  # a real datetime for Excel
    assert isinstance(ColumnInfo("a", "numeric", "int64", 0, 1), ColumnInfo)


# ------------------------------------------------------------------ AI text in reports
def test_markdown_blocks_from_the_browser_become_ordinary_blocks():
    cleaned = clean_blocks(
        [{"type": "markdown", "text": "**Resumen**\nTodo bien.\n\n- uno\n- dos\n\n### Detalle\ntexto"}]
    )
    assert [b["type"] for b in cleaned] == ["text", "list", "heading", "text"]
    assert (
        cleaned[0]["markdown"] is True and cleaned[1]["items"] == ["uno", "dos"] and cleaned[2]["level"] == 4
    )  # «###» under the section's own «Interpretación» heading: deepest level
    assert clean_blocks([{"type": "markdown", "text": "<script>x</script> normal"}])[0]["paragraphs"] == [
        "<script>x</script> normal"
    ]


def test_a_cell_interpretation_is_part_of_the_notebook_report():
    cell = {
        "cell_type": "code",
        "source": "print(6)",
        "outputs": [{"output_type": "stream", "text": "6\n"}],
        "metadata": {"sigma": {"interpretation": {"text": "**Qué hace**\nImprime 6.", "model": "claude-opus-5-5"}}},
    }
    section = notebook_section(make_notebook([cell]))
    kinds = [b["type"] for b in section.blocks]
    assert kinds == ["code", "output", "heading", "text", "alert"]
    assert section.blocks[2]["text"] == "Interpretación (IA)" and "Imprime 6." in section.blocks[3]["paragraphs"][0]
    assert (
        "puede contener errores" in section.blocks[4]["message"] and "claude-opus-5-5" in section.blocks[4]["message"]
    )
    plain = notebook_section(make_notebook([{**cell, "metadata": {}}]))
    assert [b["type"] for b in plain.blocks] == ["code", "output"]
    off = notebook_section(make_notebook([cell]), NotebookReportOptions(include_outputs=False))
    assert [b["type"] for b in off.blocks] == ["code"]  # the explanation follows the outputs setting
