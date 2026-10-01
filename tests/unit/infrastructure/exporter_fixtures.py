"""A document that exercises every block type, shared by the exporter tests."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from sigma.domain.documents import Branding, DataTable, LayoutOptions, ReportDocument, ReportSection


def make_png(width=640, height=360, color=(79, 70, 229)) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle((40, 40, width - 40, height - 40), outline=color, width=6)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def make_logo(color=(220, 38, 38)) -> bytes:
    image = Image.new("RGBA", (200, 80), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((4, 4, 76, 76), fill=color + (255,))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


GREEK = "Σ μ σ² χ² α ≤ β̂ ± ≠ → ∞"


def sample_sections(with_images=True) -> list[ReportSection]:
    regression = ReportSection(
        title="Regresión lineal múltiple: ventas ~ precio + publicidad",
        subtitle="Dataset: ventas · 300 observaciones",
        meta=["Dataset: ventas", "Fecha: 2026-10-01"],
        blocks=[
            {"type": "heading", "text": "Resumen del modelo"},
            {
                "type": "tiles",
                "items": [
                    {"label": "R²", "value": "0.8421", "hint": "ajustado 0.8398"},
                    {"label": "n", "value": "300"},
                    {"label": f"Estadístico {GREEK}", "value": "F = 812.4"},
                ],
            },
            {"type": "text", "paragraphs": ["El modelo explica el 84 % de la varianza.", f"Símbolos: {GREEK}"]},
            {
                "type": "table",
                "title": "Coeficientes",
                "columns": ["Término", "Estimación", "Error est.", "t", "Valor p", "IC 95 %"],
                "rows": [
                    ["Intercepto", "12.3456", "1.2", "10.28", {"text": "<0.001", "tone": "good"}, "[9.9, 14.7]"],
                    ["precio", "-2.5", "0.3", "-8.3", {"text": "<0.001", "tone": "good"}, "[-3.1, -1.9]"],
                    ["=1+1", "@cmd", "+SUM(A1)", None, {"text": "0.4210", "tone": "bad"}, "—"],
                ],
                "note": "Errores estándar clásicos.",
            },
            {"type": "alert", "kind": "warning", "message": "Hay 3 observaciones influyentes."},
            {"type": "figure", "index": 0, "caption": "Ajuste y residuos"},
            {"type": "list", "ordered": False, "items": ["Revisar outliers", "Probar log(ventas)"]},
            {"type": "code", "source": "df = sigma.load('ventas')\nprint(df.describe())"},
            {"type": "output", "text": "       ventas\ncount  300.0"},
        ],
        images=[make_png() if with_images else None],
    )
    anova = ReportSection(
        title="ANOVA de un factor",
        blocks=[
            {"type": "table", "title": "Tabla ANOVA", "columns": ["Fuente", "gl", "F"], "rows": [["region", 3, 4.5]]},
            {"type": "table", "title": "Tabla ANOVA", "columns": ["Fuente", "gl", "F"], "rows": [["canal", 1, 9.1]]},
            {"type": "figure", "index": 0, "caption": "Cajas por grupo"},
        ],
        images=[make_png(800, 500, (5, 150, 105)) if with_images else None],
    )
    return [regression, anova]


def sample_document(*, branded=False, layout=None, with_images=True, data=True) -> ReportDocument:
    branding = (
        Branding(
            organization="Acme Estadística",
            primary_color="#dc2626",
            footer_text="Confidencial · Acme S.A.",
            cover_text="Informe preparado para la gerencia",
            logo_png=make_logo(),
        )
        if branded
        else Branding()
    )
    tables = (
        [
            DataTable(
                "Datos ventas",
                ["id", "precio", "fecha", "nota"],
                [[i, i * 1.5, None, "=cmd" if i == 1 else "ok"] for i in range(1, 51)],
            )
        ]
        if data
        else []
    )
    return ReportDocument(
        title="Informe de ventas",
        subtitle="Análisis trimestral",
        author="Ana Pérez",
        created_at="2026-10-01T10:30:00+00:00",
        sections=sample_sections(with_images),
        branding=branding,
        layout=layout or LayoutOptions(),
        data_tables=tables,
    )
