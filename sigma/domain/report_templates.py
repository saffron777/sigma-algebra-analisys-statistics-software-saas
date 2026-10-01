"""Report templates: a user's corporate look (name, colours, footer, cover text, logo, page set-up) for exports."""

from __future__ import annotations

from dataclasses import dataclass

from sigma.domain.documents import (
    DEFAULT_PRIMARY,
    DEFAULT_SECONDARY,
    HEX_COLOR,
    ORIENTATIONS,
    PAGE_SIZES,
)
from sigma.domain.errors import InvalidInputError

MAX_LOGO_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ReportTemplate:
    id: str
    name: str
    organization: str = ""
    primary_color: str = DEFAULT_PRIMARY
    secondary_color: str = DEFAULT_SECONDARY
    footer_text: str = ""
    cover_text: str = ""
    has_logo: bool = False
    page_size: str = "A4"
    orientation: str = "portrait"
    show_cover: bool = True
    show_toc: bool = True
    is_default: bool = False
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class TemplateFields:
    """The editable part of a template, validated."""

    name: str
    organization: str
    primary_color: str
    secondary_color: str
    footer_text: str
    cover_text: str
    page_size: str
    orientation: str
    show_cover: bool
    show_toc: bool


def _text(value: str | None, label: str, limit: int, required: bool = False) -> str:
    cleaned = " ".join((value or "").split())
    if required and not cleaned:
        raise InvalidInputError(f"{label} no puede estar vacío.")
    if len(cleaned) > limit:
        raise InvalidInputError(f"{label} admite como máximo {limit} caracteres.")
    return cleaned


def _color(value: str | None, label: str, fallback: str) -> str:
    value = (value or "").strip() or fallback
    if not HEX_COLOR.match(value):
        raise InvalidInputError(f"{label} debe ser un color hexadecimal como #4f46e5.")
    return value.lower()


def validate_fields(
    name: str | None,
    organization: str | None = "",
    primary_color: str | None = None,
    secondary_color: str | None = None,
    footer_text: str | None = "",
    cover_text: str | None = "",
    page_size: str | None = "A4",
    orientation: str | None = "portrait",
    show_cover: bool = True,
    show_toc: bool = True,
) -> TemplateFields:
    if (page_size or "A4") not in PAGE_SIZES:
        raise InvalidInputError(f"Tamaño de página inválido: {', '.join(PAGE_SIZES)}.")
    if (orientation or "portrait") not in ORIENTATIONS:
        raise InvalidInputError("La orientación debe ser vertical u horizontal.")
    return TemplateFields(
        name=_text(name, "El nombre de la plantilla", 80, required=True),
        organization=_text(organization, "La organización", 120),
        primary_color=_color(primary_color, "El color principal", DEFAULT_PRIMARY),
        secondary_color=_color(secondary_color, "El color secundario", DEFAULT_SECONDARY),
        footer_text=_text(footer_text, "El texto de pie", 200),
        cover_text=_text(cover_text, "El texto de portada", 300),
        page_size=page_size or "A4",
        orientation=orientation or "portrait",
        show_cover=bool(show_cover),
        show_toc=bool(show_toc),
    )
