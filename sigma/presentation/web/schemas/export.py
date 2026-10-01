"""Request models for the export and template endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sigma.presentation.web.schemas.base import RequestModel


class ExportSectionModel(RequestModel):
    title: str = ""
    subtitle: str | None = None
    meta: list[str] = []
    blocks: list[dict[str, Any]] = []


class ExportPayload(RequestModel):
    """What the browser sends: sections already rendered as blocks (the chart images travel as separate parts)."""

    title: str | None = None
    subtitle: str | None = None
    author: str = ""
    template_id: str | None = None
    cover: bool | None = None
    toc: bool | None = None
    page_size: str | None = None
    orientation: str | None = None
    sections: list[ExportSectionModel] = Field(default_factory=list)


class TemplateRequest(RequestModel):
    name: str = ""
    organization: str = ""
    primary_color: str | None = None
    secondary_color: str | None = None
    footer_text: str = ""
    cover_text: str = ""
    page_size: str = "A4"
    orientation: str = "portrait"
    show_cover: bool = True
    show_toc: bool = True
    make_default: bool = False

    def fields(self) -> dict[str, Any]:
        return self.model_dump(exclude={"make_default"})


class DefaultTemplateRequest(RequestModel):
    default: bool = True
