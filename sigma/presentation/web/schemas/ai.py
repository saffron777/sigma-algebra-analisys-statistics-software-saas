"""Request models for the AI endpoints."""

from __future__ import annotations

from typing import Any

from sigma.presentation.web.schemas.base import RequestModel


class InterpretRequest(RequestModel):
    module: str
    title: str = ""
    focus: str | None = None
    report: dict[str, Any] | None = None  # the report blocks (never the dataset's rows)
    dataset_id: str | None = None  # for module «dataset»: the server builds the ficha itself


class AiSettingsRequest(RequestModel):
    blank_as_none = False

    key: str | None = None
    clear_key: bool = False
    model: str | None = None
    enabled: bool | None = None
