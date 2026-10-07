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
    model_id: str | None = None  # which configured model answers; omitted = the server's default


class AiSettingsRequest(RequestModel):
    """The switch, plus the shortcut to the Claude entry used by older versions of the page."""

    blank_as_none = False

    key: str | None = None
    clear_key: bool = False
    model: str | None = None
    enabled: bool | None = None


class AiModelCreateRequest(RequestModel):
    provider: str
    model: str
    label: str = ""
    base_url: str = ""
    key: str = ""
    enabled: bool = True


class AiModelUpdateRequest(RequestModel):
    blank_as_none = False

    provider: str | None = None
    model: str | None = None
    label: str | None = None
    base_url: str | None = None
    key: str | None = None
    clear_key: bool = False
    enabled: bool | None = None
    make_default: bool = False


class AiTestRequest(RequestModel):
    model_id: str | None = None
