"""Request models for dataset management and saved results."""

from __future__ import annotations

from typing import Any, Literal

from sigma.application.dto import SaveResultCommand
from sigma.presentation.web.schemas.base import NonEmpty, RequestModel


class ManualColumn(RequestModel):
    name: NonEmpty
    kind: Literal["numeric", "categorical", "boolean", "datetime", "text"]


class ManualDatasetRequest(RequestModel):
    name: str | None = None
    columns: list[ManualColumn]
    rows: list[dict[str, Any]] = []


class CommitImportRequest(RequestModel):
    token: NonEmpty
    name: str | None = None
    column_kinds: dict[str, str] = {}


class RenameRequest(RequestModel):
    name: NonEmpty


class ColumnTypesRequest(RequestModel):
    column_kinds: dict[str, str]


class SaveResultRequest(RequestModel):
    module: NonEmpty
    analysis_type: NonEmpty
    title: NonEmpty
    notes: str = ""
    dataset_id: str | None = None
    params: dict[str, Any] = {}
    result: dict[str, Any] = {}
    report: dict[str, Any] = {}
    figures: list[dict[str, Any]] = []
    interpretation: dict[str, Any] | None = None

    def to_command(self) -> SaveResultCommand:
        return SaveResultCommand(**self.model_dump())


class UpdateResultRequest(RequestModel):
    blank_as_none = False  # an empty string clears the notes; ``None`` leaves a field untouched

    title: str | None = None
    notes: str | None = None
    interpretation: dict[str, Any] | None = None  # sent as null to remove it
