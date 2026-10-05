"""Request models for the console / notebook endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from sigma.presentation.web.schemas.base import NonEmpty, RequestModel


class CreateNotebookRequest(RequestModel):
    name: str | None = None
    language: Literal["python", "r"] = "python"


class RenameNotebookRequest(RequestModel):
    name: NonEmpty


class SaveNotebookRequest(RequestModel):
    cells: list[dict[str, Any]]
    metadata: dict[str, Any] | None = None
    name: str | None = None
    base_revision: str | None = None  # the revision the editor loaded; a mismatch answers 409


class ExecuteRequest(RequestModel):
    source: str = ""
    timeout: Annotated[float, Field(gt=0)] | None = None
