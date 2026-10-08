"""Request models for survey projects."""

from __future__ import annotations

from typing import Any, Literal

from sigma.presentation.web.schemas.base import NonEmpty, RequestModel


class CreateProjectRequest(RequestModel):
    name: NonEmpty
    population_size: int | None = None
    sample_size: int | None = None
    indicators: list[str] = []
    design: list[dict[str, Any]] = []
    data: list[dict[str, Any]] = []


class SaveProjectRequest(RequestModel):
    name: NonEmpty
    population_size: int | None = None
    sample_size: int | None = None
    indicators: list[str] = []
    design: list[dict[str, Any]] = []
    data: list[dict[str, Any]] | None = None
    results: dict[str, Any] | None = None


class ProjectAnalysisRequest(RequestModel):
    analysis_type: Literal["sample_size", "estimate", "crosstab", "likert"]
    params: dict[str, Any] = {}
