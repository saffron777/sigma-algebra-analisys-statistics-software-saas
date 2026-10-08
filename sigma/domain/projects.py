"""Survey projects (market-research studies): design, responses and analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectSummary:
    """One row of the projects list (the full payload is loaded on demand)."""

    id: str
    name: str
    n_questions: int
    n_responses: int
    sample_size: int | None
    population_size: int | None
    has_results: bool
    created_at: str
    updated_at: str


@dataclass
class Project:
    """A survey project: metadata, the question design, a dataset holding the responses, and the analysis outputs."""

    id: str
    user_id: int
    name: str
    population_size: int | None
    sample_size: int | None
    indicators: list[str]
    design: list[dict[str, Any]]  # [{name, kind, categories, numericType, decimals}, …]
    dataset_id: str | None  # the responses are stored as a real dataset (like a manual one)
    dataset_name: str | None
    n_responses: int
    results: dict[str, Any] | None  # analysis outputs, None until the first one runs
    created_at: str
    updated_at: str

    @property
    def summary(self) -> ProjectSummary:
        return ProjectSummary(
            id=self.id,
            name=self.name,
            n_questions=len(self.design),
            n_responses=self.n_responses,
            sample_size=self.sample_size,
            population_size=self.population_size,
            has_results=bool(self.results),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
