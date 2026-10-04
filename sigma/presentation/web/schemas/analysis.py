"""Request models for the analysis endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from sigma.application.dto import (
    AnalysisCommand,
    AnovaCommand,
    DescriptiveCommand,
    DistributionCommand,
    HypothesisCommand,
    RegressionCommand,
)
from sigma.presentation.web.schemas.base import Confidence, NonEmpty, RequestModel

Tail = Literal["two-sided", "left", "right"]
Alpha = Annotated[float, Field(gt=0, lt=1)]


class DescriptiveRequest(RequestModel):
    dataset_id: NonEmpty
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    group_by: str | None = None
    confidence: Confidence = 0.95
    run_normality: bool = True
    fit_distributions: list[str] = []
    correlation_method: Literal["pearson", "spearman", "kendall"] | None = None

    def to_command(self) -> DescriptiveCommand:
        return DescriptiveCommand(**self.model_dump())


class HypothesisRequest(RequestModel):
    test_id: NonEmpty
    dataset_id: str | None = None
    roles: dict[str, str | None] = {}
    params: dict[str, Any] = {}
    summary: dict[str, Any] | None = None
    alpha: Alpha = 0.05
    tail: Tail = "two-sided"

    def to_command(self) -> HypothesisCommand:
        data = self.model_dump()
        data["roles"] = {k: v for k, v in self.roles.items() if v}
        data["params"] = {k: v for k, v in self.params.items() if v not in (None, "")}
        data["summary"] = {k: v for k, v in (self.summary or {}).items() if v not in (None, "")} or None
        return HypothesisCommand(**data)


class RegressionRequest(RequestModel):
    dataset_id: NonEmpty
    response: NonEmpty
    predictors: Annotated[list[str], Field(min_length=1)]
    categorical: list[str] = []
    intercept: bool = True
    confidence: Confidence = 0.95
    predict: list[dict[str, Any]] = []

    def to_command(self) -> RegressionCommand:
        return RegressionCommand(**self.model_dump())


class AnovaRequest(RequestModel):
    dataset_id: NonEmpty
    response: NonEmpty
    factors: Annotated[list[str], Field(min_length=1, max_length=2)]
    alpha: Alpha = 0.05
    interaction: bool = True
    ss_type: Literal[1, 2, 3] = 2
    post_hoc: bool = True

    def to_command(self) -> AnovaCommand:
        return AnovaCommand(**self.model_dump())


class DistributionRequest(RequestModel):
    distribution: NonEmpty
    params: dict[str, float] = {}
    mode: Literal["point", "between", "quantile", "pvalue"] = "point"
    x: float | None = None
    a: float | None = None
    b: float | None = None
    q: float | None = None
    statistic: float | None = None
    tail: Tail = "two-sided"
    alpha: Alpha = 0.05

    def to_command(self) -> DistributionCommand:
        return DistributionCommand(**self.model_dump())


class AnalysisRunRequest(RequestModel):
    analysis_id: NonEmpty
    dataset_id: str | None = None
    roles: dict[str, str | list[str] | None] = {}
    params: dict[str, Any] = {}
    save_as: dict[str, str] = {}
    replace: bool = False

    def to_command(self, module: str) -> AnalysisCommand:
        return AnalysisCommand(
            module=module,
            analysis_id=self.analysis_id,
            dataset_id=self.dataset_id or None,
            roles={k: v for k, v in self.roles.items() if v},
            params={k: v for k, v in self.params.items() if v is not None},
            save_as={k: v for k, v in self.save_as.items() if (v or "").strip()},
            replace=self.replace,
        )


class AnalysisCodeRequest(RequestModel):
    analysis_id: NonEmpty
    dataset_id: str | None = None
    roles: dict[str, str | list[str] | None] = {}
    params: dict[str, Any] = {}
