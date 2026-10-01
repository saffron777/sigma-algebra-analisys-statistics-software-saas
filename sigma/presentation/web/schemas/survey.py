"""Request models for the survey / market-research endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from sigma.application.dto import (
    CrosstabCommand,
    DrawSampleCommand,
    EstimateCommand,
    LikertCommand,
    SampleSizeCommand,
)
from sigma.presentation.web.schemas.base import Confidence, NonEmpty, RequestModel

Allocation = Literal["proportional", "equal", "neyman"]


class SampleSizeRequest(RequestModel):
    kind: Literal["proportion", "mean"] = "proportion"
    confidence: Confidence = 0.95
    margin_of_error: Annotated[float, Field(gt=0)] = 0.05
    proportion: Annotated[float, Field(gt=0, lt=1)] = 0.5
    std_dev: Annotated[float, Field(gt=0)] | None = None
    population_size: Annotated[int, Field(ge=2)] | None = None
    design_effect: Annotated[float, Field(ge=1)] = 1.0
    dataset_id: str | None = None
    strata_column: str | None = None
    allocation: Allocation = "proportional"
    allocation_sd_column: str | None = None

    def to_command(self) -> SampleSizeCommand:
        return SampleSizeCommand(**self.model_dump())


class DrawSampleRequest(RequestModel):
    dataset_id: NonEmpty
    design: Literal["srs", "systematic", "stratified", "cluster"]
    sample_size: Annotated[int, Field(ge=1)] | None = None
    seed: Annotated[int, Field(ge=0, lt=2**31)] | None = None
    strata_column: str | None = None
    allocation: Allocation = "proportional"
    sd_column: str | None = None
    cluster_column: str | None = None
    n_clusters: Annotated[int, Field(ge=1)] | None = None
    save_as: str | None = None

    def to_command(self) -> DrawSampleCommand:
        return DrawSampleCommand(**self.model_dump())


class EstimateRequest(RequestModel):
    dataset_id: NonEmpty
    variable: NonEmpty
    kind: Literal["proportion", "mean"] = "proportion"
    confidence: Confidence = 0.95
    methods: list[Literal["wilson", "wald", "agresti_coull", "clopper_pearson"]] = ["wilson"]
    population_size: Annotated[int, Field(ge=2)] | None = None
    weight_column: str | None = None
    strata_column: str | None = None
    cluster_column: str | None = None
    apply_fpc: bool = True

    def to_command(self) -> EstimateCommand:
        return EstimateCommand(**self.model_dump())


class CrosstabRequest(RequestModel):
    dataset_id: NonEmpty
    row_variable: NonEmpty
    col_variable: NonEmpty

    def to_command(self) -> CrosstabCommand:
        return CrosstabCommand(**self.model_dump())


class LikertRequest(RequestModel):
    dataset_id: NonEmpty
    items: Annotated[list[str], Field(min_length=1)]

    def to_command(self) -> LikertCommand:
        return LikertCommand(**self.model_dump())
