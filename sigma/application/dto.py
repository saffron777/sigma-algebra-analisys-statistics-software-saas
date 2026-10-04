"""Input commands of the use cases (plain dataclasses, independent of the web layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DescriptiveCommand:
    dataset_id: str
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    group_by: str | None = None
    confidence: float = 0.95
    run_normality: bool = True
    fit_distributions: list[str] = field(default_factory=list)
    correlation_method: str | None = None


@dataclass(frozen=True)
class HypothesisCommand:
    test_id: str
    dataset_id: str | None = None
    roles: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] | None = None
    alpha: float = 0.05
    tail: str = "two-sided"


@dataclass(frozen=True)
class RegressionCommand:
    dataset_id: str
    response: str
    predictors: list[str]
    categorical: list[str] = field(default_factory=list)
    intercept: bool = True
    confidence: float = 0.95
    predict: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AnovaCommand:
    dataset_id: str
    response: str
    factors: list[str]
    alpha: float = 0.05
    interaction: bool = True
    ss_type: int = 2
    post_hoc: bool = True


@dataclass(frozen=True)
class DistributionCommand:
    distribution: str
    params: dict[str, float]
    mode: str = "point"  # point | between | quantile | pvalue
    x: float | None = None
    a: float | None = None
    b: float | None = None
    q: float | None = None
    statistic: float | None = None
    tail: str = "two-sided"
    alpha: float = 0.05


@dataclass(frozen=True)
class SampleSizeCommand:
    kind: str  # proportion | mean
    confidence: float = 0.95
    margin_of_error: float = 0.05
    proportion: float = 0.5
    std_dev: float | None = None
    population_size: int | None = None
    design_effect: float = 1.0
    dataset_id: str | None = None  # sampling frame, used for the stratified allocation
    strata_column: str | None = None
    allocation: str = "proportional"
    allocation_sd_column: str | None = None


@dataclass(frozen=True)
class DrawSampleCommand:
    dataset_id: str
    design: str
    sample_size: int | None = None
    seed: int | None = None
    strata_column: str | None = None
    allocation: str = "proportional"
    sd_column: str | None = None
    cluster_column: str | None = None
    n_clusters: int | None = None
    save_as: str | None = None


@dataclass(frozen=True)
class EstimateCommand:
    dataset_id: str
    variable: str
    kind: str = "proportion"
    confidence: float = 0.95
    methods: list[str] = field(default_factory=lambda: ["wilson"])
    population_size: int | None = None
    weight_column: str | None = None
    strata_column: str | None = None
    cluster_column: str | None = None
    apply_fpc: bool = True


@dataclass(frozen=True)
class CrosstabCommand:
    dataset_id: str
    row_variable: str
    col_variable: str


@dataclass(frozen=True)
class LikertCommand:
    dataset_id: str
    items: list[str]


@dataclass(frozen=True)
class SaveResultCommand:
    module: str
    analysis_type: str
    title: str
    notes: str = ""
    dataset_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    interpretation: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisCommand:
    """One run of a catalogue analysis (see ``domain/analysis``)."""

    module: str
    analysis_id: str
    dataset_id: str | None = None
    roles: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    save_as: dict[str, str] = field(default_factory=dict)  # output key -> name of the new dataset
    replace: bool = False  # overwrite the dataset the analysis was run on (data transformations only)
