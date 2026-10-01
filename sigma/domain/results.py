"""Result value objects returned by the statistical domain services.

Fields created with :func:`transient` carry raw arrays used only to draw charts; they are
skipped when a result is serialised to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sigma.domain.distributions.spec import DistributionRef

SERIALIZE_KEY = "serialize"


def transient(default_factory=dict):
    """A dataclass field that is excluded from JSON serialisation."""
    return field(default_factory=default_factory, repr=False, metadata={SERIALIZE_KEY: False})


@dataclass
class TableData:
    title: str
    columns: list[str]
    rows: list[list[Any]]
    note: str = ""


# --------------------------------------------------------------------------- descriptive
@dataclass
class NumericSummary:
    name: str
    group: str | None
    n: int
    n_missing: int
    mean: float
    median: float
    mode: float | None
    std: float
    variance: float
    minimum: float
    maximum: float
    range: float
    q1: float
    q3: float
    iqr: float
    cv: float | None
    skewness: float
    kurtosis: float
    sem: float
    ci_low: float
    ci_high: float
    confidence: float
    percentiles: dict[str, float]


@dataclass
class FrequencyRow:
    value: str
    count: int
    percent: float
    cumulative_count: int
    cumulative_percent: float


@dataclass
class FrequencyTable:
    name: str
    n: int
    n_missing: int
    rows: list[FrequencyRow]


@dataclass
class NormalityTest:
    name: str
    group: str | None
    test: str
    statistic: float
    p_value: float
    n: int
    distribution: DistributionRef | None = None


@dataclass
class DistributionFit:
    column: str
    distribution: str
    label: str
    params: dict[str, float]
    ks_statistic: float
    ks_p_value: float
    aic: float


@dataclass
class CorrelationResult:
    method: str
    columns: list[str]
    matrix: list[list[float | None]]
    p_values: list[list[float | None]]
    n_matrix: list[list[int]]


@dataclass
class DescriptiveResult:
    confidence: float
    group_by: str | None
    numeric: list[NumericSummary]
    categorical: list[FrequencyTable]
    normality: list[NormalityTest]
    fits: list[DistributionFit]
    correlation: CorrelationResult | None
    plot_data: dict[str, Any] = transient()


# --------------------------------------------------------------------------- hypothesis
@dataclass
class HypothesisTestResult:
    test_id: str
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic_name: str
    statistic: float
    df: float | list[float] | None
    p_value: float
    alpha: float
    tail: str
    reject_null: bool
    conclusion: str
    critical_values: list[float] = field(default_factory=list)
    estimate_label: str | None = None
    estimate: float | None = None
    confidence_interval: list[float] | None = None
    confidence: float | None = None
    effect_size_label: str | None = None
    effect_size: float | None = None
    distribution: DistributionRef | None = None
    sample_info: dict[str, Any] = field(default_factory=dict)
    tables: list[TableData] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    plot_data: dict[str, Any] = transient()


# --------------------------------------------------------------------------- regression
@dataclass
class Coefficient:
    term: str
    estimate: float
    std_error: float
    t_value: float
    p_value: float
    ci_low: float
    ci_high: float
    vif: float | None = None


@dataclass
class AnovaRow:
    source: str
    sum_sq: float
    df: float
    mean_sq: float | None
    f_value: float | None
    p_value: float | None
    eta_sq: float | None = None
    partial_eta_sq: float | None = None
    omega_sq: float | None = None


@dataclass
class DiagnosticTest:
    name: str
    statistic: float
    p_value: float | None
    interpretation: str
    passed: bool = True


@dataclass
class InfluentialPoint:
    row: int
    observed: float
    fitted: float
    residual: float
    leverage: float
    cooks_distance: float


@dataclass
class PredictionRow:
    inputs: dict[str, Any]
    prediction: float
    mean_ci_low: float
    mean_ci_high: float
    pred_ci_low: float
    pred_ci_high: float


@dataclass
class RegressionResult:
    response: str
    predictors: list[str]
    is_simple: bool
    has_intercept: bool
    confidence: float
    equation: str
    n_obs: int
    n_dropped: int
    df_model: float
    df_resid: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float | None
    f_p_value: float | None
    residual_std_error: float
    aic: float
    bic: float
    log_likelihood: float
    coefficients: list[Coefficient]
    anova: list[AnovaRow]
    diagnostics: list[DiagnosticTest]
    influential: list[InfluentialPoint]
    predictions: list[PredictionRow]
    f_distribution: DistributionRef | None
    t_distribution: DistributionRef | None
    notes: list[str] = field(default_factory=list)
    plot_data: dict[str, Any] = transient()


# --------------------------------------------------------------------------- anova
@dataclass
class GroupStats:
    group: str
    n: int
    mean: float
    std: float
    sem: float
    ci_low: float
    ci_high: float


@dataclass
class AssumptionCheck:
    name: str
    test: str
    statistic: float
    p_value: float
    passed: bool
    note: str


@dataclass
class PostHocRow:
    group1: str
    group2: str
    mean_diff: float
    p_adj: float
    ci_low: float
    ci_high: float
    reject: bool


@dataclass
class FTestRef:
    label: str
    statistic: float
    df_num: float
    df_den: float
    p_value: float


@dataclass
class AnovaResult:
    design: str
    response: str
    factors: list[str]
    alpha: float
    n_obs: int
    n_dropped: int
    table: list[AnovaRow]
    groups: list[GroupStats]
    assumptions: list[AssumptionCheck]
    post_hoc: list[PostHocRow]
    post_hoc_method: str | None
    f_tests: list[FTestRef]
    welch: dict[str, float] | None = None
    kruskal: dict[str, float] | None = None
    ss_type: int | None = None
    plot_data: dict[str, Any] = transient()


# --------------------------------------------------------------------------- distributions
@dataclass
class DistributionCalcResult:
    distribution: str
    label: str
    kind: str
    params: dict[str, float]
    mode: str
    values: dict[str, float | None]
    moments: dict[str, float]
    critical_values: list[float] = field(default_factory=list)
    tail: str | None = None
    marker: float | None = None
    bounds: list[float] | None = None
    alpha: float | None = None


# --------------------------------------------------------------------------- surveys / markets
@dataclass
class StratumAllocation:
    stratum: str
    population: int
    share: float
    sample_size: int


@dataclass
class SampleSizeResult:
    kind: str
    confidence: float
    margin_of_error: float
    z: float
    n_infinite: float
    n_adjusted: float
    n_final: int
    design_effect: float
    population_size: int | None
    inputs: dict[str, float | None]
    allocation: list[StratumAllocation] = field(default_factory=list)
    allocation_method: str | None = None
    curve: list[list[float]] = field(default_factory=list)


@dataclass
class SampleDrawResult:
    design: str
    population_size: int
    sample_size: int
    seed: int | None
    interval: float | None
    notes: list[str]
    summary: list[TableData]
    sample: Any = transient(lambda: None)  # pandas DataFrame, stored by the use case


@dataclass
class EstimateRow:
    label: str
    n: int
    estimate: float
    std_error: float
    margin_of_error: float
    ci_low: float
    ci_high: float
    method: str
    count: float | None = None


@dataclass
class SurveyEstimateResult:
    kind: str
    variable: str
    design: str
    confidence: float
    n: int
    population_size: int | None
    design_effect: float | None
    effective_n: float | None
    rows: list[EstimateRow]
    notes: list[str] = field(default_factory=list)


@dataclass
class CrosstabResult:
    row_variable: str
    col_variable: str
    row_labels: list[str]
    col_labels: list[str]
    observed: list[list[float]]
    expected: list[list[float]]
    row_percent: list[list[float]]
    col_percent: list[list[float]]
    total_percent: list[list[float]]
    standardized_residuals: list[list[float]]
    chi2: float
    df: int
    p_value: float
    cramers_v: float
    n: float
    notes: list[str] = field(default_factory=list)
    distribution: DistributionRef | None = None


@dataclass
class LikertResult:
    items: list[str]
    levels: list[str]
    counts: list[list[int]]
    percents: list[list[float]]
    means: list[float]
    top2box: list[float]
    cronbach_alpha: float | None
    n_items: int
    n_complete: int
