"""Use cases for survey / market-research analysis."""

from __future__ import annotations

from sigma.application.dto import (
    CrosstabCommand,
    DrawSampleCommand,
    EstimateCommand,
    LikertCommand,
    SampleSizeCommand,
)
from sigma.application.ports import DatasetRepository
from sigma.application.use_cases.datasets import clean_dataset_name
from sigma.domain.datasets import DatasetInfo
from sigma.domain.errors import InvalidInputError
from sigma.domain.results import (
    CrosstabResult,
    LikertResult,
    SampleDrawResult,
    SampleSizeResult,
    SurveyEstimateResult,
)
from sigma.domain.statistics.survey_estimation import estimate_survey
from sigma.domain.statistics.survey_sampling import (
    allocate_strata,
    draw_sample,
    sample_size_mean,
    sample_size_proportion,
)
from sigma.domain.statistics.survey_tabulation import crosstab, likert_summary


class CalculateSampleSize:
    """Sample size for a proportion or a mean, with an optional stratified allocation."""

    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: SampleSizeCommand) -> SampleSizeResult:
        if cmd.kind == "proportion":
            result = sample_size_proportion(
                cmd.confidence, cmd.margin_of_error, cmd.proportion, cmd.population_size, cmd.design_effect
            )
        elif cmd.kind == "mean":
            if cmd.std_dev is None:
                raise InvalidInputError("Indique la desviación estándar esperada.")
            result = sample_size_mean(
                cmd.confidence, cmd.margin_of_error, cmd.std_dev, cmd.population_size, cmd.design_effect
            )
        else:
            raise InvalidInputError("El tipo de cálculo debe ser «proportion» o «mean».")
        if cmd.dataset_id and cmd.strata_column:
            frame = self._datasets.get(cmd.dataset_id).data
            result.allocation = allocate_strata(
                frame, cmd.strata_column, result.n_final, cmd.allocation, cmd.allocation_sd_column
            )
            result.allocation_method = cmd.allocation
        return result


class DrawSurveySample:
    """Selects a random sample from a sampling frame and optionally saves it as a new dataset."""

    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: DrawSampleCommand) -> tuple[SampleDrawResult, DatasetInfo | None]:
        dataset = self._datasets.get(cmd.dataset_id)
        result = draw_sample(
            dataset.data,
            cmd.design,
            cmd.sample_size,
            cmd.seed,
            cmd.strata_column,
            cmd.allocation,
            cmd.sd_column,
            cmd.cluster_column,
            cmd.n_clusters,
        )
        saved = None
        if cmd.save_as:
            meta = {"derived_from": dataset.info.id, "design": cmd.design, "seed": result.seed}
            saved = self._datasets.add(clean_dataset_name(cmd.save_as), result.sample, "sample", meta)
        return result, saved


class EstimateSurvey:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: EstimateCommand) -> SurveyEstimateResult:
        frame = self._datasets.get(cmd.dataset_id).data
        return estimate_survey(
            frame,
            cmd.variable,
            cmd.kind,
            cmd.confidence,
            cmd.methods,
            cmd.population_size,
            cmd.weight_column,
            cmd.strata_column,
            cmd.cluster_column,
            cmd.apply_fpc,
        )


class RunCrosstab:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: CrosstabCommand) -> CrosstabResult:
        return crosstab(self._datasets.get(cmd.dataset_id).data, cmd.row_variable, cmd.col_variable)


class RunLikert:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: LikertCommand) -> LikertResult:
        return likert_summary(self._datasets.get(cmd.dataset_id).data, cmd.items)
