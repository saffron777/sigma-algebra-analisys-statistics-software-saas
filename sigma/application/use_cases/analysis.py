"""Use cases that run statistical analyses over a stored dataset."""

from __future__ import annotations

from sigma.application.dto import (
    AnovaCommand,
    DescriptiveCommand,
    DistributionCommand,
    HypothesisCommand,
    RegressionCommand,
)
from sigma.application.ports import DatasetRepository
from sigma.domain.distributions.calculator import calculate
from sigma.domain.results import (
    AnovaResult,
    DescriptiveResult,
    DistributionCalcResult,
    HypothesisTestResult,
    RegressionResult,
)
from sigma.domain.statistics.anova import run_anova
from sigma.domain.statistics.descriptive import describe
from sigma.domain.statistics.hypothesis import run_hypothesis_test
from sigma.domain.statistics.regression import fit_regression


class RunDescriptive:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: DescriptiveCommand) -> DescriptiveResult:
        frame = self._datasets.get(cmd.dataset_id).data
        return describe(
            frame,
            numeric_columns=cmd.numeric_columns,
            categorical_columns=cmd.categorical_columns,
            group_by=cmd.group_by,
            confidence=cmd.confidence,
            run_normality=cmd.run_normality,
            fit_candidates=cmd.fit_distributions or None,
            correlation_method=cmd.correlation_method,
        )


class RunHypothesisTest:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: HypothesisCommand) -> HypothesisTestResult:
        use_summary = bool(cmd.summary)
        frame = None if use_summary or not cmd.dataset_id else self._datasets.get(cmd.dataset_id).data
        return run_hypothesis_test(
            cmd.test_id,
            frame=frame,
            roles=cmd.roles,
            params=cmd.params,
            summary=cmd.summary if use_summary else None,
            alpha=cmd.alpha,
            tail=cmd.tail,
        )


class RunRegression:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: RegressionCommand) -> RegressionResult:
        frame = self._datasets.get(cmd.dataset_id).data
        return fit_regression(
            frame,
            response=cmd.response,
            predictors=cmd.predictors,
            categorical=cmd.categorical,
            intercept=cmd.intercept,
            confidence=cmd.confidence,
            predict=cmd.predict,
        )


class RunAnova:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, cmd: AnovaCommand) -> AnovaResult:
        frame = self._datasets.get(cmd.dataset_id).data
        return run_anova(
            frame,
            response=cmd.response,
            factors=cmd.factors,
            alpha=cmd.alpha,
            interaction=cmd.interaction,
            ss_type=cmd.ss_type,
            post_hoc=cmd.post_hoc,
        )


class CalculateDistribution:
    def execute(self, cmd: DistributionCommand) -> DistributionCalcResult:
        return calculate(
            cmd.distribution,
            cmd.params,
            cmd.mode,
            x=cmd.x,
            a=cmd.a,
            b=cmd.b,
            q=cmd.q,
            statistic=cmd.statistic,
            tail=cmd.tail,
            alpha=cmd.alpha,
        )
