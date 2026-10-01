"""Composition root: the only place where adapters are chosen and wired into use cases."""

from __future__ import annotations

from dataclasses import dataclass

from sigma.application.use_cases.analysis import (
    CalculateDistribution,
    RunAnova,
    RunDescriptive,
    RunHypothesisTest,
    RunRegression,
)
from sigma.application.use_cases.datasets import (
    ChangeColumnTypes,
    CommitImport,
    DeleteDataset,
    GetDatasetInfo,
    InspectSource,
    ListDatasets,
    PreviewDataset,
    PreviewImport,
    RenameDataset,
)
from sigma.application.use_cases.notebooks import (
    ControlKernel,
    CreateNotebook,
    DeleteNotebook,
    ExecuteCode,
    ExportNotebook,
    GetNotebook,
    ImportNotebook,
    ListNotebooks,
    RenameNotebook,
    SaveNotebook,
)
from sigma.application.use_cases.results import (
    DeleteResult,
    ExportResult,
    GetResult,
    ListResults,
    SaveResult,
    UpdateResult,
)
from sigma.application.use_cases.survey import (
    CalculateSampleSize,
    DrawSurveySample,
    EstimateSurvey,
    RunCrosstab,
    RunLikert,
)
from sigma.config import PROJECT_ROOT, Config
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from sigma.infrastructure.importers.staging import InMemoryImportStaging
from sigma.infrastructure.kernel.jupyter_executor import JupyterKernelPool
from sigma.infrastructure.persistence.file_notebook_store import FileNotebookStore
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository
from sigma.infrastructure.persistence.sqlite.result_repository import SqliteResultRepository


@dataclass
class Container:
    """Every use case the web layer can call, already wired to its adapters."""

    config: Config
    # datasets
    inspect_source: InspectSource
    preview_import: PreviewImport
    commit_import: CommitImport
    list_datasets: ListDatasets
    get_dataset_info: GetDatasetInfo
    preview_dataset: PreviewDataset
    rename_dataset: RenameDataset
    delete_dataset: DeleteDataset
    change_column_types: ChangeColumnTypes
    # analyses
    run_descriptive: RunDescriptive
    run_hypothesis_test: RunHypothesisTest
    run_regression: RunRegression
    run_anova: RunAnova
    calculate_distribution: CalculateDistribution
    # surveys
    calculate_sample_size: CalculateSampleSize
    draw_survey_sample: DrawSurveySample
    estimate_survey: EstimateSurvey
    run_crosstab: RunCrosstab
    run_likert: RunLikert
    # saved results
    save_result: SaveResult
    list_results: ListResults
    get_result: GetResult
    update_result: UpdateResult
    delete_result: DeleteResult
    export_result: ExportResult
    # console / notebooks
    list_notebooks: ListNotebooks
    create_notebook: CreateNotebook
    import_notebook: ImportNotebook
    get_notebook: GetNotebook
    save_notebook: SaveNotebook
    rename_notebook: RenameNotebook
    delete_notebook: DeleteNotebook
    export_notebook: ExportNotebook
    execute_code: ExecuteCode
    control_kernel: ControlKernel
    kernels: JupyterKernelPool


def build_container(config: Config) -> Container:
    database = Database(config.db_path)
    datasets = SqliteDatasetRepository(database)
    results = SqliteResultRepository(database)
    importers = DefaultImporterRegistry()
    staging = InMemoryImportStaging()
    notebooks = FileNotebookStore(config.notebooks_dir)
    kernels = JupyterKernelPool(
        project_root=PROJECT_ROOT,
        db_path=config.db_path,
        work_dir=config.notebooks_dir,
        idle_minutes=config.kernel_idle_minutes,
    )
    return Container(
        config=config,
        inspect_source=InspectSource(importers),
        preview_import=PreviewImport(importers, staging),
        commit_import=CommitImport(staging, datasets),
        list_datasets=ListDatasets(datasets),
        get_dataset_info=GetDatasetInfo(datasets),
        preview_dataset=PreviewDataset(datasets),
        rename_dataset=RenameDataset(datasets),
        delete_dataset=DeleteDataset(datasets),
        change_column_types=ChangeColumnTypes(datasets),
        run_descriptive=RunDescriptive(datasets),
        run_hypothesis_test=RunHypothesisTest(datasets),
        run_regression=RunRegression(datasets),
        run_anova=RunAnova(datasets),
        calculate_distribution=CalculateDistribution(),
        calculate_sample_size=CalculateSampleSize(datasets),
        draw_survey_sample=DrawSurveySample(datasets),
        estimate_survey=EstimateSurvey(datasets),
        run_crosstab=RunCrosstab(datasets),
        run_likert=RunLikert(datasets),
        save_result=SaveResult(results, datasets),
        list_results=ListResults(results),
        get_result=GetResult(results),
        update_result=UpdateResult(results),
        delete_result=DeleteResult(results),
        export_result=ExportResult(results),
        list_notebooks=ListNotebooks(notebooks),
        create_notebook=CreateNotebook(notebooks),
        import_notebook=ImportNotebook(notebooks),
        get_notebook=GetNotebook(notebooks),
        save_notebook=SaveNotebook(notebooks),
        rename_notebook=RenameNotebook(notebooks),
        delete_notebook=DeleteNotebook(notebooks, kernels),
        export_notebook=ExportNotebook(notebooks),
        execute_code=ExecuteCode(notebooks, kernels, config.cell_timeout_seconds),
        control_kernel=ControlKernel(notebooks, kernels),
        kernels=kernels,
    )
