"""Composition root: the only place where adapters are chosen and wired into use cases.

``build_container`` creates what is shared by every request (database, importers, kernels, account use cases);
``Container.for_user`` creates the use cases of *one signed-in user*, built on repositories that already carry
that user's id. Nothing downstream can therefore read or write another user's data by mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

from sigma.application.use_cases.accounts import (
    Authenticate,
    ChangeOwnPassword,
    CreateFirstAdmin,
    CreateUser,
    DeleteUser,
    GetSessionUser,
    GetSetupState,
    ListUsers,
    ResetPassword,
    UpdateProfile,
    UpdateUser,
)
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
    LoadSampleData,
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
from sigma.application.use_cases.workspace import ClearWorkspace, GetSyncState, GetWorkspace, SaveWorkspace
from sigma.config import PROJECT_ROOT, Config
from sigma.domain.users import User
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from sigma.infrastructure.importers.staging import InMemoryImportStaging, OwnedImportStaging
from sigma.infrastructure.kernel.jupyter_executor import JupyterKernelPool
from sigma.infrastructure.persistence.file_notebook_store import FileNotebookStore
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository
from sigma.infrastructure.persistence.sqlite.result_repository import SqliteResultRepository
from sigma.infrastructure.persistence.sqlite.user_repository import SqliteUserRepository
from sigma.infrastructure.persistence.sqlite.workspace_repository import (
    SqliteSyncRepository,
    SqliteWorkspaceRepository,
)
from sigma.infrastructure.persistence.user_data import SqliteLegacyDataAdopter, UserDataPurger
from sigma.infrastructure.security.login_throttle import InMemoryLoginThrottle
from sigma.infrastructure.security.password_hasher import WerkzeugPasswordHasher


@dataclass
class UserServices:
    """Every use case a signed-in user can run, already bound to that user's data."""

    user: User
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
    load_sample_data: LoadSampleData
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
    # workspace and live synchronisation
    get_workspace: GetWorkspace
    save_workspace: SaveWorkspace
    clear_workspace: ClearWorkspace
    get_sync_state: GetSyncState
    # account
    change_own_password: ChangeOwnPassword
    update_profile: UpdateProfile


class Container:
    """Shared infrastructure plus the account use cases (which act across users)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.database = Database(config.db_path)
        self.importers = DefaultImporterRegistry()
        self._staging = InMemoryImportStaging()
        self.kernels = JupyterKernelPool(
            project_root=PROJECT_ROOT,
            db_path=config.db_path,
            work_dir=config.notebooks_dir,
            idle_minutes=config.kernel_idle_minutes,
            max_per_owner=config.max_kernels_per_user,
            max_total=config.max_kernels_total,
        )
        self.users = SqliteUserRepository(self.database)
        hasher = WerkzeugPasswordHasher()
        throttle = InMemoryLoginThrottle()
        adopter = SqliteLegacyDataAdopter(self.database, config.notebooks_dir)
        purger = UserDataPurger(self.database, config.notebooks_dir, self.kernels)
        self.get_setup_state = GetSetupState(self.users)
        self.create_first_admin = CreateFirstAdmin(self.users, hasher, adopter)
        self.authenticate = Authenticate(self.users, hasher, throttle)
        self.get_session_user = GetSessionUser(self.users)
        self.list_users = ListUsers(self.users)
        self.create_user = CreateUser(self.users, hasher)
        self.update_user = UpdateUser(self.users)
        self.reset_password = ResetPassword(self.users, hasher)
        self.delete_user = DeleteUser(self.users, purger)
        self._hasher = hasher

    def for_user(self, user: User) -> UserServices:
        owner = str(user.id)
        datasets = SqliteDatasetRepository(self.database, user.id)
        results = SqliteResultRepository(self.database, user.id)
        notebooks = FileNotebookStore(self.config.notebooks_dir / owner)
        staging = OwnedImportStaging(self._staging, owner)
        workspace = SqliteWorkspaceRepository(self.database, user.id)
        sync = SqliteSyncRepository(self.database, user.id)
        return UserServices(
            user=user,
            inspect_source=InspectSource(self.importers),
            preview_import=PreviewImport(self.importers, staging),
            commit_import=CommitImport(staging, datasets),
            list_datasets=ListDatasets(datasets),
            get_dataset_info=GetDatasetInfo(datasets),
            preview_dataset=PreviewDataset(datasets),
            rename_dataset=RenameDataset(datasets),
            delete_dataset=DeleteDataset(datasets),
            change_column_types=ChangeColumnTypes(datasets),
            load_sample_data=LoadSampleData(self.importers, datasets, self.config.sample_dir),
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
            delete_notebook=DeleteNotebook(notebooks, self.kernels, owner),
            export_notebook=ExportNotebook(notebooks),
            execute_code=ExecuteCode(notebooks, self.kernels, self.config.cell_timeout_seconds, owner),
            control_kernel=ControlKernel(notebooks, self.kernels, owner),
            get_workspace=GetWorkspace(workspace),
            save_workspace=SaveWorkspace(workspace),
            clear_workspace=ClearWorkspace(workspace),
            get_sync_state=GetSyncState(workspace, sync, notebooks),
            change_own_password=ChangeOwnPassword(self.users, self._hasher),
            update_profile=UpdateProfile(self.users),
        )


def build_container(config: Config) -> Container:
    return Container(config)
