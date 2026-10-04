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
from sigma.application.use_cases.ai import (
    GetAiAdmin,
    GetAiStatus,
    InterpretResult,
    TestAiConnection,
    UpdateAiSettings,
)
from sigma.application.use_cases.analysis import (
    CalculateDistribution,
    RunAnova,
    RunDescriptive,
    RunHypothesisTest,
    RunRegression,
)
from sigma.application.use_cases.database import ApplyDatabase, GetDatabaseStatus, GetMigrationStatus, TestDatabase
from sigma.application.use_cases.datasets import (
    ChangeColumnTypes,
    CommitImport,
    DeleteDataset,
    DiscardImportDraft,
    GetDatasetInfo,
    GetImportDraft,
    InspectSource,
    ListDatasets,
    ListExternalDatabases,
    ListImportDrafts,
    LoadSampleData,
    PreviewDataset,
    PreviewImport,
    RenameDataset,
    UploadImportFile,
)
from sigma.application.use_cases.exports import (
    ExportDatasetReport,
    ExporterRegistry,
    ExportNotebookReport,
    ExportReport,
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
from sigma.application.use_cases.run_analysis import GetAnalysisCode, ListAnalyses, RunAnalysis
from sigma.application.use_cases.survey import (
    CalculateSampleSize,
    DrawSurveySample,
    EstimateSurvey,
    RunCrosstab,
    RunLikert,
)
from sigma.application.use_cases.templates import (
    CreateTemplate,
    DeleteTemplate,
    GetTemplate,
    GetTemplateLogo,
    ListTemplates,
    RemoveTemplateLogo,
    SetDefaultTemplate,
    UpdateTemplate,
    UploadTemplateLogo,
)
from sigma.application.use_cases.workspace import ClearWorkspace, GetSyncState, GetWorkspace, SaveWorkspace
from sigma.config import PROJECT_ROOT, Config
from sigma.domain.users import User
from sigma.infrastructure.ai.anthropic_gateway import AnthropicGateway
from sigma.infrastructure.ai.settings import AiSettings
from sigma.infrastructure.compute.process_runner import AnalysisRunnerImpl
from sigma.infrastructure.exporters.docx_exporter import DocxExporter
from sigma.infrastructure.exporters.pdf_exporter import PdfExporter
from sigma.infrastructure.exporters.xlsx_exporter import XlsxExporter
from sigma.infrastructure.imaging import PillowImageProcessor
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from sigma.infrastructure.kernel.jupyter_executor import JupyterKernelPool
from sigma.infrastructure.persistence.administration import DatabaseAdmin
from sigma.infrastructure.persistence.backend import create_backend
from sigma.infrastructure.persistence.connection_tester import ConnectionTester
from sigma.infrastructure.security.login_throttle import InMemoryLoginThrottle
from sigma.infrastructure.security.password_hasher import WerkzeugPasswordHasher


def _read_default_logo() -> bytes | None:
    """The Sigma mark (PNG) that heads reports made without a template; reports still work if the file is missing."""
    path = PROJECT_ROOT / "sigma" / "presentation" / "web" / "static" / "img" / "logo-mark-256.png"
    try:
        return path.read_bytes()
    except OSError:
        return None


@dataclass
class UserServices:
    """Every use case a signed-in user can run, already bound to that user's data."""

    user: User
    # datasets
    inspect_source: InspectSource
    list_external_databases: ListExternalDatabases
    upload_import_file: UploadImportFile
    preview_import: PreviewImport
    list_import_drafts: ListImportDrafts
    get_import_draft: GetImportDraft
    discard_import_draft: DiscardImportDraft
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
    run_analysis: RunAnalysis
    list_analyses: ListAnalyses
    get_analysis_code: GetAnalysisCode
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
    # AI interpretation
    get_ai_status: GetAiStatus
    interpret_result: InterpretResult
    # report templates and exports
    list_templates: ListTemplates
    get_template: GetTemplate
    create_template: CreateTemplate
    update_template: UpdateTemplate
    delete_template: DeleteTemplate
    set_default_template: SetDefaultTemplate
    upload_template_logo: UploadTemplateLogo
    remove_template_logo: RemoveTemplateLogo
    get_template_logo: GetTemplateLogo
    export_report: ExportReport
    export_dataset_report: ExportDatasetReport
    export_notebook_report: ExportNotebookReport
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
        self.database_settings = config.database
        self.backend = create_backend(config.database)
        self.database = getattr(self.backend, "database", None)  # the SQL handle (None for MongoDB)
        self.importers = DefaultImporterRegistry(allowed_hosts=config.external_db_hosts)
        self.analysis_runner = AnalysisRunnerImpl(config.analysis_timeout_seconds, config.analysis_max_parallel)
        self.exporters = ExporterRegistry([PdfExporter(), DocxExporter(), XlsxExporter()])
        self.images = PillowImageProcessor()
        self.default_logo = _read_default_logo()
        self.backend.purge_stale_drafts(config.draft_retention_days)  # pending imports nobody touched for weeks
        self.kernels = JupyterKernelPool(
            project_root=PROJECT_ROOT,
            database_url=config.database.to_url(),
            work_dir=config.notebooks_dir,
            idle_minutes=config.kernel_idle_minutes,
            max_per_owner=config.max_kernels_per_user,
            max_total=config.max_kernels_total,
        )
        self.users = self.backend.users()
        self.ai_settings = AiSettings(self.backend.settings_store(), config.instance_dir)
        self.ai_usage = self.backend.ai_usage()
        self.ai_gateway = AnthropicGateway(self.ai_settings)
        self.get_ai_admin = GetAiAdmin(self.ai_settings, self.ai_usage, config.ai_requests_per_hour)
        self.update_ai_settings = UpdateAiSettings(self.ai_settings)
        self.database_admin = DatabaseAdmin(
            config.instance_dir, config.db_path, config.database, self.backend, ConnectionTester()
        )
        self.get_database_status = GetDatabaseStatus(self.database_admin)
        self.test_database = TestDatabase(self.database_admin)
        self.apply_database = ApplyDatabase(self.database_admin)
        self.get_migration_status = GetMigrationStatus(self.database_admin)
        self.test_ai_connection = TestAiConnection(self.ai_gateway, self.ai_settings)
        hasher = WerkzeugPasswordHasher()
        throttle = InMemoryLoginThrottle()
        adopter = self.backend.adopter(config.notebooks_dir)
        purger = self.backend.purger(config.notebooks_dir, self.kernels)
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

    def close(self) -> None:
        """Stop the kernels and release the database connections (before another container takes over)."""
        self.kernels.shutdown_all()
        self.backend.close()

    def for_user(self, user: User) -> UserServices:
        owner = str(user.id)
        datasets = self.backend.datasets(user.id)
        results = self.backend.results(user.id)
        notebooks = self.backend.notebooks(user.id, self.config.notebooks_dir)
        drafts = self.backend.drafts(user.id)
        workspace = self.backend.workspace(user.id)
        sync = self.backend.sync(user.id)
        templates = self.backend.templates(user.id)
        return UserServices(
            user=user,
            inspect_source=InspectSource(self.importers, drafts),
            list_external_databases=ListExternalDatabases(self.importers),
            upload_import_file=UploadImportFile(self.importers, drafts),
            preview_import=PreviewImport(self.importers, drafts),
            list_import_drafts=ListImportDrafts(drafts),
            get_import_draft=GetImportDraft(drafts),
            discard_import_draft=DiscardImportDraft(drafts),
            commit_import=CommitImport(drafts, datasets),
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
            run_analysis=RunAnalysis(datasets, self.analysis_runner),
            list_analyses=ListAnalyses(),
            get_analysis_code=GetAnalysisCode(datasets),
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
            get_ai_status=GetAiStatus(self.ai_settings, self.ai_usage, user, self.config.ai_requests_per_hour),
            interpret_result=InterpretResult(
                self.ai_gateway, self.ai_settings, self.ai_usage, datasets, user, self.config.ai_requests_per_hour
            ),
            list_templates=ListTemplates(templates),
            get_template=GetTemplate(templates),
            create_template=CreateTemplate(templates),
            update_template=UpdateTemplate(templates),
            delete_template=DeleteTemplate(templates),
            set_default_template=SetDefaultTemplate(templates),
            upload_template_logo=UploadTemplateLogo(templates, self.images),
            remove_template_logo=RemoveTemplateLogo(templates),
            get_template_logo=GetTemplateLogo(templates),
            export_report=ExportReport(self.exporters, templates, self.images, self.default_logo),
            export_dataset_report=ExportDatasetReport(self.exporters, templates, datasets, self.default_logo),
            export_notebook_report=ExportNotebookReport(self.exporters, templates, notebooks, self.default_logo),
            get_workspace=GetWorkspace(workspace),
            save_workspace=SaveWorkspace(workspace),
            clear_workspace=ClearWorkspace(workspace, drafts),
            get_sync_state=GetSyncState(workspace, sync, notebooks),
            change_own_password=ChangeOwnPassword(self.users, self._hasher),
            update_profile=UpdateProfile(self.users),
        )


def build_container(config: Config) -> Container:
    return Container(config)
