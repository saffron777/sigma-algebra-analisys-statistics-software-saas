"""The SQL implementation of ``Backend``: every repository of one database, whichever relational engine it is."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text

from sigma.application.ports import CodeExecutor, NotebookStore
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.file_notebook_store import FileNotebookStore
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.ai_repositories import SqlAiUsageRepository, SqlSettingsRepository
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.import_draft_store import SqlImportDraftStore, purge_stale_drafts
from sigma.infrastructure.persistence.sql.result_repository import SqlResultRepository
from sigma.infrastructure.persistence.sql.template_repository import SqlTemplateRepository
from sigma.infrastructure.persistence.sql.user_data import SqlLegacyDataAdopter, SqlUserDataPurger
from sigma.infrastructure.persistence.sql.user_repository import SqlUserRepository
from sigma.infrastructure.persistence.sql.workspace_repository import SqlSyncRepository, SqlWorkspaceRepository

_VERSION_SQL = {
    "sqlite": "SELECT sqlite_version()",
    "mssql": "SELECT @@VERSION",
    "mysql": "SELECT VERSION()",
    "mariadb": "SELECT VERSION()",
    "postgresql": "SELECT version()",
    "oracle": "SELECT version FROM product_component_version FETCH FIRST 1 ROWS ONLY",
}


class SqlBackend:
    kind = "sql"

    def __init__(self, database: Database) -> None:
        self.database = database
        self.settings: DatabaseSettings = database.settings

    # ------------------------------------------------------------------ repositories
    def users(self) -> SqlUserRepository:
        return SqlUserRepository(self.database)

    def datasets(self, user_id: int) -> SqlDatasetRepository:
        return SqlDatasetRepository(self.database, user_id)

    def results(self, user_id: int) -> SqlResultRepository:
        return SqlResultRepository(self.database, user_id)

    def workspace(self, user_id: int) -> SqlWorkspaceRepository:
        return SqlWorkspaceRepository(self.database, user_id)

    def sync(self, user_id: int) -> SqlSyncRepository:
        return SqlSyncRepository(self.database, user_id)

    def drafts(self, user_id: int) -> SqlImportDraftStore:
        return SqlImportDraftStore(self.database, user_id)

    def templates(self, user_id: int) -> SqlTemplateRepository:
        return SqlTemplateRepository(self.database, user_id)

    def settings_store(self) -> SqlSettingsRepository:
        return SqlSettingsRepository(self.database)

    def ai_usage(self) -> SqlAiUsageRepository:
        return SqlAiUsageRepository(self.database)

    def notebooks(self, user_id: int, notebooks_root: Path) -> NotebookStore:
        """SQLite keeps plain ``.ipynb`` files (they open in Jupyter); other engines keep notebooks in the database."""
        return FileNotebookStore(Path(notebooks_root) / str(user_id))

    # ------------------------------------------------------------------ account-level operations
    @property
    def notebook_files(self) -> bool:
        return True

    def adopter(self, notebooks_root: Path) -> SqlLegacyDataAdopter:
        return SqlLegacyDataAdopter(self.database, notebooks_root, self.notebook_files)

    def purger(self, notebooks_root: Path, kernels: CodeExecutor) -> SqlUserDataPurger:
        return SqlUserDataPurger(self.database, notebooks_root, kernels, self.notebook_files)

    def purge_stale_drafts(self, max_age_days: int) -> int:
        return purge_stale_drafts(self.database, max_age_days)

    # ------------------------------------------------------------------ introspection
    def describe(self) -> dict[str, Any]:
        """What the administration page shows about the live database (never credentials)."""
        sql = _VERSION_SQL.get(self.database.dialect)
        with self.database.begin() as conn:
            version = str(conn.execute(text(sql)).scalar()).splitlines()[0][:120] if sql else ""
            counts = {
                "users": conn.execute(select(func.count()).select_from(schema.users)).scalar_one(),
                "datasets": conn.execute(select(func.count()).select_from(schema.datasets)).scalar_one(),
                "results": conn.execute(select(func.count()).select_from(schema.saved_results)).scalar_one(),
            }
        return {"engine": self.settings.engine, "server_version": version, "counts": counts}

    def close(self) -> None:
        self.database.dispose()
