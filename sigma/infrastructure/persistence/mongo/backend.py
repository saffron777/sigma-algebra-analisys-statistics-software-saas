"""The MongoDB implementation of ``Backend``: the same repositories as the SQL engines, on document collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sigma.application.ports import CodeExecutor, NotebookStore
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.mongo.accounts import (
    MongoAiUsageRepository,
    MongoSettingsRepository,
    MongoUserRepository,
    remove_user_rows,
)
from sigma.infrastructure.persistence.mongo.core import MongoCore
from sigma.infrastructure.persistence.mongo.data import (
    MongoDatasetRepository,
    MongoResultRepository,
    MongoSyncRepository,
    MongoWorkspaceRepository,
)
from sigma.infrastructure.persistence.mongo.documents import (
    MongoImportDraftStore,
    MongoNotebookStore,
    MongoTemplateRepository,
    purge_stale_drafts,
)
from sigma.infrastructure.persistence.mongo.transfer import MongoTransfer


class _NoLegacyData:
    """MongoDB never held the single-user data of the first Sigma versions: nothing to adopt."""

    def adopt(self, user_id: int) -> dict[str, int]:
        return {"datasets": 0, "results": 0, "notebooks": 0}


class _MongoPurger:
    def __init__(self, core: MongoCore, kernels: CodeExecutor) -> None:
        self._core = core
        self._kernels = kernels

    def purge(self, user_id: int) -> None:
        self._kernels.shutdown_owner(str(user_id))
        remove_user_rows(self._core, user_id)


class MongoBackend:
    kind = "document"
    notebook_files = False

    def __init__(self, core: MongoCore) -> None:
        self.core = core
        self.settings: DatabaseSettings = core.settings
        self.database = None  # there is no SQL handle

    def users(self) -> MongoUserRepository:
        return MongoUserRepository(self.core)

    def datasets(self, user_id: int) -> MongoDatasetRepository:
        return MongoDatasetRepository(self.core, user_id)

    def results(self, user_id: int) -> MongoResultRepository:
        return MongoResultRepository(self.core, user_id)

    def workspace(self, user_id: int) -> MongoWorkspaceRepository:
        return MongoWorkspaceRepository(self.core, user_id)

    def sync(self, user_id: int) -> MongoSyncRepository:
        return MongoSyncRepository(self.core, user_id)

    def drafts(self, user_id: int) -> MongoImportDraftStore:
        return MongoImportDraftStore(self.core, user_id)

    def templates(self, user_id: int) -> MongoTemplateRepository:
        return MongoTemplateRepository(self.core, user_id)

    def settings_store(self) -> MongoSettingsRepository:
        return MongoSettingsRepository(self.core)

    def ai_usage(self) -> MongoAiUsageRepository:
        return MongoAiUsageRepository(self.core)

    def notebooks(self, user_id: int, notebooks_root: Path) -> NotebookStore:
        return MongoNotebookStore(self.core, user_id)

    def adopter(self, notebooks_root: Path) -> _NoLegacyData:
        return _NoLegacyData()

    def purger(self, notebooks_root: Path, kernels: CodeExecutor) -> _MongoPurger:
        return _MongoPurger(self.core, kernels)

    def purge_stale_drafts(self, max_age_days: int) -> int:
        return purge_stale_drafts(self.core, max_age_days)

    def transfer(self, notebooks_root: Path) -> MongoTransfer:
        return MongoTransfer(self.core, notebooks_root)

    def describe(self) -> dict[str, Any]:
        info = self.core.client.server_info()
        return {
            "engine": "mongodb",
            "server_version": f"MongoDB {info.get('version', '')}",
            "counts": {
                "users": self.core.col("users").count_documents({}),
                "datasets": self.core.col("datasets").count_documents({}),
                "results": self.core.col("saved_results").count_documents({}),
            },
        }

    def close(self) -> None:
        self.core.close()
