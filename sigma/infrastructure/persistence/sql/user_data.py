"""Cross-cutting data operations on whole accounts: adopting pre-accounts data and purging a user."""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import update

from sigma.application.ports import CodeExecutor
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.import_draft_store import SqlImportDraftStore


class SqlLegacyDataAdopter:
    """Gives the data created before accounts existed (``user_id IS NULL``) to the first administrator."""

    def __init__(self, database: Database, notebooks_root: Path, notebook_files: bool = True) -> None:
        self._db = database
        self._notebooks_root = Path(notebooks_root)
        self._notebook_files = notebook_files

    def adopt(self, user_id: int) -> dict[str, int]:
        with self._db.begin() as conn:
            datasets = conn.execute(
                update(schema.datasets).where(schema.datasets.c.user_id.is_(None)).values(user_id=user_id)
            ).rowcount
            results = conn.execute(
                update(schema.saved_results).where(schema.saved_results.c.user_id.is_(None)).values(user_id=user_id)
            ).rowcount
        return {"datasets": datasets, "results": results, "notebooks": self._move_notebooks(user_id)}

    def _move_notebooks(self, user_id: int) -> int:
        """Notebooks used to live directly under ``notebooks/``; each user now has ``notebooks/<user_id>/``."""
        if not self._notebook_files or not self._notebooks_root.is_dir():
            return 0
        target = self._notebooks_root / str(user_id)
        moved = 0
        for path in sorted(self._notebooks_root.glob("*.ipynb")):
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            moved += 1
        return moved


class SqlUserDataPurger:
    """Deletes what an account owns that a cascade cannot reach: data tables, notebook files and running kernels."""

    def __init__(
        self, database: Database, notebooks_root: Path, kernels: CodeExecutor, notebook_files: bool = True
    ) -> None:
        self._db = database
        self._notebooks_root = Path(notebooks_root)
        self._kernels = kernels
        self._notebook_files = notebook_files

    def purge(self, user_id: int) -> None:
        self._kernels.shutdown_owner(str(user_id))
        SqlDatasetRepository(self._db, user_id).purge()
        SqlImportDraftStore(self._db, user_id).purge()  # their staged tables are not dropped by the cascade
        if self._notebook_files:
            shutil.rmtree(self._notebooks_root / str(user_id), ignore_errors=True)
