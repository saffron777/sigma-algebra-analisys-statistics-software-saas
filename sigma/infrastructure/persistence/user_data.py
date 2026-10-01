"""Cross-cutting data operations on whole accounts: adopting legacy data and purging a user."""

from __future__ import annotations

import shutil
from pathlib import Path

from sigma.application.ports import CodeExecutor
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository


class SqliteLegacyDataAdopter:
    """Gives the data created before accounts existed (``user_id IS NULL``) to the first administrator."""

    def __init__(self, database: Database, notebooks_root: Path) -> None:
        self._db = database
        self._notebooks_root = Path(notebooks_root)

    def adopt(self, user_id: int) -> dict[str, int]:
        with self._db.connection() as conn:
            datasets = conn.execute("UPDATE datasets SET user_id = ? WHERE user_id IS NULL", (user_id,)).rowcount
            results = conn.execute("UPDATE saved_results SET user_id = ? WHERE user_id IS NULL", (user_id,)).rowcount
        return {"datasets": datasets, "results": results, "notebooks": self._move_notebooks(user_id)}

    def _move_notebooks(self, user_id: int) -> int:
        """Notebooks used to live directly under ``notebooks/``; each user now has ``notebooks/<user_id>/``."""
        if not self._notebooks_root.is_dir():
            return 0
        target = self._notebooks_root / str(user_id)
        moved = 0
        for path in sorted(self._notebooks_root.glob("*.ipynb")):
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            moved += 1
        return moved


class UserDataPurger:
    """Deletes everything a user owns: data tables, notebooks and running kernels (rows go by ON DELETE CASCADE)."""

    def __init__(self, database: Database, notebooks_root: Path, kernels: CodeExecutor) -> None:
        self._db = database
        self._notebooks_root = Path(notebooks_root)
        self._kernels = kernels

    def purge(self, user_id: int) -> None:
        self._kernels.shutdown_owner(str(user_id))
        SqliteDatasetRepository(self._db, user_id).purge()
        shutil.rmtree(self._notebooks_root / str(user_id), ignore_errors=True)
