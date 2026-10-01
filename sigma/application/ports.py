"""Ports: interfaces the application layer needs; adapters live in ``infrastructure``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from sigma.domain.datasets import Dataset, DatasetInfo
from sigma.domain.notebooks import ExecutionResult, KernelStatus, NotebookDocument, NotebookSummary
from sigma.domain.saved_results import SavedResult, SavedResultSummary
from sigma.domain.users import User


@dataclass(frozen=True)
class ImportRequest:
    """What a data source adapter needs to read a table."""

    source_type: str
    options: dict[str, Any] = field(default_factory=dict)
    file_name: str | None = None
    file_content: bytes | None = None


@dataclass
class StagedImport:
    """A parsed-but-not-yet-saved import, so the user can review it before committing."""

    frame: pd.DataFrame
    source_type: str
    source_meta: dict[str, Any]
    suggested_name: str
    detected: dict[str, Any] = field(default_factory=dict)


class DatasetImporter(Protocol):
    source_type: str

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return the table and the options that were detected/used (delimiter, sheet...)."""
        ...


class ImporterRegistry(Protocol):
    def get(self, source_type: str) -> DatasetImporter: ...

    def source_types(self) -> list[str]: ...


class ImportStaging(Protocol):
    def put(self, staged: StagedImport) -> str: ...

    def take(self, token: str) -> StagedImport: ...


class DatasetRepository(Protocol):
    def add(self, name: str, frame: pd.DataFrame, source_type: str, source_meta: dict[str, Any]) -> DatasetInfo: ...

    def replace_data(self, dataset_id: str, frame: pd.DataFrame) -> DatasetInfo: ...

    def get(self, dataset_id: str) -> Dataset: ...

    def get_by_name(self, name: str) -> Dataset: ...

    def get_info(self, dataset_id: str) -> DatasetInfo: ...

    def list(self) -> list[DatasetInfo]: ...

    def preview(self, dataset_id: str, limit: int, offset: int) -> pd.DataFrame: ...

    def rename(self, dataset_id: str, name: str) -> DatasetInfo: ...

    def delete(self, dataset_id: str) -> None: ...


class NotebookStore(Protocol):
    def list(self) -> list[NotebookSummary]: ...

    def create(self, name: str, cells: list[dict[str, Any]] | None = None) -> NotebookDocument: ...

    def import_bytes(self, file_name: str, content: bytes) -> NotebookDocument: ...

    def get(self, notebook_id: str) -> NotebookDocument: ...

    def save(
        self, notebook_id: str, cells: list[dict[str, Any]], metadata: dict[str, Any] | None, name: str | None
    ) -> NotebookDocument: ...

    def delete(self, notebook_id: str) -> None: ...

    def export_bytes(self, notebook_id: str) -> bytes: ...


class CodeExecutor(Protocol):
    """Runs code in a persistent, isolated interpreter (one per session id)."""

    def start(self, session_id: str, owner: str = "") -> None: ...

    def execute(self, session_id: str, code: str, timeout_seconds: float, owner: str = "") -> ExecutionResult: ...

    def interrupt(self, session_id: str) -> None: ...

    def restart(self, session_id: str, owner: str = "") -> None: ...

    def shutdown(self, session_id: str) -> None: ...

    def shutdown_owner(self, owner: str) -> None: ...

    def status(self, session_id: str) -> KernelStatus: ...


class UserRepository(Protocol):
    def count(self) -> int: ...

    def count_active_admins(self, excluding: int | None = None) -> int: ...

    def add(
        self,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        can_use_console: bool,
        can_use_ai: bool,
    ) -> User: ...

    def get(self, user_id: int) -> User: ...

    def find_with_hash(self, username: str) -> tuple[User, str] | None: ...

    def list(self) -> list[User]: ...

    def update(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
        can_use_console: bool | None = None,
        can_use_ai: bool | None = None,
    ) -> User: ...

    def set_password(self, user_id: int, password_hash: str) -> User: ...

    def touch_login(self, user_id: int) -> None: ...

    def delete(self, user_id: int) -> None: ...


class PasswordHasher(Protocol):
    dummy_hash: str

    def hash(self, password: str) -> str: ...

    def verify(self, password_hash: str, password: str) -> bool: ...


class LoginThrottle(Protocol):
    def check(self, username: str, ip: str) -> None: ...

    def record_failure(self, username: str, ip: str) -> None: ...

    def reset(self, username: str, ip: str) -> None: ...


class LegacyDataAdopter(Protocol):
    """Hands data created before accounts existed (no owner) to the first administrator."""

    def adopt(self, user_id: int) -> dict[str, int]: ...


class UserDataPurger(Protocol):
    """Removes everything a user owns (datasets and their tables, notebooks, running kernels...)."""

    def purge(self, user_id: int) -> None: ...


class ResultRepository(Protocol):
    def add(self, result: SavedResult) -> SavedResult: ...

    def get(self, result_id: str) -> SavedResult: ...

    def list(
        self, module: str | None, search: str | None, limit: int, offset: int
    ) -> tuple[list[SavedResultSummary], int]: ...

    def update(self, result_id: str, title: str | None, notes: str | None) -> SavedResult: ...

    def delete(self, result_id: str) -> None: ...
