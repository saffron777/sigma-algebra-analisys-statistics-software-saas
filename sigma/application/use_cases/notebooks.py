"""Use cases for the Python console: notebooks (stored as .ipynb) and kernel execution."""

from __future__ import annotations

from typing import Any

from sigma.application.ports import CodeExecutor, NotebookStore
from sigma.domain.errors import InvalidInputError
from sigma.domain.notebooks import ExecutionResult, KernelStatus, NotebookDocument, NotebookSummary

MAX_NAME_LENGTH = 120
MAX_CODE_LENGTH = 1_000_000
MAX_TIMEOUT_SECONDS = 3600


def clean_notebook_name(name: str | None, fallback: str = "Notebook sin título") -> str:
    return " ".join((name or "").split())[:MAX_NAME_LENGTH] or fallback


class ListNotebooks:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self) -> list[NotebookSummary]:
        return self._store.list()


class CreateNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, name: str | None) -> NotebookDocument:
        return self._store.create(clean_notebook_name(name))


class ImportNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, file_name: str, content: bytes) -> NotebookDocument:
        if not content:
            raise InvalidInputError("El archivo está vacío.")
        return self._store.import_bytes(file_name, content)


class GetNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, notebook_id: str) -> NotebookDocument:
        return self._store.get(notebook_id)


class SaveNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(
        self, notebook_id: str, cells: list[dict[str, Any]], metadata: dict[str, Any] | None, name: str | None
    ) -> NotebookDocument:
        return self._store.save(notebook_id, cells, metadata, clean_notebook_name(name, "") or None)


class RenameNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, notebook_id: str, name: str) -> NotebookDocument:
        document = self._store.get(notebook_id)
        return self._store.save(notebook_id, document.cells, document.metadata, clean_notebook_name(name))


class DeleteNotebook:
    def __init__(self, store: NotebookStore, executor: CodeExecutor) -> None:
        self._store = store
        self._executor = executor

    def execute(self, notebook_id: str) -> None:
        self._store.delete(notebook_id)
        self._executor.shutdown(notebook_id)


class ExportNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, notebook_id: str) -> tuple[str, bytes]:
        document = self._store.get(notebook_id)
        return document.name, self._store.export_bytes(notebook_id)


class ExecuteCode:
    """Runs one cell in the notebook's own kernel (state persists between cells)."""

    def __init__(self, store: NotebookStore, executor: CodeExecutor, default_timeout_seconds: int) -> None:
        self._store = store
        self._executor = executor
        self._default_timeout = default_timeout_seconds

    def execute(self, notebook_id: str, code: str, timeout_seconds: float | None = None) -> ExecutionResult:
        self._store.get(notebook_id)  # fails with NotebookNotFoundError for unknown ids
        if len(code) > MAX_CODE_LENGTH:
            raise InvalidInputError("El código supera el tamaño máximo permitido.")
        timeout = float(timeout_seconds or self._default_timeout)
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise InvalidInputError(f"El tiempo máximo debe estar entre 1 y {MAX_TIMEOUT_SECONDS} segundos.")
        return self._executor.execute(notebook_id, code, timeout)


class ControlKernel:
    """Interrupt, restart or inspect a notebook's kernel."""

    def __init__(self, store: NotebookStore, executor: CodeExecutor) -> None:
        self._store = store
        self._executor = executor

    def start(self, notebook_id: str) -> KernelStatus:
        self._store.get(notebook_id)
        self._executor.start(notebook_id)
        return self._executor.status(notebook_id)

    def interrupt(self, notebook_id: str) -> None:
        self._store.get(notebook_id)
        self._executor.interrupt(notebook_id)

    def restart(self, notebook_id: str) -> None:
        self._store.get(notebook_id)
        self._executor.restart(notebook_id)

    def status(self, notebook_id: str) -> KernelStatus:
        self._store.get(notebook_id)
        return self._executor.status(notebook_id)
