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

    def execute(self, name: str | None, language: str = "python") -> NotebookDocument:
        return self._store.create(clean_notebook_name(name), language=language)


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
        self,
        notebook_id: str,
        cells: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
        name: str | None,
        base_revision: str | None = None,
    ) -> NotebookDocument:
        return self._store.save(notebook_id, cells, metadata, clean_notebook_name(name, "") or None, base_revision)


class RenameNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, notebook_id: str, name: str) -> NotebookDocument:
        document = self._store.get(notebook_id)
        return self._store.save(notebook_id, document.cells, document.metadata, clean_notebook_name(name))


def kernel_key(owner: str, notebook_id: str) -> str:
    """Kernels are keyed by owner *and* notebook, so two users can never share one."""
    return f"{owner}:{notebook_id}"


def kernel_name_for(language: str) -> str:
    """The Jupyter kernelspec name for a notebook language."""
    return "ir" if language == "r" else "python3"


def notebook_language(document: NotebookDocument) -> str:
    return (document.metadata.get("sigma") or {}).get("language", "python")


class DeleteNotebook:
    def __init__(self, store: NotebookStore, executor: CodeExecutor, owner: str) -> None:
        self._store = store
        self._executor = executor
        self._owner = owner

    def execute(self, notebook_id: str) -> None:
        self._store.delete(notebook_id)
        self._executor.shutdown(kernel_key(self._owner, notebook_id))


class ExportNotebook:
    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def execute(self, notebook_id: str) -> tuple[str, bytes]:
        document = self._store.get(notebook_id)
        return document.name, self._store.export_bytes(notebook_id)


class ExecuteCode:
    """Runs one cell in the notebook's own kernel (state persists between cells)."""

    def __init__(self, store: NotebookStore, executor: CodeExecutor, default_timeout_seconds: int, owner: str) -> None:
        self._store = store
        self._executor = executor
        self._default_timeout = default_timeout_seconds
        self._owner = owner

    def execute(self, notebook_id: str, code: str, timeout_seconds: float | None = None) -> ExecutionResult:
        document = self._store.get(notebook_id)  # fails with NotebookNotFoundError for unknown ids
        if len(code) > MAX_CODE_LENGTH:
            raise InvalidInputError("El código supera el tamaño máximo permitido.")
        timeout = float(timeout_seconds or self._default_timeout)
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise InvalidInputError(f"El tiempo máximo debe estar entre 1 y {MAX_TIMEOUT_SECONDS} segundos.")
        kernel = kernel_name_for(notebook_language(document))
        return self._executor.execute(kernel_key(self._owner, notebook_id), code, timeout, self._owner, kernel)


class ControlKernel:
    """Interrupt, restart or inspect a notebook's kernel."""

    def __init__(self, store: NotebookStore, executor: CodeExecutor, owner: str) -> None:
        self._store = store
        self._executor = executor
        self._owner = owner

    def _key(self, notebook_id: str) -> str:
        self._store.get(notebook_id)  # unknown (or somebody else's) notebooks fail here
        return kernel_key(self._owner, notebook_id)

    def start(self, notebook_id: str) -> KernelStatus:
        document = self._store.get(notebook_id)
        key = kernel_key(self._owner, notebook_id)
        self._executor.start(key, self._owner, kernel_name_for(notebook_language(document)))
        return self._executor.status(key)

    def interrupt(self, notebook_id: str) -> None:
        self._executor.interrupt(self._key(notebook_id))

    def restart(self, notebook_id: str) -> None:
        document = self._store.get(notebook_id)
        key = kernel_key(self._owner, notebook_id)
        self._executor.restart(key, self._owner, kernel_name_for(notebook_language(document)))

    def status(self, notebook_id: str) -> KernelStatus:
        return self._executor.status(self._key(notebook_id))
