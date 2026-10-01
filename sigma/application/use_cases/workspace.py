"""Per-module workspace (what the user was doing) and the cheap change probe used for live synchronisation."""

from __future__ import annotations

import json
from typing import Any

from sigma.application.ports import NotebookStore, SyncRepository, WorkspaceRepository
from sigma.domain.errors import InvalidInputError
from sigma.domain.workspace import MAX_WORKSPACE_BYTES, UNSET, WorkspaceEntry, validate_module


class GetWorkspace:
    def __init__(self, workspace: WorkspaceRepository) -> None:
        self._workspace = workspace

    def execute(self, module: str) -> WorkspaceEntry:
        return self._workspace.get(validate_module(module))


class SaveWorkspace:
    """Partial update: ``state`` (the form) and ``payload`` (the last result) are saved independently."""

    def __init__(self, workspace: WorkspaceRepository) -> None:
        self._workspace = workspace

    def execute(self, module: str, state: Any = UNSET, payload: Any = UNSET) -> WorkspaceEntry:
        validate_module(module)
        for name, value in (("state", state), ("payload", payload)):
            if value is not UNSET and value is not None and not isinstance(value, dict):
                raise InvalidInputError(f"«{name}» debe ser un objeto JSON.")
        size = sum(len(json.dumps(v, ensure_ascii=False)) for v in (state, payload) if v is not UNSET and v is not None)
        if size > MAX_WORKSPACE_BYTES:
            raise InvalidInputError(
                "El espacio de trabajo de este módulo supera el tamaño máximo (16 MB). Use «Limpiar»."
            )
        return self._workspace.save(module, state, payload)


class ClearWorkspace:
    def __init__(self, workspace: WorkspaceRepository) -> None:
        self._workspace = workspace

    def execute(self, module: str) -> int:
        return self._workspace.clear(validate_module(module))

    def execute_all(self) -> dict[str, int]:
        return self._workspace.clear_all()


class GetSyncState:
    """Revisions of everything one user can change from several windows; polled every few seconds."""

    def __init__(self, workspace: WorkspaceRepository, sync: SyncRepository, notebooks: NotebookStore) -> None:
        self._workspace = workspace
        self._sync = sync
        self._notebooks = notebooks

    def execute(self) -> dict[str, Any]:
        scopes = self._sync.revisions()
        return {
            "datasets": scopes.get("datasets", 0),
            "results": scopes.get("results", 0),
            "drafts": scopes.get("drafts", 0),
            "templates": scopes.get("templates", 0),
            "notebooks": self._notebooks.signature(),
            "workspace": self._workspace.revisions(),
        }
