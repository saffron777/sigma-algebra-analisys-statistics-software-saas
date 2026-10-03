"""Notebooks stored as standard ``.ipynb`` files (nbformat v4), so they open unchanged in Jupyter."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat

from sigma.domain.errors import ConflictError, NotebookNotFoundError
from sigma.domain.notebooks import NOTEBOOK_ID_PATTERN, NotebookDocument, NotebookSummary
from sigma.infrastructure.persistence import notebook_format as nbf

_WRITE_LOCK = threading.Lock()  # a revision check and the write that follows it must not interleave


def _revision(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


class FileNotebookStore:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ reads
    def list(self) -> list[NotebookSummary]:
        summaries = []
        for path in self._dir.glob("*.ipynb"):
            if not NOTEBOOK_ID_PATTERN.match(path.stem):
                continue
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            meta = notebook.get("metadata", {}).get("sigma", {})
            updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
            summaries.append(
                NotebookSummary(
                    path.stem,
                    meta.get("name") or path.stem,
                    len(notebook.get("cells", [])),
                    meta.get("created_at") or updated,
                    updated,
                    _revision(path),
                )
            )
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)

    def get(self, notebook_id: str) -> NotebookDocument:
        path = self._path(notebook_id)
        try:
            notebook = nbformat.read(path, as_version=4)
        except (OSError, ValueError) as error:
            raise NotebookNotFoundError("El notebook no existe.") from error
        return self._to_document(notebook_id, notebook, path)

    def export_bytes(self, notebook_id: str) -> bytes:
        path = self._path(notebook_id)
        if not path.is_file():
            raise NotebookNotFoundError("El notebook no existe.")
        return path.read_bytes()

    def signature(self) -> str:
        """Changes when any notebook is added, removed or rewritten (cheap: only file stats are read)."""
        parts = []
        for path in sorted(self._dir.glob("*.ipynb")):
            if NOTEBOOK_ID_PATTERN.match(path.stem):
                try:
                    parts.append(f"{path.stem}:{_revision(path)}")
                except OSError:
                    continue  # deleted while scanning
        return hashlib.sha1("|".join(parts).encode(), usedforsecurity=False).hexdigest()[:16]

    # ------------------------------------------------------------------ writes
    def create(self, name: str, cells: list[dict[str, Any]] | None = None) -> NotebookDocument:
        notebook = nbf.new_notebook(name, cells)
        notebook_id = uuid.uuid4().hex
        self._write(notebook_id, notebook)
        return self.get(notebook_id)

    def import_bytes(self, file_name: str, content: bytes) -> NotebookDocument:
        notebook = nbf.parse_import(file_name, content)
        notebook_id = uuid.uuid4().hex
        self._write(notebook_id, notebook)
        return self.get(notebook_id)

    def save(
        self,
        notebook_id: str,
        cells: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
        name: str | None,
        base_revision: str | None = None,
    ) -> NotebookDocument:
        with _WRITE_LOCK:
            return self._save(notebook_id, cells, metadata, name, base_revision)

    def _save(
        self,
        notebook_id: str,
        cells: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
        name: str | None,
        base_revision: str | None,
    ) -> NotebookDocument:
        current = self.get(notebook_id)
        if base_revision and base_revision != current.revision:
            raise ConflictError(
                "El notebook se modificó desde otra ventana después de que lo abriera.",
                {"revision": current.revision, "updated_at": current.updated_at},
            )
        notebook = nbf.rebuild(current.metadata, current.name, current.created_at, cells, metadata, name)
        self._write(notebook_id, notebook)
        return self.get(notebook_id)

    def delete(self, notebook_id: str) -> None:
        path = self._path(notebook_id)
        if not path.is_file():
            raise NotebookNotFoundError("El notebook no existe.")
        path.unlink()

    # ------------------------------------------------------------------ internals
    def _path(self, notebook_id: str) -> Path:
        if not NOTEBOOK_ID_PATTERN.match(notebook_id or ""):
            raise NotebookNotFoundError("Identificador de notebook inválido.")
        return self._dir / f"{notebook_id}.ipynb"

    def _write(self, notebook_id: str, notebook: Any) -> None:
        target = self._path(notebook_id)
        temp = target.with_suffix(".ipynb.tmp")
        temp.write_text(nbformat.writes(notebook), encoding="utf-8")
        os.replace(temp, target)  # atomic: a crash never leaves a half-written notebook

    @staticmethod
    def _to_document(notebook_id: str, notebook: Any, path: Path) -> NotebookDocument:
        meta = dict(notebook.metadata)
        sigma_meta = meta.get("sigma", {})
        updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        return NotebookDocument(
            id=notebook_id,
            name=sigma_meta.get("name") or notebook_id,
            created_at=sigma_meta.get("created_at") or updated,
            updated_at=updated,
            cells=nbf.plain_cells(notebook),
            metadata=nbf.plain_metadata(notebook),
            revision=_revision(path),
        )
