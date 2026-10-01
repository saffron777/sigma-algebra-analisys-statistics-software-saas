"""Notebooks stored as standard ``.ipynb`` files (nbformat v4), so they open unchanged in Jupyter."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import nbformat
from nbformat import ValidationError

from sigma.domain.errors import ImportFailedError, InvalidInputError, NotebookNotFoundError
from sigma.domain.notebooks import NOTEBOOK_ID_PATTERN, NotebookDocument, NotebookSummary

NBFORMAT_MINOR = 5  # minor version that introduced cell ids
MAX_CELLS = 2000
_CELL_KEYS = {
    "code": ("id", "cell_type", "source", "metadata", "outputs", "execution_count"),
    "markdown": ("id", "cell_type", "source", "metadata"),
    "raw": ("id", "cell_type", "source", "metadata"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_cell_id() -> str:
    return uuid.uuid4().hex[:8]


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

    # ------------------------------------------------------------------ writes
    def create(self, name: str, cells: list[dict[str, Any]] | None = None) -> NotebookDocument:
        notebook = nbformat.v4.new_notebook()
        notebook.nbformat_minor = NBFORMAT_MINOR
        notebook.cells = [nbformat.from_dict(c) for c in (cells or [self._blank_code_cell()])]
        notebook.metadata["sigma"] = {"name": name, "created_at": _now()}
        notebook_id = uuid.uuid4().hex
        self._write(notebook_id, notebook)
        return self.get(notebook_id)

    def import_bytes(self, file_name: str, content: bytes) -> NotebookDocument:
        try:
            notebook = nbformat.reads(content.decode("utf-8-sig"), as_version=4)
            nbformat.validate(notebook)
        except UnicodeDecodeError as error:
            raise ImportFailedError("El archivo no es un notebook válido (no está codificado en UTF-8).") from error
        except (ValidationError, ValueError, KeyError, AttributeError) as error:
            raise ImportFailedError(
                f"El archivo no es un notebook Jupyter (.ipynb) válido: {str(error)[:200]}"
            ) from error
        if len(notebook.cells) > MAX_CELLS:
            raise ImportFailedError(f"El notebook supera el máximo de {MAX_CELLS} celdas.")
        self._normalize(notebook)
        name = (
            notebook.metadata.get("sigma", {}).get("name")
            or notebook.metadata.get("title")
            or PurePath(file_name or "").stem
            or "Notebook importado"
        )
        notebook.metadata["sigma"] = {"name": str(name)[:120], "created_at": _now()}
        notebook_id = uuid.uuid4().hex
        self._write(notebook_id, notebook)
        return self.get(notebook_id)

    def save(
        self, notebook_id: str, cells: list[dict[str, Any]], metadata: dict[str, Any] | None, name: str | None
    ) -> NotebookDocument:
        current = self.get(notebook_id)
        if len(cells) > MAX_CELLS:
            raise InvalidInputError(f"Un notebook admite como máximo {MAX_CELLS} celdas.")
        notebook = nbformat.v4.new_notebook()
        notebook.nbformat_minor = NBFORMAT_MINOR
        notebook.cells = [nbformat.from_dict(self._clean_cell(c)) for c in cells]
        kept = {k: v for k, v in (metadata or current.metadata).items() if k != "sigma"}
        notebook.metadata.update(kept)
        notebook.metadata["sigma"] = {"name": name or current.name, "created_at": current.created_at}
        self._normalize(notebook)
        try:
            nbformat.validate(notebook)
        except ValidationError as error:
            raise InvalidInputError(f"El contenido del notebook no es válido: {str(error)[:200]}") from error
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
    def _blank_code_cell() -> dict[str, Any]:
        return {
            "cell_type": "code",
            "id": _new_cell_id(),
            "metadata": {},
            "source": "",
            "outputs": [],
            "execution_count": None,
        }

    @staticmethod
    def _clean_cell(cell: dict[str, Any]) -> dict[str, Any]:
        """Keep only the nbformat keys of a cell sent by the browser (it also tracks UI-only state)."""
        kind = cell.get("cell_type")
        if kind not in _CELL_KEYS:
            raise InvalidInputError(f"Tipo de celda inválido: {kind!r}.")
        clean = {key: cell[key] for key in _CELL_KEYS[kind] if key in cell}
        source = clean.get("source", "")
        clean["source"] = "".join(source) if isinstance(source, list) else str(source)
        clean.setdefault("metadata", {})
        if kind == "code":
            clean.setdefault("outputs", [])
            clean.setdefault("execution_count", None)
        return clean

    @staticmethod
    def _normalize(notebook: Any) -> None:
        """Make sure every cell has a unique id (required by nbformat ≥ 4.5)."""
        notebook.nbformat_minor = max(notebook.get("nbformat_minor", 0), NBFORMAT_MINOR)
        seen: set[str] = set()
        for cell in notebook.cells:
            cell_id = cell.get("id")
            if not cell_id or cell_id in seen:
                cell_id = _new_cell_id()
            cell["id"] = cell_id
            seen.add(cell_id)

    @staticmethod
    def _to_document(notebook_id: str, notebook: Any, path: Path) -> NotebookDocument:
        meta = dict(notebook.metadata)
        sigma_meta = meta.get("sigma", {})
        updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        for cell in notebook.cells:
            if "id" not in cell:
                cell["id"] = _new_cell_id()
        return NotebookDocument(
            id=notebook_id,
            name=sigma_meta.get("name") or notebook_id,
            created_at=sigma_meta.get("created_at") or updated,
            updated_at=updated,
            cells=json.loads(json.dumps(list(notebook.cells))),
            metadata=json.loads(json.dumps(meta)),
        )
