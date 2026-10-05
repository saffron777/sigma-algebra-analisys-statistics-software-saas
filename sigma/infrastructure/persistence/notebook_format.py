"""nbformat rules shared by every notebook store (files for SQLite, rows for the other engines)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

import nbformat
from nbformat import ValidationError

from sigma.domain.errors import ImportFailedError, InvalidInputError

NBFORMAT_MINOR = 5  # minor version that introduced cell ids
MAX_CELLS = 2000
_CELL_KEYS = {
    "code": ("id", "cell_type", "source", "metadata", "outputs", "execution_count"),
    "markdown": ("id", "cell_type", "source", "metadata"),
    "raw": ("id", "cell_type", "source", "metadata"),
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_cell_id() -> str:
    return uuid.uuid4().hex[:8]


def blank_code_cell() -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": new_cell_id(),
        "metadata": {},
        "source": "",
        "outputs": [],
        "execution_count": None,
    }


def clean_cell(cell: dict[str, Any]) -> dict[str, Any]:
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


def normalize(notebook: Any) -> None:
    """Make sure every cell has a unique id (required by nbformat ≥ 4.5)."""
    notebook.nbformat_minor = max(notebook.get("nbformat_minor", 0), NBFORMAT_MINOR)
    seen: set[str] = set()
    for cell in notebook.cells:
        cell_id = cell.get("id")
        if not cell_id or cell_id in seen:
            cell_id = new_cell_id()
        cell["id"] = cell_id
        seen.add(cell_id)


def new_notebook(name: str, cells: list[dict[str, Any]] | None = None, language: str = "python") -> Any:
    notebook = nbformat.v4.new_notebook()
    notebook.nbformat_minor = NBFORMAT_MINOR
    notebook.cells = [nbformat.from_dict(c) for c in (cells or [blank_code_cell()])]
    notebook.metadata["sigma"] = {"name": name, "created_at": now_iso(), "language": language}
    return notebook


def parse_import(file_name: str, content: bytes) -> Any:
    """A notebook read from an uploaded ``.ipynb`` (validated, ids normalised, named, stamped)."""
    try:
        notebook = nbformat.reads(content.decode("utf-8-sig"), as_version=4)
        nbformat.validate(notebook)
    except UnicodeDecodeError as error:
        raise ImportFailedError("El archivo no es un notebook válido (no está codificado en UTF-8).") from error
    except (ValidationError, ValueError, KeyError, AttributeError) as error:
        raise ImportFailedError(f"El archivo no es un notebook Jupyter (.ipynb) válido: {str(error)[:200]}") from error
    if len(notebook.cells) > MAX_CELLS:
        raise ImportFailedError(f"El notebook supera el máximo de {MAX_CELLS} celdas.")
    normalize(notebook)
    name = (
        notebook.metadata.get("sigma", {}).get("name")
        or notebook.metadata.get("title")
        or PurePath(file_name or "").stem
        or "Notebook importado"
    )
    kernelspec = notebook.metadata.get("kernelspec", {})
    language = "r" if kernelspec.get("name") == "ir" or kernelspec.get("language") == "R" else "python"
    notebook.metadata["sigma"] = {"name": str(name)[:120], "created_at": now_iso(), "language": language}
    return notebook


def rebuild(
    current_metadata: dict[str, Any],
    current_name: str,
    created_at: str,
    cells: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    name: str | None,
) -> Any:
    """The notebook a save produces: the sent cells and metadata on top of what the stored one keeps."""
    if len(cells) > MAX_CELLS:
        raise InvalidInputError(f"Un notebook admite como máximo {MAX_CELLS} celdas.")
    notebook = nbformat.v4.new_notebook()
    notebook.nbformat_minor = NBFORMAT_MINOR
    notebook.cells = [nbformat.from_dict(clean_cell(c)) for c in cells]
    kept = {k: v for k, v in (metadata or current_metadata).items() if k != "sigma"}
    notebook.metadata.update(kept)
    language = (current_metadata or {}).get("sigma", {}).get("language", "python")
    notebook.metadata["sigma"] = {"name": name or current_name, "created_at": created_at, "language": language}
    normalize(notebook)
    try:
        nbformat.validate(notebook)
    except ValidationError as error:
        raise InvalidInputError(f"El contenido del notebook no es válido: {str(error)[:200]}") from error
    return notebook


def plain_cells(notebook: Any) -> list[dict[str, Any]]:
    for cell in notebook.cells:
        if "id" not in cell:
            cell["id"] = new_cell_id()
    return json.loads(json.dumps(list(notebook.cells)))


def plain_metadata(notebook: Any) -> dict[str, Any]:
    return json.loads(json.dumps(dict(notebook.metadata)))


def iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")
