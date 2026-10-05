"""Notebooks kept as rows of the ``notebooks`` table: the store for every engine except SQLite (which uses files)."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import nbformat
from sqlalchemy import delete, insert, select, update

from sigma.domain.errors import ConflictError, NotebookNotFoundError
from sigma.domain.notebooks import NOTEBOOK_ID_PATTERN, NotebookDocument, NotebookSummary
from sigma.infrastructure.persistence import notebook_format as nbf
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database

_T = schema.notebooks


class SqlNotebookStore:
    """Same contract as ``FileNotebookStore``; the revision is an integer that every save increments, so the
    compare-and-swap is one atomic ``UPDATE … WHERE revision = :base`` (safe across processes and servers)."""

    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user_id = user_id

    # ------------------------------------------------------------------ reads
    def list(self) -> list[NotebookSummary]:
        statement = (
            select(_T.c.id, _T.c.name, _T.c.n_cells, _T.c.created_at, _T.c.updated_at, _T.c.revision)
            .where(_T.c.user_id == self._user_id)
            .order_by(_T.c.updated_at.desc(), _T.c.id)
        )
        with self._db.begin() as conn:
            rows = conn.execute(statement).all()
        return [
            NotebookSummary(r.id, r.name, int(r.n_cells), r.created_at, r.updated_at, str(r.revision)) for r in rows
        ]

    def get(self, notebook_id: str) -> NotebookDocument:
        row = self._row(notebook_id)
        notebook = nbformat.reads(row.content, as_version=4)
        return NotebookDocument(
            id=row.id,
            name=row.name,
            created_at=row.created_at,
            updated_at=row.updated_at,
            cells=nbf.plain_cells(notebook),
            metadata=nbf.plain_metadata(notebook),
            revision=str(row.revision),
        )

    def export_bytes(self, notebook_id: str) -> bytes:
        return self._row(notebook_id).content.encode("utf-8")

    def signature(self) -> str:
        statement = select(_T.c.id, _T.c.revision).where(_T.c.user_id == self._user_id).order_by(_T.c.id)
        with self._db.begin() as conn:
            parts = [f"{r.id}:{r.revision}" for r in conn.execute(statement)]
        return hashlib.sha1("|".join(parts).encode(), usedforsecurity=False).hexdigest()[:16]

    # ------------------------------------------------------------------ writes
    def create(self, name: str, cells: list[dict[str, Any]] | None = None, language: str = "python") -> NotebookDocument:
        return self._insert(nbf.new_notebook(name, cells, language))

    def import_bytes(self, file_name: str, content: bytes) -> NotebookDocument:
        return self._insert(nbf.parse_import(file_name, content))

    def save(
        self,
        notebook_id: str,
        cells: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
        name: str | None,
        base_revision: str | None = None,
    ) -> NotebookDocument:
        current = self.get(notebook_id)
        if base_revision and base_revision != current.revision:
            raise self._conflict(current)
        notebook = nbf.rebuild(current.metadata, current.name, current.created_at, cells, metadata, name)
        statement = (
            update(_T)
            .where(_T.c.id == notebook_id, _T.c.user_id == self._user_id, _T.c.revision == int(current.revision))
            .values(
                name=(notebook.metadata["sigma"]["name"])[:120],
                n_cells=len(notebook.cells),
                updated_at=nbf.now_iso(),
                revision=_T.c.revision + 1,
                content=nbformat.writes(notebook),
            )
        )
        with self._db.begin() as conn:
            changed = conn.execute(statement).rowcount
        if not changed:  # somebody saved between our read and our write
            raise self._conflict(self.get(notebook_id))
        return self.get(notebook_id)

    def delete(self, notebook_id: str) -> None:
        self._check_id(notebook_id)
        with self._db.begin() as conn:
            removed = conn.execute(delete(_T).where(_T.c.id == notebook_id, _T.c.user_id == self._user_id)).rowcount
        if not removed:
            raise NotebookNotFoundError("El notebook no existe.")

    # ------------------------------------------------------------------ internals
    def _insert(self, notebook: Any) -> NotebookDocument:
        notebook_id = uuid.uuid4().hex
        sigma = notebook.metadata["sigma"]
        now = nbf.now_iso()
        with self._db.begin() as conn:
            conn.execute(
                insert(_T).values(
                    id=notebook_id,
                    user_id=self._user_id,
                    name=str(sigma["name"])[:120],
                    n_cells=len(notebook.cells),
                    created_at=sigma["created_at"],
                    updated_at=now,
                    revision=1,
                    content=nbformat.writes(notebook),
                )
            )
        return self.get(notebook_id)

    @staticmethod
    def _check_id(notebook_id: str) -> None:
        if not NOTEBOOK_ID_PATTERN.match(notebook_id or ""):
            raise NotebookNotFoundError("Identificador de notebook inválido.")

    def _row(self, notebook_id: str) -> Any:
        self._check_id(notebook_id)
        with self._db.begin() as conn:
            row = conn.execute(select(_T).where(_T.c.id == notebook_id, _T.c.user_id == self._user_id)).first()
        if row is None:
            raise NotebookNotFoundError("El notebook no existe.")
        return row

    @staticmethod
    def _conflict(current: NotebookDocument) -> ConflictError:
        return ConflictError(
            "El notebook se modificó desde otra ventana después de que lo abriera.",
            {"revision": current.revision, "updated_at": current.updated_at},
        )
