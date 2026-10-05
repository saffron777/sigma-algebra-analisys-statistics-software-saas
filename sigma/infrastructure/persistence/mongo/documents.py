"""Import drafts, report templates and notebooks on MongoDB."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import nbformat
from bson.binary import Binary
from pymongo.errors import DuplicateKeyError

from sigma.application.ports import DraftInfo, StagedImport
from sigma.domain.datasets import ColumnInfo, describe_columns
from sigma.domain.errors import ConflictError, InvalidInputError, NotebookNotFoundError, TemplateNotFoundError
from sigma.domain.notebooks import NOTEBOOK_ID_PATTERN, NotebookDocument, NotebookSummary
from sigma.domain.report_templates import ReportTemplate, TemplateFields
from sigma.infrastructure.persistence import notebook_format as nbf
from sigma.infrastructure.persistence.mongo import frames
from sigma.infrastructure.persistence.mongo.core import MongoCore
from sigma.infrastructure.persistence.sql.helpers import now_iso

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
EXPIRED = "La importación pendiente ya no existe (se descartó, expiró o ya fue guardada). Cargue la fuente otra vez."
CHUNK_BYTES = 1024 * 1024


# ------------------------------------------------------------------ import drafts
class MongoImportDraftStore:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._core = core
        self._user = int(user_id)
        self._drafts = core.col("import_drafts")
        self._files = core.col("import_draft_files")

    def attach_file(self, source_type: str, file_name: str, content: bytes) -> DraftInfo:
        now = now_iso()
        previous = self._for_source(source_type)
        if previous is not None:  # a new file makes the old parsed preview stale
            frames.drop_frame(self._core, previous.get("staged_owner"))
        fresh = {
            "file_name": file_name,
            "file_size": len(content),
            "options": "{}",
            "staged_owner": None,
            "n_rows": None,
            "n_cols": None,
            "dtypes": None,
            "columns_info": None,
            "source_meta": "{}",
            "detected": "{}",
            "suggested_name": None,
            "updated_at": now,
        }
        self._drafts.update_one(
            {"user_id": self._user, "source_type": source_type},
            {"$set": fresh, "$setOnInsert": {"id": uuid.uuid4().hex, "created_at": now}},
            upsert=True,
        )
        draft = self._for_source(source_type)
        self._files.delete_many({"draft_id": draft["id"]})
        for seq, start in enumerate(range(0, max(len(content), 1), CHUNK_BYTES)):
            self._files.insert_one(
                {"draft_id": draft["id"], "seq": seq, "data": Binary(content[start : start + CHUNK_BYTES])}
            )
        self._core.bump_sync(self._user, "drafts")
        return self._to_info(draft)

    def file(self, token: str) -> tuple[DraftInfo, bytes | None]:
        info = self.info(token)
        chunks = list(self._files.find({"draft_id": token}).sort("seq", 1))
        return info, (b"".join(bytes(c["data"]) for c in chunks) if chunks else None)

    def put(self, staged: StagedImport, token: str | None = None, options: dict[str, Any] | None = None) -> str:
        frame = staged.frame
        dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
        columns = [vars(c) for c in describe_columns(frame)]
        now = now_iso()
        doc = self._by_id(token) if token else self._for_source(staged.source_type)
        if token and doc is None:
            raise InvalidInputError(EXPIRED)
        owner = frames.new_owner("stg")
        try:
            frames.write_frame(self._core, owner, frame)
        except BaseException:
            frames.drop_frame(self._core, owner)
            raise
        values = {
            "staged_owner": owner,
            "n_rows": len(frame),
            "n_cols": frame.shape[1],
            "dtypes": json.dumps(dtypes),
            "columns_info": json.dumps(columns),
            "source_meta": json.dumps(staged.source_meta, ensure_ascii=False, default=str),
            "detected": json.dumps(staged.detected, ensure_ascii=False, default=str),
            "suggested_name": staged.suggested_name,
            "options": json.dumps(options or {}, ensure_ascii=False, default=str),
            "updated_at": now,
        }
        if doc is None:
            self._drafts.update_one(
                {"user_id": self._user, "source_type": staged.source_type},
                {
                    "$set": values,
                    "$setOnInsert": {
                        "id": uuid.uuid4().hex,
                        "file_name": None,
                        "file_size": 0,
                        "created_at": now,
                    },
                },
                upsert=True,
            )
            doc = self._for_source(staged.source_type)
        else:
            frames.drop_frame(self._core, doc.get("staged_owner"))
            self._drafts.update_one({"id": doc["id"], "user_id": self._user}, {"$set": values})
        self._core.bump_sync(self._user, "drafts")
        return doc["id"]

    def take(self, token: str) -> StagedImport:
        doc = self._by_id(token)
        if doc is None or not doc.get("staged_owner"):
            raise InvalidInputError(EXPIRED)
        return StagedImport(
            frame=self._read(doc, None),
            source_type=doc["source_type"],
            source_meta=json.loads(doc["source_meta"]),
            suggested_name=doc.get("suggested_name") or "importado",
            detected=json.loads(doc["detected"]),
        )

    def preview(self, token: str, limit: int) -> tuple[DraftInfo, tuple[ColumnInfo, ...], Any, dict[str, Any]]:
        doc = self._by_id(token)
        if doc is None or not doc.get("staged_owner"):
            raise InvalidInputError(EXPIRED)
        columns = tuple(ColumnInfo(**item) for item in json.loads(doc["columns_info"]))
        return self._to_info(doc), columns, self._read(doc, limit), json.loads(doc["detected"])

    def info(self, token: str) -> DraftInfo:
        doc = self._by_id(token)
        if doc is None:
            raise InvalidInputError(EXPIRED)
        return self._to_info(doc)

    def list(self) -> list[DraftInfo]:
        return [self._to_info(d) for d in self._drafts.find({"user_id": self._user}).sort("updated_at", -1)]

    def discard(self, token: str) -> None:
        doc = self._by_id(token)
        if doc is None:
            raise InvalidInputError(EXPIRED)
        frames.drop_frame(self._core, doc.get("staged_owner"))
        self._files.delete_many({"draft_id": token})
        self._drafts.delete_one({"id": token, "user_id": self._user})
        self._core.bump_sync(self._user, "drafts")

    def discard_preview(self, token: str) -> None:
        doc = self._by_id(token)
        if doc is None:
            raise InvalidInputError(EXPIRED)
        frames.drop_frame(self._core, doc.get("staged_owner"))
        self._drafts.update_one(
            {"id": token},
            {
                "$set": {
                    "staged_owner": None,
                    "n_rows": None,
                    "n_cols": None,
                    "dtypes": None,
                    "columns_info": None,
                    "detected": "{}",
                    "suggested_name": None,
                    "updated_at": now_iso(),
                }
            },
        )
        self._core.bump_sync(self._user, "drafts")

    def purge(self) -> int:
        drafts = self.list()
        for draft in drafts:
            self.discard(draft.id)
        return len(drafts)

    # ------------------------------------------------------------------ internals
    def _read(self, doc: dict[str, Any], limit: int | None) -> Any:
        dtypes = json.loads(doc["dtypes"])
        return frames.read_frame(self._core, doc["staged_owner"], list(dtypes), dtypes, limit, 0)

    def _by_id(self, token: str | None) -> dict[str, Any] | None:
        if not _ID_PATTERN.match(token or ""):
            return None
        return self._drafts.find_one({"id": token, "user_id": self._user})

    def _for_source(self, source_type: str) -> dict[str, Any] | None:
        return self._drafts.find_one({"user_id": self._user, "source_type": source_type})

    @staticmethod
    def _to_info(doc: dict[str, Any]) -> DraftInfo:
        return DraftInfo(
            id=doc["id"],
            source_type=doc["source_type"],
            file_name=doc.get("file_name"),
            file_size=doc.get("file_size") or 0,
            options=json.loads(doc["options"]),
            n_rows=doc.get("n_rows"),
            n_cols=doc.get("n_cols"),
            suggested_name=doc.get("suggested_name"),
            has_preview=bool(doc.get("staged_owner")),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )


def purge_stale_drafts(core: MongoCore, max_age_days: int = 14) -> int:
    cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
    removed = 0
    for doc in list(core.col("import_drafts").find({"updated_at": {"$lt": cutoff}})):
        frames.drop_frame(core, doc.get("staged_owner"))
        core.col("import_draft_files").delete_many({"draft_id": doc["id"]})
        core.col("import_drafts").delete_one({"id": doc["id"]})
        removed += 1
    return removed


# ------------------------------------------------------------------ report templates
class MongoTemplateRepository:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._core = core
        self._user = int(user_id)
        self._templates = core.col("report_templates")

    def add(self, fields: TemplateFields, make_default: bool = False) -> ReportTemplate:
        template_id = uuid.uuid4().hex
        now = now_iso()
        first = self._templates.count_documents({"user_id": self._user}) == 0
        try:
            self._templates.insert_one(
                {
                    "id": template_id,
                    "user_id": self._user,
                    **self._values(fields),
                    "logo": None,
                    "is_default": 0,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
        if make_default or first:  # the first template is the default until another one is chosen
            self._mark_default(template_id)
        self._core.bump_sync(self._user, "templates")
        return self.get(template_id)

    def update(self, template_id: str, fields: TemplateFields) -> ReportTemplate:
        self.get(template_id)
        try:
            self._templates.update_one(
                {"id": template_id, "user_id": self._user},
                {"$set": {**self._values(fields), "updated_at": now_iso()}},
            )
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
        self._core.bump_sync(self._user, "templates")
        return self.get(template_id)

    def delete(self, template_id: str) -> None:
        template = self.get(template_id)
        self._templates.delete_one({"id": template_id, "user_id": self._user})
        if template.is_default:  # somebody must stay the default
            heirs = self._templates.find({"user_id": self._user}).sort([("created_at", 1), ("id", 1)]).limit(1)
            heir = next(iter(heirs), None)
            if heir:
                self._mark_default(heir["id"])
        self._core.bump_sync(self._user, "templates")

    def set_default(self, template_id: str, default: bool) -> ReportTemplate:
        self.get(template_id)
        if default:
            self._mark_default(template_id)
        else:
            self._templates.update_one(
                {"id": template_id, "user_id": self._user}, {"$set": {"is_default": 0, "updated_at": now_iso()}}
            )
        self._core.bump_sync(self._user, "templates")
        return self.get(template_id)

    def set_logo(self, template_id: str, logo: bytes | None) -> ReportTemplate:
        self.get(template_id)
        self._templates.update_one(
            {"id": template_id, "user_id": self._user},
            {"$set": {"logo": None if logo is None else Binary(logo), "updated_at": now_iso()}},
        )
        self._core.bump_sync(self._user, "templates")
        return self.get(template_id)

    def get(self, template_id: str) -> ReportTemplate:
        if not _ID_PATTERN.match(template_id or ""):
            raise TemplateNotFoundError("La plantilla no existe.")
        doc = self._templates.find_one({"id": template_id, "user_id": self._user})
        if doc is None:
            raise TemplateNotFoundError("La plantilla no existe.")
        return self._to_template(doc)

    def get_default(self) -> ReportTemplate | None:
        doc = self._templates.find_one({"user_id": self._user, "is_default": 1})
        return self._to_template(doc) if doc else None

    def list(self) -> list[ReportTemplate]:
        docs = self._templates.find({"user_id": self._user}).sort([("is_default", -1), ("name", 1)])
        return [self._to_template(doc) for doc in docs]

    def get_logo(self, template_id: str) -> bytes | None:
        self.get(template_id)
        doc = self._templates.find_one({"id": template_id, "user_id": self._user}, {"logo": 1})
        return bytes(doc["logo"]) if doc and doc.get("logo") is not None else None

    @staticmethod
    def _values(fields: TemplateFields) -> dict[str, object]:
        return {
            "name": fields.name,
            "name_key": fields.name.lower(),
            "organization": fields.organization,
            "primary_color": fields.primary_color,
            "secondary_color": fields.secondary_color,
            "footer_text": fields.footer_text,
            "cover_text": fields.cover_text,
            "page_size": fields.page_size,
            "orientation": fields.orientation,
            "show_cover": int(fields.show_cover),
            "show_toc": int(fields.show_toc),
        }

    def _mark_default(self, template_id: str) -> None:
        self._templates.update_many({"user_id": self._user, "is_default": 1}, {"$set": {"is_default": 0}})
        self._templates.update_one(
            {"id": template_id, "user_id": self._user}, {"$set": {"is_default": 1, "updated_at": now_iso()}}
        )

    @staticmethod
    def _to_template(doc: dict[str, Any]) -> ReportTemplate:
        return ReportTemplate(
            id=doc["id"],
            name=doc["name"],
            organization=doc.get("organization") or "",
            primary_color=doc["primary_color"],
            secondary_color=doc["secondary_color"],
            footer_text=doc.get("footer_text") or "",
            cover_text=doc.get("cover_text") or "",
            has_logo=doc.get("logo") is not None,
            page_size=doc["page_size"],
            orientation=doc["orientation"],
            show_cover=bool(doc["show_cover"]),
            show_toc=bool(doc["show_toc"]),
            is_default=bool(doc["is_default"]),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )


# ------------------------------------------------------------------ notebooks
class MongoNotebookStore:
    """Same contract as the other stores; the revision is an integer and a save is one compare-and-swap update."""

    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._notebooks = core.col("notebooks")
        self._user = int(user_id)

    def list(self) -> list[NotebookSummary]:
        docs = self._notebooks.find({"user_id": self._user}).sort([("updated_at", -1), ("id", 1)])
        return [
            NotebookSummary(d["id"], d["name"], int(d["n_cells"]), d["created_at"], d["updated_at"], str(d["revision"]))
            for d in docs
        ]

    def get(self, notebook_id: str) -> NotebookDocument:
        doc = self._doc(notebook_id)
        notebook = nbformat.reads(doc["content"], as_version=4)
        return NotebookDocument(
            id=doc["id"],
            name=doc["name"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            cells=nbf.plain_cells(notebook),
            metadata=nbf.plain_metadata(notebook),
            revision=str(doc["revision"]),
        )

    def export_bytes(self, notebook_id: str) -> bytes:
        return self._doc(notebook_id)["content"].encode("utf-8")

    def signature(self) -> str:
        parts = [f"{d['id']}:{d['revision']}" for d in self._notebooks.find({"user_id": self._user}).sort("id", 1)]
        return hashlib.sha1("|".join(parts).encode(), usedforsecurity=False).hexdigest()[:16]

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
        changed = self._notebooks.update_one(
            {"id": notebook_id, "user_id": self._user, "revision": int(current.revision)},
            {
                "$set": {
                    "name": str(notebook.metadata["sigma"]["name"])[:120],
                    "n_cells": len(notebook.cells),
                    "updated_at": nbf.now_iso(),
                    "content": nbformat.writes(notebook),
                },
                "$inc": {"revision": 1},
            },
        ).matched_count
        if not changed:  # somebody saved between our read and our write
            raise self._conflict(self.get(notebook_id))
        return self.get(notebook_id)

    def delete(self, notebook_id: str) -> None:
        self._check_id(notebook_id)
        if not self._notebooks.delete_one({"id": notebook_id, "user_id": self._user}).deleted_count:
            raise NotebookNotFoundError("El notebook no existe.")

    def _insert(self, notebook: Any) -> NotebookDocument:
        notebook_id = uuid.uuid4().hex
        sigma = notebook.metadata["sigma"]
        self._notebooks.insert_one(
            {
                "id": notebook_id,
                "user_id": self._user,
                "name": str(sigma["name"])[:120],
                "n_cells": len(notebook.cells),
                "created_at": sigma["created_at"],
                "updated_at": nbf.now_iso(),
                "revision": 1,
                "content": nbformat.writes(notebook),
            }
        )
        return self.get(notebook_id)

    @staticmethod
    def _check_id(notebook_id: str) -> None:
        if not NOTEBOOK_ID_PATTERN.match(notebook_id or ""):
            raise NotebookNotFoundError("Identificador de notebook inválido.")

    def _doc(self, notebook_id: str) -> dict[str, Any]:
        self._check_id(notebook_id)
        doc = self._notebooks.find_one({"id": notebook_id, "user_id": self._user})
        if doc is None:
            raise NotebookNotFoundError("El notebook no existe.")
        return doc

    @staticmethod
    def _conflict(current: NotebookDocument) -> ConflictError:
        return ConflictError(
            "El notebook se modificó desde otra ventana después de que lo abriera.",
            {"revision": current.revision, "updated_at": current.updated_at},
        )
