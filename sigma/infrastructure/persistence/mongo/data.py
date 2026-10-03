"""Datasets, saved results, the per-module workspace and the sync revisions on MongoDB."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pandas as pd
from pymongo.errors import DuplicateKeyError

from sigma.domain.datasets import ColumnInfo, Dataset, DatasetInfo, describe_columns
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError, ResultNotFoundError
from sigma.domain.saved_results import SavedResult, SavedResultSummary
from sigma.domain.workspace import UNSET, WorkspaceEntry
from sigma.infrastructure.persistence.mongo import frames
from sigma.infrastructure.persistence.mongo.core import MongoCore
from sigma.infrastructure.persistence.sql.helpers import now_iso, now_ms

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------------ datasets
class MongoDatasetRepository:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._core = core
        self._user = int(user_id)
        self._datasets = core.col("datasets")

    def add(self, name: str, frame: pd.DataFrame, source_type: str, source_meta: dict[str, Any]) -> DatasetInfo:
        name = self._valid_name(name)
        if self._datasets.find_one({"user_id": self._user, "name_key": name.lower()}):
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.")
        dataset_id, data_id = uuid.uuid4().hex, frames.new_owner("ds")
        try:
            frames.write_frame(self._core, data_id, frame)
            dtypes, columns = self._frame_metadata(frame)
            self._datasets.insert_one(
                {
                    "id": dataset_id,
                    "user_id": self._user,
                    "name": name,
                    "name_key": name.lower(),
                    "source_type": source_type,
                    "source_meta": json.dumps(source_meta, ensure_ascii=False, default=str),
                    "n_rows": len(frame),
                    "n_cols": frame.shape[1],
                    "dtypes": dtypes,
                    "columns_info": columns,
                    "created_at": now_iso(),
                    "data_id": data_id,
                }
            )
        except DuplicateKeyError:
            frames.drop_frame(self._core, data_id)
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.") from None
        except BaseException:
            frames.drop_frame(self._core, data_id)
            raise
        self._core.bump_sync(self._user, "datasets")
        return self.get_info(dataset_id)

    def replace_data(self, dataset_id: str, frame: pd.DataFrame) -> DatasetInfo:
        doc = self._doc(dataset_id)
        new_owner = frames.new_owner("ds")
        try:
            frames.write_frame(self._core, new_owner, frame)
            dtypes, columns = self._frame_metadata(frame)
            self._datasets.update_one(
                {"id": dataset_id, "user_id": self._user},
                {
                    "$set": {
                        "n_rows": len(frame),
                        "n_cols": frame.shape[1],
                        "dtypes": dtypes,
                        "columns_info": columns,
                        "data_id": new_owner,
                    }
                },
            )
        except BaseException:
            frames.drop_frame(self._core, new_owner)
            raise
        frames.drop_frame(self._core, doc.get("data_id"))
        self._core.bump_sync(self._user, "datasets")
        return self.get_info(dataset_id)

    def rename(self, dataset_id: str, name: str) -> DatasetInfo:
        name = self._valid_name(name)
        self._doc(dataset_id)
        clash = self._datasets.find_one({"user_id": self._user, "name_key": name.lower()})
        if clash and clash["id"] != dataset_id:
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.")
        try:
            self._datasets.update_one(
                {"id": dataset_id, "user_id": self._user}, {"$set": {"name": name, "name_key": name.lower()}}
            )
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe un dataset llamado «{name}». Elija otro nombre.") from None
        self._core.col("saved_results").update_many(
            {"dataset_id": dataset_id, "user_id": self._user}, {"$set": {"dataset_name": name}}
        )
        self._core.bump_sync(self._user, "datasets")
        self._core.bump_sync(self._user, "results")
        return self.get_info(dataset_id)

    def delete(self, dataset_id: str) -> None:
        doc = self._doc(dataset_id)
        self._release([dataset_id])
        self._datasets.delete_one({"id": dataset_id, "user_id": self._user})
        frames.drop_frame(self._core, doc.get("data_id"))
        self._core.bump_sync(self._user, "datasets")

    def purge(self) -> int:
        docs = list(self._datasets.find({"user_id": self._user}))
        self._release([doc["id"] for doc in docs])
        for doc in docs:
            frames.drop_frame(self._core, doc.get("data_id"))
        self._datasets.delete_many({"user_id": self._user})
        return len(docs)

    # ------------------------------------------------------------------ reads
    def get_info(self, dataset_id: str) -> DatasetInfo:
        return self._to_info(self._doc(dataset_id))

    def list(self) -> list[DatasetInfo]:
        docs = self._datasets.find({"user_id": self._user}).sort([("created_at", -1), ("name", 1)])
        return [self._to_info(doc) for doc in docs]

    def get(self, dataset_id: str) -> Dataset:
        doc = self._doc(dataset_id)
        return Dataset(self._to_info(doc), self._read(doc, None, None))

    def get_by_name(self, name: str) -> Dataset:
        doc = self._datasets.find_one({"user_id": self._user, "name_key": (name or "").strip().lower()})
        if doc is None:
            raise DatasetNotFoundError(f"No existe un dataset llamado «{name}».")
        return Dataset(self._to_info(doc), self._read(doc, None, None))

    def preview(self, dataset_id: str, limit: int, offset: int) -> pd.DataFrame:
        return self._read(self._doc(dataset_id), limit, offset)

    # ------------------------------------------------------------------ internals
    def _doc(self, dataset_id: str) -> dict[str, Any]:
        if not _ID_PATTERN.match(dataset_id or ""):
            raise DatasetNotFoundError("Identificador de dataset inválido.")
        doc = self._datasets.find_one({"id": dataset_id, "user_id": self._user})
        if doc is None:
            raise DatasetNotFoundError("El dataset no existe (puede haber sido eliminado).")
        return doc

    def _read(self, doc: dict[str, Any], limit: int | None, offset: int | None) -> pd.DataFrame:
        dtypes = json.loads(doc["dtypes"])
        return frames.read_frame(self._core, doc["data_id"], list(dtypes), dtypes, limit, offset)

    def _release(self, dataset_ids: list[str]) -> None:
        """Results keep their name but lose the link to a dataset that is going away."""
        if dataset_ids:
            self._core.col("saved_results").update_many(
                {"dataset_id": {"$in": dataset_ids}, "user_id": self._user}, {"$set": {"dataset_id": None}}
            )
            self._core.bump_sync(self._user, "results")

    @staticmethod
    def _valid_name(name: str) -> str:
        name = " ".join((name or "").split())
        if not name:
            raise InvalidInputError("El nombre del dataset no puede estar vacío.")
        return name

    @staticmethod
    def _frame_metadata(frame: pd.DataFrame) -> tuple[str, str]:
        dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
        columns = [vars(c) for c in describe_columns(frame)]
        return json.dumps(dtypes), json.dumps(columns)

    @staticmethod
    def _to_info(doc: dict[str, Any]) -> DatasetInfo:
        return DatasetInfo(
            id=doc["id"],
            name=doc["name"],
            source_type=doc["source_type"],
            n_rows=doc["n_rows"],
            n_cols=doc["n_cols"],
            columns=tuple(ColumnInfo(**item) for item in json.loads(doc["columns_info"])),
            created_at=doc["created_at"],
            source_meta=json.loads(doc["source_meta"]),
        )


# ------------------------------------------------------------------ saved results
def _summary(doc: dict[str, Any]) -> SavedResultSummary:
    return SavedResultSummary(
        id=doc["id"],
        module=doc["module"],
        analysis_type=doc["analysis_type"],
        title=doc["title"],
        notes=doc.get("notes") or "",
        dataset_id=doc.get("dataset_id"),
        dataset_name=doc.get("dataset_name"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


class MongoResultRepository:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._core = core
        self._user = int(user_id)
        self._results = core.col("saved_results")

    def add(self, result: SavedResult) -> SavedResult:
        self._results.insert_one(
            {
                "id": result.id,
                "user_id": self._user,
                "module": result.module,
                "analysis_type": result.analysis_type,
                "title": result.title,
                "notes": result.notes,
                "dataset_id": result.dataset_id,
                "dataset_name": result.dataset_name,
                "params": _dump(result.params),
                "result": _dump(result.result),
                "report": _dump(result.report),
                "figures": _dump(result.figures),
                "created_at": result.created_at,
                "updated_at": result.updated_at,
                "interpretation": None if result.interpretation is None else _dump(result.interpretation),
            }
        )
        self._core.bump_sync(self._user, "results")
        return result

    def get(self, result_id: str) -> SavedResult:
        doc = self._results.find_one({"id": result_id, "user_id": self._user})
        if doc is None:
            raise ResultNotFoundError("El resultado guardado no existe.")
        return SavedResult(
            **vars(_summary(doc)),
            params=json.loads(doc["params"]),
            result=json.loads(doc["result"]),
            report=json.loads(doc["report"]),
            figures=json.loads(doc["figures"]),
            interpretation=json.loads(doc["interpretation"]) if doc.get("interpretation") else None,
        )

    def list(
        self, module: str | None, search: str | None, limit: int, offset: int
    ) -> tuple[list[SavedResultSummary], int]:
        query: dict[str, Any] = {"user_id": self._user}
        if module:
            query["module"] = module
        if search:
            pattern = {"$regex": re.escape(search), "$options": "i"}
            query["$or"] = [{"title": pattern}, {"notes": pattern}, {"dataset_name": pattern}]
        total = self._results.count_documents(query)
        docs = self._results.find(query).sort([("created_at", -1), ("id", 1)]).skip(int(offset)).limit(int(limit))
        return [_summary(doc) for doc in docs], total

    def update(
        self, result_id: str, title: str | None, notes: str | None, interpretation: object = UNSET
    ) -> SavedResult:
        self.get(result_id)
        changes: dict[str, Any] = {"updated_at": now_iso()}
        if title is not None:
            changes["title"] = title
        if notes is not None:
            changes["notes"] = notes
        if interpretation is not UNSET:  # None removes it
            changes["interpretation"] = None if interpretation is None else _dump(interpretation)
        self._results.update_one({"id": result_id, "user_id": self._user}, {"$set": changes})
        self._core.bump_sync(self._user, "results")
        return self.get(result_id)

    def delete(self, result_id: str) -> None:
        self.get(result_id)
        self._results.delete_one({"id": result_id, "user_id": self._user})
        self._core.bump_sync(self._user, "results")


# ------------------------------------------------------------------ workspace and sync
def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class MongoWorkspaceRepository:
    """One document per (user, module). Revisions only ever grow, so a client can tell any change from its own."""

    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._workspace = core.col("workspace")
        self._user = int(user_id)

    def get(self, module: str) -> WorkspaceEntry:
        doc = self._workspace.find_one({"user_id": self._user, "module": module})
        if doc is None:
            return WorkspaceEntry(module)
        return WorkspaceEntry(
            module, _load(doc.get("state")), _load(doc.get("payload")), doc["revision"], doc["updated_at"]
        )

    def save(self, module: str, state: Any = UNSET, payload: Any = UNSET) -> WorkspaceEntry:
        """Partial update: a field left as ``UNSET`` keeps its stored value; ``None`` clears it."""
        changes: dict[str, Any] = {"updated_at": now_ms()}
        defaults: dict[str, Any] = {}
        for name, value in (("state", state), ("payload", payload)):
            if value is UNSET:
                defaults[name] = None
            else:
                changes[name] = None if value is None else _dump(value)
        self._workspace.update_one(
            {"user_id": self._user, "module": module},
            {"$set": changes, "$setOnInsert": defaults, "$inc": {"revision": 1}},
            upsert=True,
        )
        return self.get(module)

    def clear(self, module: str) -> int:
        """Empty a module but keep the document, so its revision keeps growing (other windows notice the change)."""
        return self._clear([module])[module]

    def clear_all(self) -> dict[str, int]:
        modules = [doc["module"] for doc in self._workspace.find({"user_id": self._user}, {"module": 1})]
        return self._clear(modules)

    def _clear(self, modules: list[str]) -> dict[str, int]:
        revisions: dict[str, int] = {}
        for module in modules:
            self._workspace.update_one(
                {"user_id": self._user, "module": module},
                {"$set": {"state": None, "payload": None, "updated_at": now_ms()}, "$inc": {"revision": 1}},
                upsert=True,
            )
            revisions[module] = self._workspace.find_one({"user_id": self._user, "module": module})["revision"]
        return revisions

    def revisions(self) -> dict[str, int]:
        return {d["module"]: d["revision"] for d in self._workspace.find({"user_id": self._user})}


class MongoSyncRepository:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._state = core.col("sync_state")
        self._user = int(user_id)

    def revisions(self) -> dict[str, int]:
        return {doc["scope"]: doc["revision"] for doc in self._state.find({"user_id": self._user})}
