"""Whole-database copy for MongoDB: the same plain records as ``SqlTransfer`` (see ``MigrateData``)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sigma.infrastructure.persistence.mongo import frames
from sigma.infrastructure.persistence.mongo.core import MongoCore, strip
from sigma.infrastructure.persistence.mongo.documents import CHUNK_BYTES

# entity -> collection (they share names)
_COLLECTIONS = (
    "users",
    "app_settings",
    "report_templates",
    "datasets",
    "saved_results",
    "workspace",
    "sync_state",
    "import_drafts",
    "ai_usage",
    "notebooks",
)
_DERIVED = ("frame_chunks", "import_draft_files")  # emptied together with their owners
_BATCH = 200
_SQL_ONLY_DRAFT_FIELDS = ("staged_table", "col_map")


class MongoTransfer:
    def __init__(self, core: MongoCore, notebooks_root: Path | None = None) -> None:
        self._core = core

    def counts(self) -> dict[str, int]:
        return {name: self._core.col(name).count_documents({}) for name in _COLLECTIONS}

    def is_empty(self) -> bool:
        return not any(self.counts().values())

    # ------------------------------------------------------------------ reading
    def read(self, entity: str) -> Iterator[dict[str, Any]]:
        collection = self._core.col(entity)
        for document in collection.find().sort("_id", 1):
            record = strip(document)
            if entity == "datasets":
                data_id = record.pop("data_id")
                dtypes = json.loads(record["dtypes"])
                record["frame"] = frames.read_frame(self._core, data_id, list(dtypes), dtypes)
            elif entity == "import_drafts":
                record.pop("staged_owner", None)
                chunks = list(self._core.col("import_draft_files").find({"draft_id": record["id"]}).sort("seq", 1))
                record["file"] = b"".join(bytes(c["data"]) for c in chunks) if chunks else None
            yield record

    # ------------------------------------------------------------------ writing
    def write(self, entity: str, records: Iterable[dict[str, Any]]) -> int:
        if entity == "datasets":
            return self._write_datasets(records)
        if entity == "import_drafts":
            return self._write_drafts(records)
        collection = self._core.col(entity)
        count, batch = 0, []
        for record in records:
            batch.append(dict(record))
            if len(batch) >= _BATCH:
                collection.insert_many(batch)
                count, batch = count + len(batch), []
        if batch:
            collection.insert_many(batch)
            count += len(batch)
        return count

    def _write_datasets(self, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            frame = record.pop("frame")
            data_id = frames.new_owner("ds")
            frames.write_frame(self._core, data_id, frame)
            self._core.col("datasets").insert_one({**record, "data_id": data_id})
            count += 1
        return count

    def _write_drafts(self, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            content = record.pop("file")
            document = {k: v for k, v in record.items() if k not in _SQL_ONLY_DRAFT_FIELDS}
            document.update(
                staged_owner=None,
                n_rows=None,
                n_cols=None,
                dtypes=None,
                columns_info=None,
                detected="{}",
                suggested_name=None,
            )
            self._core.col("import_drafts").insert_one(document)
            if content is not None:
                for seq, start in enumerate(range(0, max(len(content), 1), CHUNK_BYTES)):
                    self._core.col("import_draft_files").insert_one(
                        {"draft_id": record["id"], "seq": seq, "data": content[start : start + CHUNK_BYTES]}
                    )
            count += 1
        return count

    # ------------------------------------------------------------------ after the copy
    def finish(self) -> None:
        for counter in ("users", "ai_usage"):
            top = self._core.col(counter).find_one(sort=[("id", -1)])
            if top:
                self._core.set_counter(counter, top["id"])

    def clear(self) -> None:
        """Empty what ``write`` filled (only ever called on a target that was empty when the migration began)."""
        for name in (*_COLLECTIONS, *_DERIVED):
            self._core.col(name).delete_many({})
        self._core.col("counters").delete_many({})
