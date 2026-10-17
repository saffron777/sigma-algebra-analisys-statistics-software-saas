"""The MongoDB handle shared by every repository: collections, id counters, sync revisions and the schema marker.

Documents mirror the rows of the SQL schema (same field names and values), so records move between engines unchanged.
There are no multi-document transactions (a standalone server has none): every write is atomic per document, and the
few operations that touch several documents clean up after themselves in code.
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.database import Database

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.schema import SCHEMA_VERSION

# collection -> indexes as (fields, unique)
_INDEXES: dict[str, list[tuple[list[tuple[str, int]], bool]]] = {
    "users": [([("id", 1)], True), ([("username_key", 1)], True)],
    "datasets": [([("id", 1)], True), ([("user_id", 1), ("name_key", 1)], True)],
    "frame_chunks": [([("owner", 1), ("seq", 1)], True)],
    "saved_results": [([("id", 1)], True), ([("user_id", 1), ("created_at", -1)], False)],
    "sync_state": [([("user_id", 1), ("scope", 1)], True)],
    "workspace": [([("user_id", 1), ("module", 1)], True)],
    "import_drafts": [([("id", 1)], True), ([("user_id", 1), ("source_type", 1)], True)],
    "import_draft_files": [([("draft_id", 1), ("seq", 1)], True)],
    "report_templates": [([("id", 1)], True), ([("user_id", 1), ("name_key", 1)], True)],
    "projects": [([("id", 1)], True), ([("user_id", 1), ("name_key", 1)], True)],
    "app_settings": [([("key", 1)], True)],
    "ai_usage": [([("id", 1)], True), ([("user_id", 1), ("created_at", 1)], False)],
    "notebooks": [([("id", 1)], True), ([("user_id", 1)], False)],
}
MARKER = "sigma_schema"


class MongoCore:
    """``client`` may be any pymongo-compatible client (the tests inject mongomock)."""

    def __init__(self, client: MongoClient, database_name: str, settings: DatabaseSettings) -> None:
        self.client = client
        self.db: Database = client[database_name]
        self.settings = settings
        self.ensure_schema()

    def col(self, name: str) -> Any:
        return self.db[name]

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> None:
        names = set(self.db.list_collection_names())
        if names and MARKER not in names:
            raise InvalidInputError(
                "La base de datos elegida ya contiene colecciones que no son de Sigma Algebra. Use una base vacía."
            )
        marker = self.db[MARKER].find_one({"_id": 1})
        if marker and int(marker["version"]) > SCHEMA_VERSION:
            raise InvalidInputError("La base de datos fue creada por una versión más reciente de Sigma Algebra.")
        for collection, indexes in _INDEXES.items():
            for fields, unique in indexes:
                self.db[collection].create_index([(f, d if d else ASCENDING) for f, d in fields], unique=unique)
        self.db[MARKER].update_one({"_id": 1}, {"$set": {"version": SCHEMA_VERSION}}, upsert=True)

    # ------------------------------------------------------------------ counters and revisions
    def next_id(self, name: str) -> int:
        """Atomic integer ids for ``users`` and ``ai_usage`` (the SQL schema has auto-increment columns)."""
        doc = self.db["counters"].find_one_and_update(
            {"_id": name}, {"$inc": {"seq": 1}}, upsert=True, return_document=ReturnDocument.AFTER
        )
        return int(doc["seq"])

    def set_counter(self, name: str, value: int) -> None:
        self.db["counters"].update_one({"_id": name}, {"$max": {"seq": int(value)}}, upsert=True)

    def bump_sync(self, user_id: int | None, scope: str) -> None:
        if user_id is None:
            return
        self.db["sync_state"].update_one({"user_id": user_id, "scope": scope}, {"$inc": {"revision": 1}}, upsert=True)


def connect(settings: DatabaseSettings, timeout_ms: int = 15000) -> MongoCore:
    from sigma.infrastructure.persistence.database_settings import ENGINES, driver_available

    if not driver_available("mongodb"):
        raise InvalidInputError(
            f"Falta el controlador de MongoDB. Instálelo con: pip install {ENGINES['mongodb']['pip']}"
        )
    client = MongoClient(settings.to_url(), serverSelectionTimeoutMS=timeout_ms)
    try:
        return MongoCore(client, settings.database, settings)
    except BaseException:
        client.close()
        raise


def strip(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """A copy without Mongo's own ``_id`` (the documents carry their real keys in ``id``/``key``…)."""
    if document is None:
        return None
    return {k: v for k, v in document.items() if k != "_id"}
