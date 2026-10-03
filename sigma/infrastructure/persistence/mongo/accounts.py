"""Accounts, server settings and the AI usage log on MongoDB."""

from __future__ import annotations

from typing import Any

from pymongo.errors import DuplicateKeyError

from sigma.domain.errors import InvalidInputError, UserNotFoundError
from sigma.domain.users import User
from sigma.infrastructure.persistence.mongo.core import MongoCore, strip
from sigma.infrastructure.persistence.sql.helpers import now_iso


def _to_user(doc: dict[str, Any]) -> User:
    return User(
        id=doc["id"],
        username=doc["username"],
        display_name=doc["display_name"],
        role=doc["role"],
        can_use_console=bool(doc["can_use_console"]),
        can_use_ai=bool(doc["can_use_ai"]),
        is_active=bool(doc["is_active"]),
        auth_epoch=doc["auth_epoch"],
        created_at=doc["created_at"],
        last_login_at=doc.get("last_login_at"),
    )


class MongoUserRepository:
    def __init__(self, core: MongoCore) -> None:
        self._users = core.col("users")
        self._core = core

    def count(self) -> int:
        return self._users.count_documents({})

    def count_active_admins(self, excluding: int | None = None) -> int:
        query: dict[str, Any] = {"role": "admin", "is_active": 1}
        if excluding:
            query["id"] = {"$ne": excluding}
        return self._users.count_documents(query)

    def add(
        self,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        can_use_console: bool,
        can_use_ai: bool,
    ) -> User:
        user_id = self._core.next_id("users")
        try:
            self._users.insert_one(
                {
                    "id": user_id,
                    "username": username,
                    "username_key": username.lower(),
                    "display_name": display_name,
                    "password_hash": password_hash,
                    "role": role,
                    "can_use_console": int(can_use_console),
                    "can_use_ai": int(can_use_ai),
                    "is_active": 1,
                    "auth_epoch": 1,
                    "created_at": now_iso(),
                    "last_login_at": None,
                }
            )
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe un usuario llamado «{username}».") from None
        return self.get(user_id)

    def get(self, user_id: int) -> User:
        doc = self._users.find_one({"id": user_id})
        if doc is None:
            raise UserNotFoundError("El usuario no existe.")
        return _to_user(doc)

    def find_with_hash(self, username: str) -> tuple[User, str] | None:
        doc = self._users.find_one({"username_key": (username or "").strip().lower()})
        return (_to_user(doc), doc["password_hash"]) if doc else None

    def list(self) -> list[User]:
        docs = self._users.find().sort([("role", 1), ("username", 1)])
        return [_to_user(doc) for doc in docs]

    def update(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        is_active: bool | None = None,
        can_use_console: bool | None = None,
        can_use_ai: bool | None = None,
    ) -> User:
        changes: dict[str, Any] = {
            "display_name": display_name,
            "role": role,
            "is_active": None if is_active is None else int(is_active),
            "can_use_console": None if can_use_console is None else int(can_use_console),
            "can_use_ai": None if can_use_ai is None else int(can_use_ai),
        }
        changes = {column: value for column, value in changes.items() if value is not None}
        self.get(user_id)
        if changes:
            operation: dict[str, Any] = {"$set": changes}
            if "is_active" in changes or "role" in changes:
                operation["$inc"] = {"auth_epoch": 1}  # their open sessions stop being valid
            self._users.update_one({"id": user_id}, operation)
        return self.get(user_id)

    def set_password(self, user_id: int, password_hash: str) -> User:
        self.get(user_id)
        self._users.update_one({"id": user_id}, {"$set": {"password_hash": password_hash}, "$inc": {"auth_epoch": 1}})
        return self.get(user_id)

    def touch_login(self, user_id: int) -> None:
        self._users.update_one({"id": user_id}, {"$set": {"last_login_at": now_iso()}})

    def delete(self, user_id: int) -> None:
        self.get(user_id)
        self._users.delete_one({"id": user_id})
        remove_user_rows(self._core, user_id)  # there is no ON DELETE CASCADE: everything of theirs goes here


def remove_user_rows(core: MongoCore, user_id: int) -> None:
    """Delete what an account owns in every collection (data chunks, drafts and their files included)."""
    from sigma.infrastructure.persistence.mongo.frames import COLLECTION as FRAMES

    for dataset in core.col("datasets").find({"user_id": user_id}, {"data_id": 1}):
        core.col(FRAMES).delete_many({"owner": dataset.get("data_id")})
    for draft in core.col("import_drafts").find({"user_id": user_id}, {"id": 1, "staged_owner": 1}):
        core.col(FRAMES).delete_many({"owner": draft.get("staged_owner")})
        core.col("import_draft_files").delete_many({"draft_id": draft["id"]})
    for name in (
        "datasets",
        "saved_results",
        "workspace",
        "sync_state",
        "import_drafts",
        "report_templates",
        "ai_usage",
        "notebooks",
    ):
        core.col(name).delete_many({"user_id": user_id})


class MongoSettingsRepository:
    def __init__(self, core: MongoCore) -> None:
        self._settings = core.col("app_settings")

    def get(self, key: str, default: str | None = None) -> str | None:
        doc = self._settings.find_one({"key": key})
        return doc["value"] if doc else default

    def set(self, key: str, value: str) -> None:
        self._settings.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)


class MongoAiUsageRepository:
    def __init__(self, core: MongoCore) -> None:
        self._core = core
        self._usage = core.col("ai_usage")

    def add(self, user_id: int, module: str, model: str, input_tokens: int, output_tokens: int) -> None:
        self._usage.insert_one(
            {
                "id": self._core.next_id("ai_usage"),
                "user_id": user_id,
                "module": module,
                "model": model,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "created_at": now_iso(),
            }
        )

    def count_since(self, user_id: int, since_iso: str) -> int:
        return self._usage.count_documents({"user_id": user_id, "created_at": {"$gte": since_iso}})

    def oldest_since(self, user_id: int, since_iso: str) -> str | None:
        docs = self._usage.find({"user_id": user_id, "created_at": {"$gte": since_iso}}).sort("created_at", 1).limit(1)
        return next(iter(docs), {}).get("created_at")

    def summary(self, since_iso: str) -> list[dict[str, Any]]:
        """Requests and tokens per account since ``since_iso`` (for the administrator's page)."""
        totals: dict[int, list[int]] = {}
        for doc in self._usage.find({"created_at": {"$gte": since_iso}}):
            entry = totals.setdefault(doc["user_id"], [0, 0, 0])
            entry[0] += 1
            entry[1] += doc["input_tokens"]
            entry[2] += doc["output_tokens"]
        rows = [
            {
                "user_id": doc["id"],
                "username": doc["username"],
                "requests": totals.get(doc["id"], [0, 0, 0])[0],
                "input_tokens": totals.get(doc["id"], [0, 0, 0])[1],
                "output_tokens": totals.get(doc["id"], [0, 0, 0])[2],
            }
            for doc in strip_many(self._core.col("users").find())
        ]
        return sorted(rows, key=lambda row: (-row["requests"], row["username"]))


def strip_many(documents: Any) -> list[dict[str, Any]]:
    return [strip(d) for d in documents]
