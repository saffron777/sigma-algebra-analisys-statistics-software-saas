"""Survey projects on MongoDB (document collection)."""

from __future__ import annotations

import re
from typing import Any

from pymongo.errors import DuplicateKeyError

from sigma.domain.errors import InvalidInputError, ProjectNotFoundError
from sigma.domain.projects import Project, ProjectSummary
from sigma.infrastructure.persistence.mongo.core import MongoCore
from sigma.infrastructure.persistence.sql.helpers import now_iso

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class MongoProjectRepository:
    def __init__(self, core: MongoCore, user_id: int) -> None:
        self._core = core
        self._user = int(user_id)
        self._projects = core.col("projects")

    def add(self, project: Project) -> Project:
        doc = {
            "id": project.id,
            "user_id": self._user,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            **self._values(project),
        }
        try:
            self._projects.insert_one(doc)
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe un proyecto llamado «{project.name}».") from None
        self._core.bump_sync(self._user, "projects")
        return self.get(project.id)

    def update(self, project: Project) -> Project:
        self.get(project.id)
        try:
            self._projects.update_one(
                {"id": project.id, "user_id": self._user},
                {"$set": {**self._values(project), "updated_at": now_iso()}},
            )
        except DuplicateKeyError:
            raise InvalidInputError(f"Ya existe un proyecto llamado «{project.name}».") from None
        self._core.bump_sync(self._user, "projects")
        return self.get(project.id)

    def delete(self, project_id: str) -> None:
        self._check_id(project_id)
        if not self._projects.delete_one({"id": project_id, "user_id": self._user}).deleted_count:
            raise ProjectNotFoundError("El proyecto no existe.")
        self._core.bump_sync(self._user, "projects")

    def get(self, project_id: str) -> Project:
        return self._to_project(self._doc(project_id))

    def list(self) -> list[ProjectSummary]:
        docs = self._projects.find({"user_id": self._user}).sort([("updated_at", -1), ("name", 1)])
        return [self._to_summary(d) for d in docs]

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _values(project: Project) -> dict[str, Any]:
        return {
            "name": project.name,
            "name_key": project.name.lower(),
            "population_size": project.population_size,
            "sample_size": project.sample_size,
            "n_questions": len(project.design),
            "n_responses": project.n_responses,
            "indicators": project.indicators,
            "design": project.design,
            "dataset_id": project.dataset_id,
            "dataset_name": project.dataset_name,
            "results": project.results,
        }

    @staticmethod
    def _to_project(doc: dict[str, Any]) -> Project:
        return Project(
            id=doc["id"],
            user_id=doc["user_id"],
            name=doc["name"],
            population_size=doc.get("population_size"),
            sample_size=doc.get("sample_size"),
            indicators=doc.get("indicators") or [],
            design=doc.get("design") or [],
            dataset_id=doc.get("dataset_id"),
            dataset_name=doc.get("dataset_name"),
            n_responses=int(doc.get("n_responses") or 0),
            results=doc.get("results"),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    @staticmethod
    def _to_summary(doc: dict[str, Any]) -> ProjectSummary:
        return ProjectSummary(
            id=doc["id"],
            name=doc["name"],
            n_questions=int(doc.get("n_questions") or 0),
            n_responses=int(doc.get("n_responses") or 0),
            sample_size=doc.get("sample_size"),
            population_size=doc.get("population_size"),
            has_results=doc.get("results") is not None,
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    def _doc(self, project_id: str) -> dict[str, Any]:
        self._check_id(project_id)
        doc = self._projects.find_one({"id": project_id, "user_id": self._user})
        if doc is None:
            raise ProjectNotFoundError("El proyecto no existe.")
        return doc

    @staticmethod
    def _check_id(project_id: str) -> None:
        if not _ID_PATTERN.match(project_id or ""):
            raise ProjectNotFoundError("El proyecto no existe.")
