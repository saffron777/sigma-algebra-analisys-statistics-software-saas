"""Survey projects on any SQL engine, scoped to one user."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import case, delete, insert, select, update
from sqlalchemy.engine import Row
from sqlalchemy.exc import IntegrityError

from sigma.domain.errors import InvalidInputError, ProjectNotFoundError
from sigma.domain.projects import Project, ProjectSummary
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import bump_sync, now_iso

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_t = schema.projects
_PUBLIC = [
    _t.c.id,
    _t.c.name,
    _t.c.n_questions,
    _t.c.n_responses,
    _t.c.sample_size,
    _t.c.population_size,
    case((_t.c.results.is_not(None), 1), else_=0).label("has_results"),
    _t.c.created_at,
    _t.c.updated_at,
]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class SqlProjectRepository:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ writes
    def add(self, project: Project) -> Project:
        try:
            with self._db.begin() as conn:
                conn.execute(
                    insert(_t).values(
                        id=project.id,
                        user_id=self._user,
                        name=project.name,
                        name_key=project.name.lower(),
                        population_size=project.population_size,
                        sample_size=project.sample_size,
                        n_questions=len(project.design),
                        n_responses=project.n_responses,
                        indicators=_dumps(project.indicators),
                        design=_dumps(project.design),
                        dataset_id=project.dataset_id,
                        dataset_name=project.dataset_name,
                        results=_dumps(project.results) if project.results is not None else None,
                        created_at=project.created_at,
                        updated_at=project.updated_at,
                    )
                )
                bump_sync(conn, self._user, "projects")
        except IntegrityError:
            raise InvalidInputError(f"Ya existe un proyecto llamado «{project.name}».") from None
        return self.get(project.id)

    def update(self, project: Project) -> Project:
        self.get(project.id)
        try:
            with self._db.begin() as conn:
                conn.execute(
                    update(_t)
                    .where(_t.c.id == project.id, _t.c.user_id == self._user)
                    .values(
                        name=project.name,
                        name_key=project.name.lower(),
                        population_size=project.population_size,
                        sample_size=project.sample_size,
                        n_questions=len(project.design),
                        n_responses=project.n_responses,
                        indicators=_dumps(project.indicators),
                        design=_dumps(project.design),
                        dataset_id=project.dataset_id,
                        dataset_name=project.dataset_name,
                        results=_dumps(project.results) if project.results is not None else None,
                        updated_at=now_iso(),
                    )
                )
                bump_sync(conn, self._user, "projects")
        except IntegrityError:
            raise InvalidInputError(f"Ya existe un proyecto llamado «{project.name}».") from None
        return self.get(project.id)

    def delete(self, project_id: str) -> None:
        self.get(project_id)
        with self._db.begin() as conn:
            conn.execute(delete(_t).where(_t.c.id == project_id, _t.c.user_id == self._user))
            bump_sync(conn, self._user, "projects")

    # ------------------------------------------------------------------ reads
    def get(self, project_id: str) -> Project:
        if not _ID_PATTERN.match(project_id or ""):
            raise ProjectNotFoundError("El proyecto no existe.")
        with self._db.begin() as conn:
            row = conn.execute(select(_t).where(_t.c.id == project_id, _t.c.user_id == self._user)).first()
        if row is None:
            raise ProjectNotFoundError("El proyecto no existe.")
        return self._to_project(row)

    def list(self) -> list[ProjectSummary]:
        with self._db.begin() as conn:
            rows = conn.execute(
                select(*_PUBLIC).where(_t.c.user_id == self._user).order_by(_t.c.updated_at.desc(), _t.c.name)
            ).all()
        return [self._to_summary(row) for row in rows]

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _to_project(row: Row) -> Project:
        return Project(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            population_size=row.population_size,
            sample_size=row.sample_size,
            indicators=json.loads(row.indicators),
            design=json.loads(row.design),
            dataset_id=row.dataset_id,
            dataset_name=row.dataset_name,
            n_responses=int(row.n_responses or 0),
            results=json.loads(row.results) if row.results else None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_summary(row: Row) -> ProjectSummary:
        return ProjectSummary(
            id=row.id,
            name=row.name,
            n_questions=int(row.n_questions),
            n_responses=int(row.n_responses),
            sample_size=row.sample_size,
            population_size=row.population_size,
            has_results=bool(row.has_results),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
