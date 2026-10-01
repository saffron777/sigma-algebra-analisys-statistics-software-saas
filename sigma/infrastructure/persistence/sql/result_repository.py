"""Saved analysis results on any SQL engine, scoped to one user."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Row

from sigma.domain.errors import ResultNotFoundError
from sigma.domain.saved_results import SavedResult, SavedResultSummary
from sigma.domain.workspace import UNSET
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import bump_sync, like_pattern, now_iso

_r = schema.saved_results
_SUMMARY = [
    _r.c.id,
    _r.c.module,
    _r.c.analysis_type,
    _r.c.title,
    _r.c.notes,
    _r.c.dataset_id,
    _r.c.dataset_name,
    _r.c.created_at,
    _r.c.updated_at,
]


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _summary(row: Row) -> SavedResultSummary:
    return SavedResultSummary(
        id=row.id,
        module=row.module,
        analysis_type=row.analysis_type,
        title=row.title,
        notes=row.notes or "",  # Oracle stores '' as NULL
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlResultRepository:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def add(self, result: SavedResult) -> SavedResult:
        with self._db.begin() as conn:
            conn.execute(
                insert(_r).values(
                    id=result.id,
                    user_id=self._user,
                    module=result.module,
                    analysis_type=result.analysis_type,
                    title=result.title,
                    notes=result.notes,
                    dataset_id=result.dataset_id,
                    dataset_name=result.dataset_name,
                    params=_dump(result.params),
                    result=_dump(result.result),
                    report=_dump(result.report),
                    figures=_dump(result.figures),
                    created_at=result.created_at,
                    updated_at=result.updated_at,
                    interpretation=None if result.interpretation is None else _dump(result.interpretation),
                )
            )
            bump_sync(conn, self._user, "results")
        return result

    def get(self, result_id: str) -> SavedResult:
        with self._db.begin() as conn:
            row = conn.execute(select(_r).where(_r.c.id == result_id, _r.c.user_id == self._user)).first()
        if row is None:
            raise ResultNotFoundError("El resultado guardado no existe.")
        summary = _summary(row)
        return SavedResult(
            **vars(summary),
            params=json.loads(row.params),
            result=json.loads(row.result),
            report=json.loads(row.report),
            figures=json.loads(row.figures),
            interpretation=json.loads(row.interpretation) if row.interpretation else None,
        )

    def list(
        self, module: str | None, search: str | None, limit: int, offset: int
    ) -> tuple[list[SavedResultSummary], int]:
        conditions: list[Any] = [_r.c.user_id == self._user]
        if module:
            conditions.append(_r.c.module == module)
        if search:
            pattern = like_pattern(search)
            conditions.append(
                func.lower(_r.c.title).like(pattern, escape="\\")
                | func.lower(_r.c.notes).like(pattern, escape="\\")
                | func.lower(_r.c.dataset_name).like(pattern, escape="\\")
            )
        with self._db.begin() as conn:
            total = conn.execute(select(func.count()).select_from(_r).where(*conditions)).scalar_one()
            rows = conn.execute(
                select(*_SUMMARY)
                .where(*conditions)
                .order_by(_r.c.created_at.desc(), _r.c.id)
                .limit(limit)
                .offset(offset)
            ).all()
        return [_summary(row) for row in rows], int(total)

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
        with self._db.begin() as conn:
            conn.execute(update(_r).where(_r.c.id == result_id, _r.c.user_id == self._user).values(**changes))
            bump_sync(conn, self._user, "results")
        return self.get(result_id)

    def delete(self, result_id: str) -> None:
        self.get(result_id)
        with self._db.begin() as conn:
            conn.execute(delete(_r).where(_r.c.id == result_id, _r.c.user_id == self._user))
            bump_sync(conn, self._user, "results")
