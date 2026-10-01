"""SQLite-backed store for saved analysis results, scoped to one user."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from sigma.domain.errors import ResultNotFoundError
from sigma.domain.saved_results import SavedResult, SavedResultSummary
from sigma.domain.workspace import UNSET
from sigma.infrastructure.persistence.sqlite.database import Database

_SUMMARY_COLUMNS = "id, module, analysis_type, title, notes, dataset_id, dataset_name, created_at, updated_at"


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SqliteResultRepository:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    def add(self, result: SavedResult) -> SavedResult:
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO saved_results (id, user_id, module, analysis_type, title, notes, dataset_id, dataset_name,"
                " params, result, report, figures, created_at, updated_at, interpretation)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.id,
                    self._user,
                    result.module,
                    result.analysis_type,
                    result.title,
                    result.notes,
                    result.dataset_id,
                    result.dataset_name,
                    _dump(result.params),
                    _dump(result.result),
                    _dump(result.report),
                    _dump(result.figures),
                    result.created_at,
                    result.updated_at,
                    None if result.interpretation is None else _dump(result.interpretation),
                ),
            )
        return result

    def get(self, result_id: str) -> SavedResult:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM saved_results WHERE id = ? AND user_id = ?", (result_id, self._user)
            ).fetchone()
        if row is None:
            raise ResultNotFoundError("El resultado guardado no existe.")
        return SavedResult(
            id=row["id"],
            module=row["module"],
            analysis_type=row["analysis_type"],
            title=row["title"],
            notes=row["notes"],
            dataset_id=row["dataset_id"],
            dataset_name=row["dataset_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            params=json.loads(row["params"]),
            result=json.loads(row["result"]),
            report=json.loads(row["report"]),
            figures=json.loads(row["figures"]),
            interpretation=json.loads(row["interpretation"]) if row["interpretation"] else None,
        )

    def list(
        self, module: str | None, search: str | None, limit: int, offset: int
    ) -> tuple[list[SavedResultSummary], int]:
        clauses: list[str] = ["user_id = ?"]
        params: list[object] = [self._user]
        if module:
            clauses.append("module = ?")
            params.append(module)
        if search:
            pattern = f"%{_escape_like(search)}%"
            clauses.append("(title LIKE ? ESCAPE '\\' OR notes LIKE ? ESCAPE '\\' OR dataset_name LIKE ? ESCAPE '\\')")
            params.extend([pattern] * 3)
        where = f" WHERE {' AND '.join(clauses)}"
        with self._db.connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM saved_results{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM saved_results{where} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_summary(row) for row in rows], int(total)

    def update(
        self, result_id: str, title: str | None, notes: str | None, interpretation: object = UNSET
    ) -> SavedResult:
        self.get(result_id)
        with self._db.connection() as conn:
            if interpretation is not UNSET:  # None removes it
                conn.execute(
                    "UPDATE saved_results SET interpretation = ? WHERE id = ? AND user_id = ?",
                    (None if interpretation is None else _dump(interpretation), result_id, self._user),
                )
            if title is not None:
                conn.execute(
                    "UPDATE saved_results SET title = ? WHERE id = ? AND user_id = ?", (title, result_id, self._user)
                )
            if notes is not None:
                conn.execute(
                    "UPDATE saved_results SET notes = ? WHERE id = ? AND user_id = ?", (notes, result_id, self._user)
                )
            conn.execute(
                "UPDATE saved_results SET updated_at = ? WHERE id = ? AND user_id = ?",
                (datetime.now(UTC).isoformat(timespec="seconds"), result_id, self._user),
            )
        return self.get(result_id)

    def delete(self, result_id: str) -> None:
        self.get(result_id)
        with self._db.connection() as conn:
            conn.execute("DELETE FROM saved_results WHERE id = ? AND user_id = ?", (result_id, self._user))


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _summary(row: sqlite3.Row) -> SavedResultSummary:
    # sqlite3.Row iterates over its *values*, so the column names must come from .keys()
    return SavedResultSummary(**{key: row[key] for key in row.keys()})  # noqa: SIM118
