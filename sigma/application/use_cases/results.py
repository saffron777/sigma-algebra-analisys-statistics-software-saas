"""Use cases for results the user chose to keep."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from sigma.application.dto import SaveResultCommand
from sigma.application.ports import DatasetRepository, ResultRepository
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError
from sigma.domain.saved_results import MODULES, SavedResult, SavedResultSummary

MAX_TITLE_LENGTH = 200
MAX_NOTES_LENGTH = 5000
MAX_PAYLOAD_BYTES = 5 * 1024 * 1024
MAX_PAGE_SIZE = 200


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_text(title: str | None, notes: str | None) -> None:
    if title is not None and not title.strip():
        raise InvalidInputError("El título no puede estar vacío.")
    if title is not None and len(title) > MAX_TITLE_LENGTH:
        raise InvalidInputError(f"El título no puede superar {MAX_TITLE_LENGTH} caracteres.")
    if notes is not None and len(notes) > MAX_NOTES_LENGTH:
        raise InvalidInputError(f"Las notas no pueden superar {MAX_NOTES_LENGTH} caracteres.")


class SaveResult:
    def __init__(self, results: ResultRepository, datasets: DatasetRepository) -> None:
        self._results = results
        self._datasets = datasets

    def execute(self, cmd: SaveResultCommand) -> SavedResult:
        if cmd.module not in MODULES:
            raise InvalidInputError(f"Módulo inválido «{cmd.module}».")
        _validate_text(cmd.title, cmd.notes)
        if not cmd.report and not cmd.result:
            raise InvalidInputError("No hay ningún resultado que guardar.")
        payload = json.dumps([cmd.params, cmd.result, cmd.report, cmd.figures], ensure_ascii=False)
        if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise InvalidInputError("El resultado supera el tamaño máximo permitido (5 MB).")
        dataset_id, dataset_name = None, None
        if cmd.dataset_id:
            try:
                info = self._datasets.get_info(cmd.dataset_id)
                dataset_id, dataset_name = info.id, info.name
            except DatasetNotFoundError:
                pass  # the dataset was deleted meanwhile; keep the result without a link
        now = _now()
        return self._results.add(
            SavedResult(
                id=uuid.uuid4().hex,
                module=cmd.module,
                analysis_type=cmd.analysis_type[:80],
                title=cmd.title.strip(),
                notes=cmd.notes.strip(),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                created_at=now,
                updated_at=now,
                params=cmd.params,
                result=cmd.result,
                report=cmd.report,
                figures=cmd.figures,
            )
        )


class ListResults:
    def __init__(self, results: ResultRepository) -> None:
        self._results = results

    def execute(
        self, module: str | None = None, search: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[SavedResultSummary], int]:
        if module and module not in MODULES:
            raise InvalidInputError(f"Módulo inválido «{module}».")
        return self._results.list(
            module or None, (search or "").strip() or None, max(1, min(limit, MAX_PAGE_SIZE)), max(0, offset)
        )


class GetResult:
    def __init__(self, results: ResultRepository) -> None:
        self._results = results

    def execute(self, result_id: str) -> SavedResult:
        return self._results.get(result_id)


class UpdateResult:
    def __init__(self, results: ResultRepository) -> None:
        self._results = results

    def execute(self, result_id: str, title: str | None, notes: str | None) -> SavedResult:
        _validate_text(title, notes)
        return self._results.update(
            result_id, title.strip() if title is not None else None, notes.strip() if notes is not None else None
        )


class DeleteResult:
    def __init__(self, results: ResultRepository) -> None:
        self._results = results

    def execute(self, result_id: str) -> None:
        self._results.delete(result_id)


class ExportResult:
    """Full content of a saved result as a plain dict, ready to be downloaded as JSON."""

    def __init__(self, results: ResultRepository) -> None:
        self._results = results

    def execute(self, result_id: str) -> dict[str, Any]:
        return asdict(self._results.get(result_id))
