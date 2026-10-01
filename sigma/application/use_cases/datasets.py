"""Use cases for importing and managing datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

import pandas as pd

from sigma.application.ports import (
    DatasetRepository,
    ImporterRegistry,
    ImportRequest,
    ImportStaging,
    StagedImport,
)
from sigma.domain.datasets import (
    ColumnInfo,
    Dataset,
    DatasetInfo,
    convert_column,
    describe_columns,
    infer_dates,
    normalize_columns,
    stringify_complex_cells,
)
from sigma.domain.errors import InvalidInputError

PREVIEW_ROWS = 50
MAX_NAME_LENGTH = 120


def clean_dataset_name(name: str | None, fallback: str = "dataset") -> str:
    cleaned = " ".join((name or "").split())[:MAX_NAME_LENGTH]
    return cleaned or fallback


@dataclass
class ImportPreview:
    token: str
    suggested_name: str
    n_rows: int
    n_cols: int
    columns: tuple[ColumnInfo, ...]
    rows: pd.DataFrame
    detected: dict


class PreviewImport:
    """Reads a source, normalises it and stages it so the user can review before saving."""

    def __init__(self, importers: ImporterRegistry, staging: ImportStaging) -> None:
        self._importers = importers
        self._staging = staging

    def execute(self, request: ImportRequest) -> ImportPreview:
        frame, detected = self._importers.get(request.source_type).read(request)
        frame = infer_dates(normalize_columns(stringify_complex_cells(frame)))
        if frame.empty and frame.shape[1] == 0:
            raise InvalidInputError("La fuente de datos no devolvió columnas.")
        suggested = clean_dataset_name(
            PurePath(request.file_name).stem
            if request.file_name
            else request.options.get("table") or request.options.get("name"),
            fallback=f"importado_{request.source_type}",
        )
        meta = {
            "source_type": request.source_type,
            "file_name": request.file_name,
            **{k: v for k, v in detected.items() if k != "password"},
        }
        token = self._staging.put(StagedImport(frame, request.source_type, meta, suggested, detected))
        return ImportPreview(
            token, suggested, len(frame), frame.shape[1], describe_columns(frame), frame.head(PREVIEW_ROWS), detected
        )


class InspectSource:
    """Asks an importer for extra choices before reading (e.g. the sheets of an Excel workbook)."""

    def __init__(self, importers: ImporterRegistry) -> None:
        self._importers = importers

    def execute(self, request: ImportRequest) -> dict:
        importer = self._importers.get(request.source_type)
        inspect = getattr(importer, "inspect", None)
        return inspect(request) if inspect else {}


class CommitImport:
    """Saves a staged import, optionally overriding column types."""

    def __init__(self, staging: ImportStaging, datasets: DatasetRepository) -> None:
        self._staging = staging
        self._datasets = datasets

    def execute(self, token: str, name: str | None, column_kinds: dict[str, str] | None = None) -> DatasetInfo:
        staged = self._staging.take(token)
        frame = staged.frame
        for column, kind in (column_kinds or {}).items():
            if column not in frame.columns:
                raise InvalidInputError(f"La columna «{column}» no existe.")
            frame[column], _ = convert_column(frame[column], kind)
        return self._datasets.add(
            clean_dataset_name(name, staged.suggested_name), frame, staged.source_type, staged.source_meta
        )


class ListDatasets:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self) -> list[DatasetInfo]:
        return self._datasets.list()


class GetDatasetInfo:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, dataset_id: str) -> DatasetInfo:
        return self._datasets.get_info(dataset_id)


class PreviewDataset:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, dataset_id: str, limit: int = PREVIEW_ROWS, offset: int = 0) -> tuple[DatasetInfo, pd.DataFrame]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        return self._datasets.get_info(dataset_id), self._datasets.preview(dataset_id, limit, offset)


class RenameDataset:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, dataset_id: str, name: str) -> DatasetInfo:
        return self._datasets.rename(dataset_id, clean_dataset_name(name, ""))


class DeleteDataset:
    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, dataset_id: str) -> None:
        self._datasets.delete(dataset_id)


class ChangeColumnTypes:
    """Converts columns of a saved dataset and reports values that could not be converted."""

    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, dataset_id: str, column_kinds: dict[str, str]) -> tuple[DatasetInfo, dict[str, int]]:
        if not column_kinds:
            raise InvalidInputError("Indique al menos una columna a convertir.")
        dataset: Dataset = self._datasets.get(dataset_id)
        frame = dataset.data
        failures: dict[str, int] = {}
        for column, kind in column_kinds.items():
            if column not in frame.columns:
                raise InvalidInputError(f"La columna «{column}» no existe.")
            frame[column], failures[column] = convert_column(frame[column], kind)
        return self._datasets.replace_data(dataset_id, frame), failures
