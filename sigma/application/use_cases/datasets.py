"""Use cases for importing and managing datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any

import pandas as pd

from sigma.application.ports import (
    DatasetRepository,
    DraftInfo,
    ImportDraftStore,
    ImporterRegistry,
    ImportRequest,
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
# options that must never be written to the database (a pending import is stored, a password must not be)
SECRET_OPTIONS = {
    "sqlserver": ("password",),
    "mysql": ("password",),
    "postgresql": ("password",),
    "oracle": ("password",),
    "mongodb": ("password",),
    "json_api": ("headers", "body", "params"),
}


def safe_options(source_type: str, options: dict[str, Any]) -> dict[str, Any]:
    """The import options that are fine to keep with a draft: no passwords, headers, bodies or URL query strings."""
    kept = {k: v for k, v in options.items() if k not in SECRET_OPTIONS.get(source_type, ())}
    if isinstance(kept.get("url"), str):
        kept["url"] = kept["url"].split("?", 1)[0].split("#", 1)[0]  # a token often travels in the query string
    return kept


def clean_dataset_name(name: str | None, fallback: str = "dataset") -> str:
    cleaned = " ".join((name or "").split())[:MAX_NAME_LENGTH]
    return cleaned or fallback


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The normalisation every imported table goes through before it is shown or stored."""
    return infer_dates(normalize_columns(stringify_complex_cells(frame)))


@dataclass
class ImportPreview:
    token: str
    suggested_name: str
    n_rows: int
    n_cols: int
    columns: tuple[ColumnInfo, ...]
    rows: pd.DataFrame
    detected: dict


def _with_stored_file(drafts: ImportDraftStore, request: ImportRequest, draft_id: str | None) -> ImportRequest:
    """A request that names a draft reads that draft's saved file instead of a new upload."""
    if draft_id is None or request.file_content is not None:
        return request
    info, content = drafts.file(draft_id)
    if info.source_type != request.source_type:
        raise InvalidInputError("El archivo guardado corresponde a otro tipo de fuente.")
    if content is None:
        return request
    return replace(request, file_name=info.file_name, file_content=content)


class PreviewImport:
    """Reads a source, normalises it and keeps it as the user's pending import so they can review it before saving."""

    def __init__(self, importers: ImporterRegistry, drafts: ImportDraftStore) -> None:
        self._importers = importers
        self._drafts = drafts

    def execute(self, request: ImportRequest, draft_id: str | None = None) -> ImportPreview:
        token = draft_id
        if request.file_content is not None:  # a file arrived with the request: keep it as the draft's file
            token = self._drafts.attach_file(request.source_type, request.file_name or "", request.file_content).id
        else:
            request = _with_stored_file(self._drafts, request, draft_id)
        frame, detected = self._importers.get(request.source_type).read(request)
        frame = prepare_frame(frame)
        if frame.empty and frame.shape[1] == 0:
            raise InvalidInputError("La fuente de datos no devolvió columnas.")
        suggested = clean_dataset_name(
            PurePath(request.file_name).stem
            if request.file_name
            else request.options.get("table") or request.options.get("collection") or request.options.get("name"),
            fallback=f"importado_{request.source_type}",
        )
        meta = {
            "source_type": request.source_type,
            "file_name": request.file_name,
            **{k: v for k, v in detected.items() if k != "password"},
        }
        token = self._drafts.put(
            StagedImport(frame, request.source_type, meta, suggested, detected),
            token,
            safe_options(request.source_type, request.options),
        )
        return ImportPreview(
            token, suggested, len(frame), frame.shape[1], describe_columns(frame), frame.head(PREVIEW_ROWS), detected
        )


class LoadSampleData:
    """Imports the demo CSV files (``sample_data/``) into the user's datasets, skipping names already in use."""

    def __init__(self, importers: ImporterRegistry, datasets: DatasetRepository, sample_dir: Path) -> None:
        self._importers = importers
        self._datasets = datasets
        self._sample_dir = Path(sample_dir)

    def execute(self) -> dict[str, list[str]]:
        existing = {d.name.lower() for d in self._datasets.list()}
        created: list[str] = []
        skipped: list[str] = []
        reader = self._importers.get("csv")
        for path in sorted(self._sample_dir.glob("*.csv")):
            if path.stem.lower() in existing:
                skipped.append(path.stem)
                continue
            frame, detected = reader.read(ImportRequest("csv", {}, path.name, path.read_bytes()))
            meta = {"source_type": "csv", "file_name": path.name, **detected}
            self._datasets.add(path.stem, prepare_frame(frame), "csv", meta)
            created.append(path.stem)
        return {"created": created, "skipped": skipped}


class InspectSource:
    """Asks an importer for extra choices before reading (e.g. the sheets of an Excel workbook)."""

    def __init__(self, importers: ImporterRegistry, drafts: ImportDraftStore) -> None:
        self._importers = importers
        self._drafts = drafts

    def execute(self, request: ImportRequest, draft_id: str | None = None) -> dict:
        request = _with_stored_file(self._drafts, request, draft_id)
        importer = self._importers.get(request.source_type)
        inspect = getattr(importer, "inspect", None)
        return inspect(request) if inspect else {}


class ListExternalDatabases:
    """The database engines the data page can read from, with whether this server has each driver."""

    def __init__(self, importers: ImporterRegistry) -> None:
        self._importers = importers

    def execute(self) -> list[dict]:
        return self._importers.external_databases()


class UploadImportFile:
    """Keeps an uploaded file on the server so it survives page changes (it becomes the draft of that source)."""

    def __init__(self, importers: ImporterRegistry, drafts: ImportDraftStore) -> None:
        self._importers = importers
        self._drafts = drafts

    def execute(self, source_type: str, file_name: str, content: bytes) -> DraftInfo:
        self._importers.get(source_type)  # rejects unknown source types
        if not content:
            raise InvalidInputError("El archivo está vacío.")
        return self._drafts.attach_file(source_type, file_name, content)


class ListImportDrafts:
    def __init__(self, drafts: ImportDraftStore) -> None:
        self._drafts = drafts

    def execute(self) -> list[DraftInfo]:
        return self._drafts.list()


class GetImportDraft:
    """Rebuilds the preview of a pending import (after a page change or in another window)."""

    def __init__(self, drafts: ImportDraftStore) -> None:
        self._drafts = drafts

    def execute(self, token: str) -> ImportPreview:
        info, columns, rows, detected = self._drafts.preview(token, PREVIEW_ROWS)
        return ImportPreview(
            token, info.suggested_name or "importado", info.n_rows or 0, info.n_cols or 0, columns, rows, detected
        )


class DiscardImportDraft:
    def __init__(self, drafts: ImportDraftStore) -> None:
        self._drafts = drafts

    def execute(self, token: str, preview_only: bool = False) -> None:
        if preview_only:
            self._drafts.discard_preview(token)
        else:
            self._drafts.discard(token)


class CommitImport:
    """Saves a pending import as a dataset, optionally overriding column types; the draft goes away on success."""

    def __init__(self, drafts: ImportDraftStore, datasets: DatasetRepository) -> None:
        self._drafts = drafts
        self._datasets = datasets

    def execute(self, token: str, name: str | None, column_kinds: dict[str, str] | None = None) -> DatasetInfo:
        staged = self._drafts.take(token)
        frame = staged.frame
        for column, kind in (column_kinds or {}).items():
            if column not in frame.columns:
                raise InvalidInputError(f"La columna «{column}» no existe.")
            frame[column], _ = convert_column(frame[column], kind)
        info = self._datasets.add(
            clean_dataset_name(name, staged.suggested_name), frame, staged.source_type, staged.source_meta
        )
        self._drafts.discard(token)  # only after the dataset exists: a name clash leaves the draft to retry
        return info


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
