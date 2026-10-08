"""Dataset import and management endpoints."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, jsonify, request

from sigma.application.ports import ImportRequest
from sigma.domain.errors import InvalidInputError
from sigma.presentation.web.blueprints._helpers import container, export_options_from_query, file_response, parse_body
from sigma.presentation.web.schemas.datasets import ColumnTypesRequest, CommitImportRequest, ManualDatasetRequest, RenameRequest, UpdateDatasetDataRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("datasets_api", __name__, url_prefix="/api/datasets")


def _draft_id() -> str | None:
    """The pending import a request refers to (its stored file is used instead of a new upload)."""
    if request.mimetype == "multipart/form-data":
        return request.form.get("draft_id") or None
    return (request.get_json(silent=True) or {}).get("draft_id") or None


def _import_request() -> ImportRequest:
    """Accepts multipart (file sources) or JSON (SQL Server / API sources, or a file already kept as a draft)."""
    if request.files or request.mimetype == "multipart/form-data":
        source_type = request.form.get("source_type", "")
        raw_options = request.form.get("options") or "{}"
        upload = request.files.get("file")
        try:
            options: dict[str, Any] = json.loads(raw_options)
        except json.JSONDecodeError:
            raise InvalidInputError("Las opciones de importación no son un JSON válido.") from None
        return ImportRequest(
            source_type, options, upload.filename if upload else None, upload.read() if upload else None
        )
    body = request.get_json(silent=True) or {}
    return ImportRequest(str(body.get("source_type", "")), dict(body.get("options") or {}))


@bp.get("")
def list_datasets():
    return jsonify({"datasets": to_jsonable(container().list_datasets.execute())})


@bp.post("/samples")
def load_samples():
    """Imports the bundled demo CSV files into the signed-in user's datasets."""
    outcome = container().load_sample_data.execute()
    return jsonify({**outcome, "datasets": to_jsonable(container().list_datasets.execute())}), 201


@bp.post("/manual")
def create_manual_dataset():
    """Creates a dataset from data typed by hand (SPSS/Minitab-style data editor)."""
    body = parse_body(ManualDatasetRequest)
    info = container().create_manual_dataset.execute(
        body.name, [column.model_dump() for column in body.columns], body.rows
    )
    return jsonify({"dataset": to_jsonable(info)}), 201


@bp.put("/<dataset_id>/data")
def update_dataset_data(dataset_id: str):
    """Replaces a dataset's data with manually corrected rows (schema unchanged)."""
    body = parse_body(UpdateDatasetDataRequest)
    info = container().update_dataset_data.execute(
        dataset_id, [column.model_dump() for column in body.columns], body.rows
    )
    return jsonify({"dataset": to_jsonable(info)})


@bp.post("/import/inspect")
def inspect_source():
    return jsonify(to_jsonable(container().inspect_source.execute(_import_request(), _draft_id())))


@bp.get("/import/databases")
def external_databases():
    """The database engines that can be read from (and whether this server has their driver)."""
    return jsonify({"databases": container().list_external_databases.execute()})


@bp.post("/import/draft")
def upload_draft():
    """Keep an uploaded file on the server (one pending import per source type); no parsing yet."""
    upload = request.files.get("file")
    if upload is None:
        raise InvalidInputError("Seleccione un archivo.")
    info = container().upload_import_file.execute(
        request.form.get("source_type", ""), upload.filename or "archivo", upload.read()
    )
    return jsonify({"draft": to_jsonable(info)}), 201


@bp.get("/import/drafts")
def list_drafts():
    return jsonify({"drafts": to_jsonable(container().list_import_drafts.execute())})


@bp.get("/import/drafts/<token>")
def get_draft(token: str):
    """The preview of a pending import, as it was shown before the page changed."""
    return jsonify(_preview_payload(container().get_import_draft.execute(token)))


@bp.delete("/import/drafts/<token>")
def discard_draft(token: str):
    """Remove the pending import (file included); ``?preview_only=1`` keeps the file and drops only the parsed table."""
    container().discard_import_draft.execute(token, request.args.get("preview_only") == "1")
    return jsonify({"discarded": token})


@bp.post("/import/preview")
def preview_import():
    return jsonify(_preview_payload(container().preview_import.execute(_import_request(), _draft_id())))


def _preview_payload(preview) -> dict:
    return {
        "token": preview.token,
        "suggested_name": preview.suggested_name,
        "n_rows": preview.n_rows,
        "n_cols": preview.n_cols,
        "columns": to_jsonable(preview.columns),
        "rows": to_jsonable(preview.rows),
        "detected": to_jsonable(preview.detected),
    }


@bp.post("/import/commit")
def commit_import():
    body = parse_body(CommitImportRequest)
    info = container().commit_import.execute(body.token, body.name, body.column_kinds)
    return jsonify({"dataset": to_jsonable(info)}), 201


@bp.get("/<dataset_id>")
def get_dataset(dataset_id: str):
    return jsonify({"dataset": to_jsonable(container().get_dataset_info.execute(dataset_id))})


@bp.get("/<dataset_id>/preview")
def preview_dataset(dataset_id: str):
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)
    info, frame = container().preview_dataset.execute(dataset_id, limit, offset)
    return jsonify(
        {
            "dataset": to_jsonable(info),
            "columns": [str(c) for c in frame.columns],
            "rows": [to_jsonable(row) for row in frame.to_numpy(dtype=object).tolist()],
            "total": info.n_rows,
            "offset": offset,
        }
    )


@bp.patch("/<dataset_id>")
def rename_dataset(dataset_id: str):
    body = parse_body(RenameRequest)
    return jsonify({"dataset": to_jsonable(container().rename_dataset.execute(dataset_id, body.name))})


@bp.delete("/<dataset_id>")
def delete_dataset(dataset_id: str):
    container().delete_dataset.execute(dataset_id)
    return jsonify({"deleted": dataset_id})


@bp.post("/<dataset_id>/column-types")
def change_column_types(dataset_id: str):
    body = parse_body(ColumnTypesRequest)
    info, failures = container().change_column_types.execute(dataset_id, body.column_kinds)
    return jsonify({"dataset": to_jsonable(info), "failed_conversions": failures})


@bp.get("/<dataset_id>/report/<fmt>")
def dataset_report(dataset_id: str, fmt: str):
    """The dataset's ficha as PDF, Word or Excel (Excel also carries every row)."""
    return file_response(container().export_dataset_report.execute(dataset_id, fmt, export_options_from_query()))
