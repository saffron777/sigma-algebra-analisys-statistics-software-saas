"""Dataset import and management endpoints."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, jsonify, request

from sigma.application.ports import ImportRequest
from sigma.domain.errors import InvalidInputError
from sigma.presentation.web.blueprints._helpers import container, parse_body
from sigma.presentation.web.schemas.datasets import ColumnTypesRequest, CommitImportRequest, RenameRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("datasets_api", __name__, url_prefix="/api/datasets")


def _import_request() -> ImportRequest:
    """Accepts multipart (file sources) or JSON (SQL Server / API sources)."""
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


@bp.post("/import/inspect")
def inspect_source():
    return jsonify(to_jsonable(container().inspect_source.execute(_import_request())))


@bp.post("/import/preview")
def preview_import():
    preview = container().preview_import.execute(_import_request())
    return jsonify(
        {
            "token": preview.token,
            "suggested_name": preview.suggested_name,
            "n_rows": preview.n_rows,
            "n_cols": preview.n_cols,
            "columns": to_jsonable(preview.columns),
            "rows": to_jsonable(preview.rows),
            "detected": to_jsonable(preview.detected),
        }
    )


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
