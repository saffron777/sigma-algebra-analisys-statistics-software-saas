"""Saved results: create, list, read, edit, delete and export."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, request

from sigma.domain.analysis.modules import CATALOG_MODULES
from sigma.domain.saved_results import MODULE_LABELS
from sigma.domain.workspace import UNSET
from sigma.presentation.web.blueprints._helpers import container, parse_body
from sigma.presentation.web.schemas.datasets import SaveResultRequest, UpdateResultRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("results_api", __name__, url_prefix="/api/results")


@bp.post("")
def save():
    saved = container().save_result.execute(parse_body(SaveResultRequest).to_command())
    return jsonify({"result": to_jsonable(_summary_of(saved))}), 201


@bp.get("")
def list_results():
    items, total = container().list_results.execute(
        request.args.get("module") or None,
        request.args.get("search") or None,
        request.args.get("limit", default=50, type=int),
        request.args.get("offset", default=0, type=int),
    )
    pages = {m.key: m.path for m in CATALOG_MODULES}
    return jsonify({"results": to_jsonable(items), "total": total, "modules": MODULE_LABELS, "pages": pages})


@bp.get("/<result_id>")
def get_result(result_id: str):
    return jsonify({"result": to_jsonable(container().get_result.execute(result_id))})


@bp.patch("/<result_id>")
def update_result(result_id: str):
    body = parse_body(UpdateResultRequest)
    interpretation = body.interpretation if "interpretation" in body.model_fields_set else UNSET
    updated = container().update_result.execute(result_id, body.title, body.notes, interpretation)
    return jsonify({"result": to_jsonable(_summary_of(updated))})


@bp.delete("/<result_id>")
def delete_result(result_id: str):
    container().delete_result.execute(result_id)
    return jsonify({"deleted": result_id})


@bp.get("/<result_id>/export")
def export_result(result_id: str):
    data = to_jsonable(container().export_result.execute(result_id))
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="sigma-resultado-{result_id[:8]}.json"'},
    )


def _summary_of(saved):
    """Listing-sized view of a saved result (without the heavy report/figures)."""
    keys = ("id", "module", "analysis_type", "title", "notes", "dataset_id", "dataset_name", "created_at", "updated_at")
    return {key: getattr(saved, key) for key in keys}
