"""Per-module workspace (form + last result, kept until «Limpiar») and the revision probe for live sync."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.workspace import UNSET
from sigma.presentation.web.blueprints._helpers import container, parse_body
from sigma.presentation.web.schemas.workspace import SaveWorkspaceRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("workspace_api", __name__, url_prefix="/api")


@bp.get("/workspace/<module>")
def get_workspace(module: str):
    return jsonify({"workspace": to_jsonable(container().get_workspace.execute(module))})


@bp.put("/workspace/<module>")
def save_workspace(module: str):
    body = parse_body(SaveWorkspaceRequest)
    sent = body.model_fields_set
    entry = container().save_workspace.execute(
        module,
        state=body.state if "state" in sent else UNSET,
        payload=body.payload if "payload" in sent else UNSET,
    )
    return jsonify({"revision": entry.revision, "updated_at": entry.updated_at})


@bp.delete("/workspace/<module>")
def clear_workspace(module: str):
    return jsonify({"revision": container().clear_workspace.execute(module)})


@bp.delete("/workspace")
def clear_all_workspaces():
    return jsonify({"revisions": container().clear_workspace.execute_all()})


@bp.get("/sync")
def sync_state():
    """What changed? Cheap enough to poll every few seconds from every open window."""
    return jsonify(container().get_sync_state.execute())
