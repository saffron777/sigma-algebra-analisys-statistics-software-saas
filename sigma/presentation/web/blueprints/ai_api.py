"""AI interpretation: availability for the signed-in user, the interpretation itself, and the model catalogue."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.presentation.web.auth import require_admin
from sigma.presentation.web.blueprints._helpers import container, parse_body, shared
from sigma.presentation.web.schemas.ai import (
    AiModelCreateRequest,
    AiModelUpdateRequest,
    AiSettingsRequest,
    AiTestRequest,
    InterpretRequest,
)
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("ai_api", __name__, url_prefix="/api/ai")
admin_bp = Blueprint("ai_admin_api", __name__, url_prefix="/api/admin/ai")
admin_bp.before_request(require_admin)


@bp.get("/status")
def status():
    return jsonify(container().get_ai_status.execute())


@bp.post("/interpret")
def interpret():
    body = parse_body(InterpretRequest)
    result = container().interpret_result.execute(
        body.module, body.report, body.title, body.focus, body.dataset_id, model_id=body.model_id
    )
    return jsonify({"interpretation": to_jsonable(result)})


@admin_bp.get("")
def admin_status():
    return jsonify(shared().get_ai_admin.execute())


@admin_bp.put("")
def admin_update():
    """The global switch, and the shortcut to the Claude entry (kept for older pages and scripts)."""
    body = parse_body(AiSettingsRequest)
    shared().update_ai_settings.execute(body.key, body.clear_key, body.model, body.enabled)
    return jsonify(shared().get_ai_admin.execute())


@admin_bp.get("/providers")
def admin_providers():
    return jsonify({"providers": shared().ai_servers.providers()})


@admin_bp.post("/servers")
def add_server():
    body = parse_body(AiModelCreateRequest)
    return jsonify(shared().ai_servers.add_model(body.model_dump())), 201


@admin_bp.patch("/servers/<server_id>")
def update_server(server_id: str):
    body = parse_body(AiModelUpdateRequest)
    changes = {key: value for key, value in body.model_dump().items() if value is not None}
    return jsonify(shared().ai_servers.update_model(server_id, changes))


@admin_bp.delete("/servers/<server_id>")
def delete_server(server_id: str):
    """Removing a model answers with the whole administration view, like the other changes."""
    return jsonify(shared().ai_servers.remove_model(server_id))


@admin_bp.post("/servers/<server_id>/test")
def test_server(server_id: str):
    return jsonify(shared().ai_servers.test_model(server_id))


@admin_bp.post("/test")
def admin_test():
    body = parse_body(AiTestRequest)
    return jsonify(shared().test_ai_connection.execute(body.model_id))
