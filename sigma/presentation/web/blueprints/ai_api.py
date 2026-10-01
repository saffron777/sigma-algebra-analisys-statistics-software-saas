"""AI interpretation: availability for the signed-in user, the interpretation itself, and the admin settings."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.presentation.web.auth import require_admin
from sigma.presentation.web.blueprints._helpers import container, parse_body, shared
from sigma.presentation.web.schemas.ai import AiSettingsRequest, InterpretRequest
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
    result = container().interpret_result.execute(body.module, body.report, body.title, body.focus, body.dataset_id)
    return jsonify({"interpretation": to_jsonable(result)})


@admin_bp.get("")
def admin_status():
    return jsonify(shared().get_ai_admin.execute())


@admin_bp.put("")
def admin_update():
    body = parse_body(AiSettingsRequest)
    shared().update_ai_settings.execute(body.key, body.clear_key, body.model, body.enabled)
    return jsonify(shared().get_ai_admin.execute())


@admin_bp.post("/test")
def admin_test():
    return jsonify(shared().test_ai_connection.execute())
