"""Account administration (administrators only)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.presentation.web.auth import require_admin
from sigma.presentation.web.blueprints._helpers import container, parse_body, shared
from sigma.presentation.web.schemas.auth import CreateUserRequest, ResetPasswordRequest, UpdateUserRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("users_api", __name__, url_prefix="/api/users")
bp.before_request(require_admin)


@bp.get("")
def list_users():
    return jsonify({"users": to_jsonable(shared().list_users.execute())})


@bp.post("")
def create_user():
    body = parse_body(CreateUserRequest)
    user = shared().create_user.execute(
        body.username, body.display_name or "", body.password, body.role, body.can_use_console, body.can_use_ai
    )
    return jsonify({"user": to_jsonable(user)}), 201


@bp.patch("/<int:user_id>")
def update_user(user_id: int):
    body = parse_body(UpdateUserRequest)
    user = shared().update_user.execute(
        user_id,
        display_name=body.display_name,
        role=body.role,
        is_active=body.is_active,
        can_use_console=body.can_use_console,
        can_use_ai=body.can_use_ai,
    )
    return jsonify({"user": to_jsonable(user)})


@bp.post("/<int:user_id>/reset-password")
def reset_password(user_id: int):
    body = parse_body(ResetPasswordRequest)
    return jsonify({"user": to_jsonable(shared().reset_password.execute(user_id, body.new_password))})


@bp.delete("/<int:user_id>")
def delete_user(user_id: int):
    shared().delete_user.execute(container().user, user_id)
    return jsonify({"deleted": user_id})
