"""Authentication endpoints (public: status, setup, login, logout) and the signed-in user's own account."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, session

from sigma.presentation.web.auth import login_user, logout_user, session_user
from sigma.presentation.web.blueprints._helpers import container, parse_body, shared
from sigma.presentation.web.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    ProfileRequest,
    SetupRequest,
)
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("auth_api", __name__, url_prefix="/api/auth")


def _client_ip() -> str:
    return request.remote_addr or "unknown"


@bp.get("/status")
def status():
    """Lets the login/setup pages know what to show (no authentication needed)."""
    services = shared()
    user = session_user(services)
    return jsonify({"setup_required": services.get_setup_state.setup_required(), "user": to_jsonable(user)})


@bp.post("/setup")
def setup():
    body = parse_body(SetupRequest)
    user, adopted = shared().create_first_admin.execute(body.username, body.display_name or "", body.password)
    login_user(user)
    current_app.logger.info("Administrador inicial creado; datos adoptados: %s", adopted)
    return jsonify({"user": to_jsonable(user), "adopted": adopted}), 201


@bp.post("/login")
def login():
    body = parse_body(LoginRequest)
    user = shared().authenticate.execute(body.username, body.password, _client_ip())
    login_user(user)
    return jsonify({"user": to_jsonable(user)})


@bp.post("/logout")
def logout():
    logout_user()
    return jsonify({"signed_out": True})


@bp.get("/me")
def me():
    return jsonify({"user": to_jsonable(container().user)})


@bp.post("/change-password")
def change_password():
    body = parse_body(ChangePasswordRequest)
    updated = container().change_own_password.execute(container().user, body.current_password, body.new_password)
    session["epoch"] = updated.auth_epoch  # every other session of this account is now invalid; this one stays
    return jsonify({"user": to_jsonable(updated)})


@bp.patch("/profile")
def profile():
    body = parse_body(ProfileRequest)
    return jsonify({"user": to_jsonable(container().update_profile.execute(container().user, body.display_name))})
