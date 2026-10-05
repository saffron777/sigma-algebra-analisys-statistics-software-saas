"""Session handling and the global authentication guard.

Every request except the login/setup flow and static files needs a valid session. The session stores only
the user id and the account's ``auth_epoch``: changing a password or disabling an account bumps the epoch,
which invalidates every session of that account on its next request.
"""

from __future__ import annotations

import secrets
from urllib.parse import quote

from flask import Flask, g, jsonify, redirect, request, session

from sigma.bootstrap import Container, UserServices
from sigma.domain.errors import PermissionDeniedError
from sigma.domain.users import User

PUBLIC_PATHS = frozenset(
    {
        "/login",
        "/setup",
        "/health",
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/status",
        "/api/auth/logout",
        # the wizard's «Base de datos» step: open only while no account exists (the endpoints check it)
        "/api/setup/database",
        "/api/setup/database/test",
        "/api/setup/database/apply",
    }
)
_AUTH_KEYS = ("uid", "epoch")


def login_user(user: User) -> None:
    """Start a fresh session for ``user`` (clearing the old one prevents session fixation)."""
    session.clear()
    session["uid"] = user.id
    session["epoch"] = user.auth_epoch
    session["csrf"] = secrets.token_urlsafe(32)
    session.permanent = True


def logout_user() -> None:
    session.clear()


def current_user() -> User:
    return g.user


def services() -> UserServices:
    return g.services


def require_admin() -> None:
    if not g.user.is_admin:
        raise PermissionDeniedError("Esta acción es solo para administradores.")


def require_console() -> None:
    if not g.user.can_use_console:
        raise PermissionDeniedError("Su cuenta no tiene permiso para usar la consola.")


def safe_next(target: str | None) -> str:
    """Only same-site paths are valid redirect targets (blocks open redirects)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/"


def _wants_json() -> bool:
    return request.path.startswith("/api/")


def session_user(container: Container) -> User | None:
    """The account behind the current session cookie, or ``None`` if there is no valid session."""
    uid = session.get("uid")
    if not isinstance(uid, int):
        return None
    return container.get_session_user.execute(uid, session.get("epoch", 0))


def _reject(code: str, message: str, page_target: str):
    if _wants_json():
        return jsonify({"error": {"code": code, "message": message, "details": None}}), 401
    return redirect(page_target)


def init_auth(app: Flask) -> None:
    @app.before_request
    def authenticate_request():
        container: Container = app.extensions["sigma"]  # read per request: the administrator can switch databases
        path = request.path
        g.user, g.services = None, None
        if path.startswith("/static/") or path == "/favicon.ico" or path in PUBLIC_PATHS:
            return None
        if container.get_setup_state.setup_required():
            return _reject("setup_required", "Cree primero la cuenta de administrador.", "/setup")
        user = session_user(container)
        if user is None:
            for key in _AUTH_KEYS:
                session.pop(key, None)
            target = request.full_path.rstrip("?") if request.method == "GET" else "/"
            return _reject(
                "unauthenticated", "Inicie sesión para continuar.", f"/login?next={quote(safe_next(target))}"
            )
        g.user, g.services = user, container.for_user(user)
        return None

    @app.context_processor
    def inject_user():
        return {"current_user": g.get("user")}
