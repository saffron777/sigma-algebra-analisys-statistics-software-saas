"""Request-level protections. Sigma can execute code, so it must only answer its own local UI.

* ``Host`` allow-list  → blocks DNS-rebinding attacks.
* ``Origin`` check     → blocks cross-site requests from other web pages.
* CSRF token           → every mutating request must echo the session token in ``X-CSRF-Token``.
* Security headers + a restrictive Content-Security-Policy.
"""

from __future__ import annotations

import hmac
import secrets
from datetime import timedelta
from urllib.parse import urlsplit

from flask import Flask, abort, request, session

from sigma.config import Config

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_HEADER = "X-CSRF-Token"
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-eval'",  # Alpine.js evaluates expressions with the Function constructor
        "script-src-attr 'unsafe-inline'",  # Jspreadsheet/jSuites attach inline handlers (onclick, onchange) to grid cells
        "style-src 'self' 'unsafe-inline'",  # Plotly and Alpine set inline styles
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def _hostname(host_header: str) -> str:
    """'localhost:5000' → 'localhost'; '[::1]:5000' → '[::1]'."""
    if host_header.startswith("["):
        return host_header.split("]", 1)[0] + "]"
    return host_header.rsplit(":", 1)[0] if ":" in host_header else host_header


def csrf_token() -> str:
    token = session.get("csrf")
    if not token:
        token = session["csrf"] = secrets.token_urlsafe(32)
    return token


def init_security(app: Flask, config: Config) -> None:
    allowed = {h.lower() for h in config.allowed_hosts}
    app.config.update(
        SECRET_KEY=config.secret_key,
        MAX_CONTENT_LENGTH=config.max_upload_bytes,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=config.secure_cookies,
        SESSION_COOKIE_NAME="sigma_session",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=config.session_hours),  # sliding: renewed on every request
    )
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def guard_request() -> None:
        if _hostname(request.host).lower() not in allowed:
            abort(400, description="Host no permitido.")
        origin = request.headers.get("Origin")
        if origin and request.method in MUTATING_METHODS and urlsplit(origin).netloc.lower() != request.host.lower():
            abort(403, description="Origen no permitido.")
        if request.method in MUTATING_METHODS:
            expected = session.get("csrf", "")
            supplied = request.headers.get(CSRF_HEADER, "")
            if not expected or not hmac.compare_digest(expected, supplied):
                abort(403, description="Token CSRF inválido o ausente. Recargue la página.")

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"  # nothing private may be served from a browser cache
        return response
