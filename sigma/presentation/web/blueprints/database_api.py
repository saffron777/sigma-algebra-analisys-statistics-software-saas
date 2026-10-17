"""Choosing the database Sigma Algebra persists to: the administration page and the first-run wizard's step."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.errors import PermissionDeniedError
from sigma.presentation.web.auth import require_admin
from sigma.presentation.web.blueprints._helpers import parse_body, shared
from sigma.presentation.web.schemas.database import DatabaseRequest

admin_bp = Blueprint("database_admin_api", __name__, url_prefix="/api/admin/database")
admin_bp.before_request(require_admin)

setup_bp = Blueprint("database_setup_api", __name__, url_prefix="/api/setup/database")


@setup_bp.before_request
def only_before_setup() -> None:
    """The wizard's step is public (nobody can sign in yet) but only until the first administrator exists."""
    if not shared().get_setup_state.setup_required():
        raise PermissionDeniedError("La configuración inicial ya se completó.")


def _status():
    return jsonify(shared().get_database_status.execute())


def _test():
    return jsonify(shared().test_database.execute(parse_body(DatabaseRequest).payload()))


def _apply():
    return jsonify(shared().apply_database.execute(parse_body(DatabaseRequest).payload()))


def _apply_in_setup():
    """Nothing exists yet to be copied, so the wizard never starts a migration."""
    return jsonify(shared().apply_database.execute({**parse_body(DatabaseRequest).payload(), "migrate": False}))


@admin_bp.get("/migration")
def migration():
    return jsonify(shared().get_migration_status.execute())


for _bp, _apply_view in ((admin_bp, _apply), (setup_bp, _apply_in_setup)):
    _bp.add_url_rule("", "status", _status, methods=["GET"])
    _bp.add_url_rule("/test", "test", _test, methods=["POST"])
    _bp.add_url_rule("/apply", "apply", _apply_view, methods=["POST"])
