"""Report templates (corporate look of the exports): CRUD and logo."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from sigma.domain.errors import InvalidInputError
from sigma.presentation.web.blueprints._helpers import container, parse_body
from sigma.presentation.web.schemas.export import DefaultTemplateRequest, TemplateRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("templates_api", __name__, url_prefix="/api/templates")


@bp.get("")
def list_templates():
    return jsonify({"templates": to_jsonable(container().list_templates.execute())})


@bp.post("")
def create_template():
    body = parse_body(TemplateRequest)
    template = container().create_template.execute(make_default=body.make_default, **body.fields())
    return jsonify({"template": to_jsonable(template)}), 201


@bp.get("/<template_id>")
def get_template(template_id: str):
    return jsonify({"template": to_jsonable(container().get_template.execute(template_id))})


@bp.put("/<template_id>")
@bp.patch("/<template_id>")
def update_template(template_id: str):
    body = parse_body(TemplateRequest)
    template = container().update_template.execute(template_id, **body.fields())
    if body.make_default:
        template = container().set_default_template.execute(template_id, True)
    return jsonify({"template": to_jsonable(template)})


@bp.delete("/<template_id>")
def delete_template(template_id: str):
    container().delete_template.execute(template_id)
    return jsonify({"deleted": template_id})


@bp.post("/<template_id>/default")
def set_default(template_id: str):
    body = parse_body(DefaultTemplateRequest)
    return jsonify({"template": to_jsonable(container().set_default_template.execute(template_id, body.default))})


@bp.get("/<template_id>/logo")
def get_logo(template_id: str):
    logo = container().get_template_logo.execute(template_id)
    return Response(logo, mimetype="image/png", headers={"X-Content-Type-Options": "nosniff"})


@bp.post("/<template_id>/logo")
def upload_logo(template_id: str):
    upload = request.files.get("file")
    if upload is None:
        raise InvalidInputError("Seleccione una imagen PNG o JPG.")
    template = container().upload_template_logo.execute(template_id, upload.stream.read(3 * 1024 * 1024))
    return jsonify({"template": to_jsonable(template)}), 201


@bp.delete("/<template_id>/logo")
def remove_logo(template_id: str):
    return jsonify({"template": to_jsonable(container().remove_template_logo.execute(template_id))})
