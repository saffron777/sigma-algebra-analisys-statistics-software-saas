"""The analysis catalogue: what a module offers, running one analysis, and the Python that reproduces it."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.presentation.web.blueprints._helpers import analysis_response, container, parse_body
from sigma.presentation.web.reports import analysis as report
from sigma.presentation.web.schemas.analysis import AnalysisCodeRequest, AnalysisRunRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("analysis_api", __name__, url_prefix="/api/analysis")


def _role_json(role: RoleSpec) -> dict:
    return {
        "name": role.name,
        "label": role.label,
        "kind": role.kind,
        "multiple": role.multiple,
        "optional": role.optional,
        "help": role.help,
    }


def _field_json(field: FieldSpec) -> dict:
    return {
        "name": field.name,
        "label": field.label,
        "kind": field.kind,
        "default": field.default,
        "options": [{"value": value, "label": label} for value, label in field.options],
        "help": field.help,
        "optional": field.optional,
        "minimum": field.minimum,
        "maximum": field.maximum,
        "visible_if": {"param": field.visible_if[0], "values": list(field.visible_if[1])} if field.visible_if else None,
    }


def _spec_json(spec: AnalysisSpec) -> dict:
    return {
        "id": spec.id,
        "label": spec.label,
        "group": spec.group,
        "description": spec.description,
        "needs_dataset": spec.needs_dataset,
        "can_replace": spec.can_replace,
        "has_code": spec.code is not None,
        "heavy": spec.heavy,
        "roles": [_role_json(r) for r in spec.roles],
        "params": [_field_json(f) for f in spec.params],
        "outputs": [{"key": o.key, "label": o.label, "required": o.required} for o in spec.outputs],
    }


@bp.get("/<module>/catalog")
def catalog(module: str):
    info, specs = container().list_analyses.execute(module)
    return jsonify(
        {
            "module": {"key": info.key, "label": info.label, "description": info.description},
            "analyses": [_spec_json(spec) for spec in specs],
        }
    )


@bp.post("/<module>/run")
def run(module: str):
    body = parse_body(AnalysisRunRequest)
    outcome = container().run_analysis.execute(body.to_command(module))
    subtitle = " · ".join(part for part in (outcome.dataset_name, outcome.output.subtitle) if part) or None
    builder = report.build_report(outcome.output, subtitle)
    title = f"{outcome.output.title} — {outcome.dataset_name}" if outcome.dataset_name else outcome.output.title
    return analysis_response(
        module,
        outcome.spec.id,
        title,
        body,
        outcome.output.result,
        builder,
        body.dataset_id if outcome.spec.needs_dataset else None,
        extra={"datasets": to_jsonable(outcome.created)},
    )


@bp.post("/<module>/code")
def code(module: str):
    body = parse_body(AnalysisCodeRequest)
    text = container().get_analysis_code.execute(module, body.analysis_id, body.dataset_id, body.roles, body.params)
    return jsonify({"code": text})
