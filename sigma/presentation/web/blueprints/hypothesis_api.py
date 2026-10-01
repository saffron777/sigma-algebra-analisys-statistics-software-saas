"""Hypothesis-test catalogue and runner."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.statistics.hypothesis import FieldSpec, TestSpec, list_tests
from sigma.presentation.web.blueprints._helpers import analysis_response, container, dataset_name, parse_body
from sigma.presentation.web.reports import hypothesis as report
from sigma.presentation.web.schemas.analysis import HypothesisRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("hypothesis_api", __name__, url_prefix="/api/hypothesis")


def _field_json(field: FieldSpec) -> dict:
    return {
        "name": field.name,
        "label": field.label,
        "kind": field.kind,
        "default": field.default,
        "options": [{"value": v, "label": label} for v, label in field.options],
        "help": field.help,
        "optional": field.optional,
    }


def _spec_json(spec: TestSpec) -> dict:
    return {
        "id": spec.id,
        "label": spec.label,
        "group": spec.group,
        "description": spec.description,
        "roles": [to_jsonable(vars(r)) for r in spec.roles],
        "params": [_field_json(f) for f in spec.params],
        "summary_fields": [_field_json(f) for f in spec.summary_fields],
        "tails": list(spec.tails),
    }


@bp.get("/catalog")
def catalog():
    return jsonify({"tests": [_spec_json(spec) for spec in list_tests()]})


@bp.post("/run")
def run():
    body = parse_body(HypothesisRequest)
    result = container().run_hypothesis_test.execute(body.to_command())
    used_dataset = None if body.summary else body.dataset_id
    builder = report.build_report(result)
    name = dataset_name(used_dataset)
    title = f"{result.test_name} — {name}" if name else f"{result.test_name} (estadísticos resumidos)"
    return analysis_response("hypothesis", body.test_id, title, body, result, builder, used_dataset)
