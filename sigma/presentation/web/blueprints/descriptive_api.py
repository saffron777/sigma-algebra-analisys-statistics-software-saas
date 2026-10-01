"""Descriptive statistics endpoint."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.distributions import get_distribution
from sigma.domain.statistics.descriptive import FITTABLE
from sigma.presentation.web.blueprints._helpers import analysis_response, container, dataset_name, parse_body
from sigma.presentation.web.reports import descriptive as report
from sigma.presentation.web.schemas.analysis import DescriptiveRequest

bp = Blueprint("descriptive_api", __name__, url_prefix="/api/descriptive")


@bp.get("/options")
def options():
    return jsonify({"fittable": [{"key": key, "label": get_distribution(key).label} for key in FITTABLE]})


@bp.post("/run")
def run():
    body = parse_body(DescriptiveRequest)
    result = container().run_descriptive.execute(body.to_command())
    name = dataset_name(body.dataset_id) or "dataset"
    builder = report.build_report(result, name)
    return analysis_response("descriptive", "descriptive", builder.title, body, result, builder, body.dataset_id)
