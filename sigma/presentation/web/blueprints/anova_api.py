"""ANOVA endpoint."""

from __future__ import annotations

from flask import Blueprint

from sigma.presentation.web.blueprints._helpers import analysis_response, container, dataset_name, parse_body
from sigma.presentation.web.reports import anova as report
from sigma.presentation.web.schemas.analysis import AnovaRequest

bp = Blueprint("anova_api", __name__, url_prefix="/api/anova")


@bp.post("/run")
def run():
    body = parse_body(AnovaRequest)
    result = container().run_anova.execute(body.to_command())
    builder = report.build_report(result, dataset_name(body.dataset_id) or "dataset")
    analysis_type = "anova_one_way" if result.design == "one_way" else "anova_two_way"
    return analysis_response("anova", analysis_type, builder.title, body, result, builder, body.dataset_id)
