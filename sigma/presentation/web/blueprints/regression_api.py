"""Linear regression endpoint."""

from __future__ import annotations

from flask import Blueprint

from sigma.presentation.web.blueprints._helpers import analysis_response, container, dataset_name, parse_body
from sigma.presentation.web.reports import regression as report
from sigma.presentation.web.schemas.analysis import RegressionRequest

bp = Blueprint("regression_api", __name__, url_prefix="/api/regression")


@bp.post("/run")
def run():
    body = parse_body(RegressionRequest)
    result = container().run_regression.execute(body.to_command())
    builder = report.build_report(result, dataset_name(body.dataset_id) or "dataset")
    analysis_type = "ols_simple" if result.is_simple else "ols_multiple"
    return analysis_response("regression", analysis_type, builder.title, body, result, builder, body.dataset_id)
