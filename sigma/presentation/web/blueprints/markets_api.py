"""Survey / market-research endpoints (sample size, sampling, estimation, tabulation)."""

from __future__ import annotations

from flask import Blueprint

from sigma.presentation.web.blueprints._helpers import analysis_response, container, dataset_name, parse_body
from sigma.presentation.web.reports import markets as report
from sigma.presentation.web.schemas.survey import (
    CrosstabRequest,
    DrawSampleRequest,
    EstimateRequest,
    LikertRequest,
    SampleSizeRequest,
)
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("markets_api", __name__, url_prefix="/api/markets")


@bp.post("/sample-size")
def sample_size():
    body = parse_body(SampleSizeRequest)
    result = container().calculate_sample_size.execute(body.to_command())
    builder = report.sample_size_report(result)
    return analysis_response(
        "markets", f"sample_size_{body.kind}", builder.title, body, result, builder, body.dataset_id
    )


@bp.post("/draw")
def draw():
    body = parse_body(DrawSampleRequest)
    result, saved = container().draw_survey_sample.execute(body.to_command())
    builder = report.draw_report(result, saved.name if saved else None)
    extra = {"saved_dataset": to_jsonable(saved)} if saved else None
    return analysis_response(
        "markets", f"draw_{body.design}", builder.title, body, result, builder, body.dataset_id, extra
    )


@bp.post("/estimate")
def estimate():
    body = parse_body(EstimateRequest)
    result = container().estimate_survey.execute(body.to_command())
    builder = report.estimate_report(result)
    builder.title = f"{builder.title} · {dataset_name(body.dataset_id)}"
    return analysis_response("markets", f"estimate_{body.kind}", builder.title, body, result, builder, body.dataset_id)


@bp.post("/crosstab")
def crosstab():
    body = parse_body(CrosstabRequest)
    result = container().run_crosstab.execute(body.to_command())
    builder = report.crosstab_report(result)
    return analysis_response("markets", "crosstab", builder.title, body, result, builder, body.dataset_id)


@bp.post("/likert")
def likert():
    body = parse_body(LikertRequest)
    result = container().run_likert.execute(body.to_command())
    builder = report.likert_report(result)
    return analysis_response("markets", "likert", builder.title, body, result, builder, body.dataset_id)
