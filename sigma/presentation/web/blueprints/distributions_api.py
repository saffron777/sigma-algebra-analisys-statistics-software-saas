"""Distribution catalogue and calculator."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.domain.distributions import list_distributions
from sigma.presentation.web.blueprints._helpers import analysis_response, container, parse_body
from sigma.presentation.web.reports import distributions as report
from sigma.presentation.web.schemas.analysis import DistributionRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("distributions_api", __name__, url_prefix="/api/distributions")

_MODE_LABELS = {
    "point": "evaluación en un punto",
    "between": "probabilidad entre dos valores",
    "quantile": "cuantil",
    "pvalue": "valor p",
}


@bp.get("")
def catalogue():
    return jsonify(
        {
            "distributions": [
                {
                    "key": s.key,
                    "label": s.label,
                    "kind": s.kind,
                    "description": s.description,
                    "params": [to_jsonable(vars(p)) for p in s.params],
                }
                for s in list_distributions()
            ]
        }
    )


@bp.post("/calculate")
def calculate():
    body = parse_body(DistributionRequest)
    result = container().calculate_distribution.execute(body.to_command())
    builder = report.build_report(result)
    title = f"{builder.title} — {_MODE_LABELS[result.mode]}"
    builder.title = title
    return analysis_response("distributions", f"{body.distribution}_{body.mode}", title, body, result, builder)
