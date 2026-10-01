"""Small helpers shared by the blueprints."""

from __future__ import annotations

from typing import Any, TypeVar

from flask import current_app, jsonify, request

from sigma.bootstrap import Container
from sigma.presentation.web.reports.blocks import ReportBuilder
from sigma.presentation.web.schemas.base import RequestModel
from sigma.presentation.web.serializers import to_jsonable

M = TypeVar("M", bound=RequestModel)


def container() -> Container:
    return current_app.extensions["sigma"]


def parse_body(model: type[M]) -> M:
    """Validate the JSON body; a ``ValidationError`` is turned into a 422 by the error handler."""
    return model.model_validate(request.get_json(silent=True) or {})


def dataset_name(dataset_id: str | None) -> str | None:
    return container().get_dataset_info.execute(dataset_id).name if dataset_id else None


def analysis_response(
    module: str,
    analysis_type: str,
    title: str,
    params: RequestModel,
    result: Any,
    builder: ReportBuilder,
    dataset_id: str | None = None,
    extra: dict[str, Any] | None = None,
):
    """Standard payload of every analysis endpoint (also what the *Guardar* button persists).

    ``extra`` adds top-level keys that are not part of the saved result (e.g. a dataset created by the call).
    """
    return jsonify(
        {
            "meta": {
                "module": module,
                "analysis_type": analysis_type,
                "title": title,
                "dataset_id": dataset_id,
                "params": to_jsonable(params.model_dump()),
            },
            "result": to_jsonable(result),
            "report": builder.report(),
            "figures": builder.figures,
            **(extra or {}),
        }
    )
