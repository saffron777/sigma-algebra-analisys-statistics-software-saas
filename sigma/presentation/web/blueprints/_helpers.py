"""Small helpers shared by the blueprints."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from flask import Response, current_app, g, jsonify, request

from sigma.application.use_cases.exports import ExportedFile, ExportOptions
from sigma.bootstrap import Container, UserServices
from sigma.presentation.web.reports.blocks import ReportBuilder
from sigma.presentation.web.schemas.base import RequestModel
from sigma.presentation.web.serializers import to_jsonable

M = TypeVar("M", bound=RequestModel)
log = logging.getLogger(__name__)
HUGE_RESPONSE_BYTES = 5_000_000  # a report bigger than this is a chart that should have been down-sampled


def container() -> UserServices:
    """The use cases of the signed-in user (set by the authentication guard); they only see that user's data."""
    return g.services


def shared() -> Container:
    """The application-wide container (account administration, kernels)."""
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
    payload = {
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
    _warn_if_huge(module, payload)
    return jsonify(payload)


def _warn_if_huge(module: str, payload: dict[str, Any]) -> None:
    """A figure with one point per row is what once froze a browser tab: say so in the log while it is shipped."""
    try:
        figures = payload.get("figures") or []
        size = len(json.dumps(figures))
    except (TypeError, ValueError):  # measuring must never break the answer
        return
    if size > HUGE_RESPONSE_BYTES:
        biggest = max(range(len(figures)), key=lambda index: len(json.dumps(figures[index])), default=-1)
        log.warning(
            "analysis %s: respuesta de %.1f MB (figura %s es la mayor); ¿falta remuestrear sus puntos?",
            module,
            size / 1e6,
            biggest,
        )


def _flag(name: str) -> bool | None:
    value = request.args.get(name)
    return None if value is None else value.lower() in {"1", "true", "yes", "on"}


def export_options_from_query() -> ExportOptions:
    """The export dialog's choices when they travel in the query string (dataset and notebook reports)."""
    args = request.args
    return ExportOptions(
        template_id=args.get("template") or None,
        title=args.get("title") or None,
        subtitle=args.get("subtitle") or None,
        author=args.get("author", ""),
        cover=_flag("cover"),
        toc=_flag("toc"),
        page_size=args.get("page_size") or None,
        orientation=args.get("orientation") or None,
    )


def file_response(exported: ExportedFile) -> Response:
    """A download: UTF-8 file name (with an ASCII fallback), never sniffed, never cached."""
    import unicodedata
    from urllib.parse import quote

    ascii_name = unicodedata.normalize("NFKD", exported.filename).encode("ascii", "ignore").decode() or "informe"
    ascii_name = ascii_name.replace('"', "'")
    return Response(
        exported.data,
        mimetype=exported.content_type,
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(exported.filename)}"
            ),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )
