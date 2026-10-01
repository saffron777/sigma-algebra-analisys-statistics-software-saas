"""Exporting reports as PDF, Word or Excel."""

from __future__ import annotations

import json
import re

from flask import Blueprint, request

from sigma.application.use_cases.exports import ExportOptions, SectionInput
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.imaging import MAX_CHART_BYTES
from sigma.presentation.web.blueprints._helpers import container, file_response
from sigma.presentation.web.schemas.export import ExportPayload

bp = Blueprint("export_api", __name__, url_prefix="/api/export")

_IMAGE_PART = re.compile(r"^img-(\d{1,3})-(\d{1,3})$")


def _payload() -> tuple[ExportPayload, dict[tuple[int, int], bytes]]:
    """JSON body, or multipart: a ``payload`` JSON part plus one ``img-<section>-<figure>`` PNG part per chart."""
    images: dict[tuple[int, int], bytes] = {}
    if request.mimetype == "multipart/form-data":
        try:
            raw = json.loads(request.form.get("payload", ""))
        except json.JSONDecodeError:
            raise InvalidInputError("El contenido del informe no es un JSON válido.") from None
        for name, upload in request.files.items():
            match = _IMAGE_PART.match(name)
            if match:
                data = upload.stream.read(MAX_CHART_BYTES + 1)
                images[(int(match.group(1)), int(match.group(2)))] = data
    else:
        raw = request.get_json(silent=True) or {}
    return ExportPayload.model_validate(raw), images


@bp.post("/<fmt>")
def export_report(fmt: str):
    payload, images = _payload()
    sections = [
        SectionInput(
            title=s.title,
            subtitle=s.subtitle,
            meta=s.meta,
            blocks=s.blocks,
            images={figure: data for (index, figure), data in images.items() if index == position},
        )
        for position, s in enumerate(payload.sections)
    ]
    options = ExportOptions(
        template_id=payload.template_id,
        title=payload.title,
        subtitle=payload.subtitle,
        author=payload.author,
        cover=payload.cover,
        toc=payload.toc,
        page_size=payload.page_size,
        orientation=payload.orientation,
    )
    return file_response(container().export_report.execute(fmt, sections, options))
