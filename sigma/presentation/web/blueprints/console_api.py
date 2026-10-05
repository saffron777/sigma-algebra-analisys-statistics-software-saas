"""Python console: notebooks (.ipynb) and kernel execution."""

from __future__ import annotations

import re

from flask import Blueprint, Response, jsonify, request

from sigma.application.use_cases.exports import NotebookReportOptions
from sigma.domain.errors import InvalidInputError
from sigma.presentation.web.auth import require_console
from sigma.presentation.web.blueprints._helpers import (
    container,
    export_options_from_query,
    file_response,
    parse_body,
)
from sigma.presentation.web.schemas.notebooks import (
    CreateNotebookRequest,
    ExecuteRequest,
    RenameNotebookRequest,
    SaveNotebookRequest,
)
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("console_api", __name__, url_prefix="/api/notebooks")
bp.before_request(require_console)  # running code is a privilege the administrator grants per account


@bp.get("")
def list_notebooks():
    return jsonify({"notebooks": to_jsonable(container().list_notebooks.execute())})


@bp.post("")
def create_notebook():
    body = parse_body(CreateNotebookRequest)
    return jsonify({"notebook": to_jsonable(container().create_notebook.execute(body.name, body.language))}), 201


@bp.post("/import")
def import_notebook():
    upload = request.files.get("file")
    if upload is None:
        raise InvalidInputError("Seleccione un archivo .ipynb para importar.")
    document = container().import_notebook.execute(upload.filename or "", upload.read())
    return jsonify({"notebook": to_jsonable(document)}), 201


@bp.get("/<notebook_id>")
def get_notebook(notebook_id: str):
    return jsonify({"notebook": to_jsonable(container().get_notebook.execute(notebook_id))})


@bp.put("/<notebook_id>")
def save_notebook(notebook_id: str):
    body = parse_body(SaveNotebookRequest)
    document = container().save_notebook.execute(notebook_id, body.cells, body.metadata, body.name, body.base_revision)
    return jsonify({"notebook": _stamp(document)})


@bp.patch("/<notebook_id>")
def rename_notebook(notebook_id: str):
    body = parse_body(RenameNotebookRequest)
    document = container().rename_notebook.execute(notebook_id, body.name)
    return jsonify({"notebook": _stamp(document)})


@bp.delete("/<notebook_id>")
def delete_notebook(notebook_id: str):
    container().delete_notebook.execute(notebook_id)
    return jsonify({"deleted": notebook_id})


@bp.get("/<notebook_id>/export")
def export_notebook(notebook_id: str):
    name, content = container().export_notebook.execute(notebook_id)
    filename = re.sub(r"[^\w\- ]+", "_", name).strip() or "notebook"
    return Response(
        content,
        mimetype="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}.ipynb"'},
    )


@bp.post("/<notebook_id>/execute")
def execute(notebook_id: str):
    body = parse_body(ExecuteRequest)
    result = container().execute_code.execute(notebook_id, body.source, body.timeout)
    return jsonify(to_jsonable(result))


@bp.get("/<notebook_id>/kernel")
def kernel_status(notebook_id: str):
    return jsonify({"kernel": to_jsonable(container().control_kernel.status(notebook_id))})


@bp.post("/<notebook_id>/kernel/start")
def kernel_start(notebook_id: str):
    return jsonify({"kernel": to_jsonable(container().control_kernel.start(notebook_id))})


@bp.post("/<notebook_id>/kernel/interrupt")
def kernel_interrupt(notebook_id: str):
    container().control_kernel.interrupt(notebook_id)
    return jsonify({"interrupted": True})


@bp.post("/<notebook_id>/kernel/restart")
def kernel_restart(notebook_id: str):
    container().control_kernel.restart(notebook_id)
    return jsonify({"restarted": True})


def _stamp(document) -> dict:
    """What the editor needs after a save: the new revision to send with the next one."""
    return {"id": document.id, "name": document.name, "updated_at": document.updated_at, "revision": document.revision}


def _flag(name: str, default: bool) -> bool:
    value = request.args.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@bp.get("/<notebook_id>/report/<fmt>")
def notebook_report(notebook_id: str, fmt: str):
    """The notebook as a report (PDF, Word or Excel); ``text``/``code``/``outputs`` choose what to include."""
    content = NotebookReportOptions(
        include_text=_flag("text", True),
        include_code=_flag("code", True),
        include_outputs=_flag("outputs", True),
        number_cells=_flag("numbers", False),
        break_before_h1=_flag("break_h1", False),
        break_before_h2=_flag("break_h2", False),
    )
    exported = container().export_notebook_report.execute(notebook_id, fmt, export_options_from_query(), content)
    return file_response(exported)
