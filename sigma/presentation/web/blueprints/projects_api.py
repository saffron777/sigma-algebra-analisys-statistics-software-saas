"""Survey project endpoints (create, list, load, save, analyse, delete)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from sigma.presentation.web.blueprints._helpers import container, parse_body
from sigma.presentation.web.reports import markets as report
from sigma.presentation.web.schemas.projects import CreateProjectRequest, ProjectAnalysisRequest, SaveProjectRequest
from sigma.presentation.web.serializers import to_jsonable

bp = Blueprint("projects_api", __name__, url_prefix="/api/projects")


@bp.get("")
def list_projects():
    return jsonify({"projects": to_jsonable(container().list_projects.execute())})


@bp.post("")
def create_project():
    body = parse_body(CreateProjectRequest)
    project = container().create_project.execute(
        body.name, body.population_size, body.sample_size, body.indicators, body.design
    )
    return jsonify({"project": to_jsonable(project)}), 201


@bp.get("/<project_id>")
def get_project(project_id: str):
    project = container().get_project.execute(project_id)
    payload: dict = {"project": to_jsonable(project)}
    if project.dataset_id:
        info, frame = container().preview_dataset.execute(project.dataset_id, 1_000_000, 0)
        payload["data"] = {
            "columns": [{"name": c.name, "kind": c.kind} for c in info.columns],
            "rows": [to_jsonable(row) for row in frame.to_numpy(dtype=object).tolist()],
        }
    return jsonify(payload)


@bp.put("/<project_id>")
def save_project(project_id: str):
    body = parse_body(SaveProjectRequest)
    project = container().save_project.execute(project_id, body.model_dump())
    return jsonify({"project": to_jsonable(project)})


@bp.delete("/<project_id>")
def delete_project(project_id: str):
    container().delete_project.execute(project_id)
    return jsonify({"deleted": project_id})


@bp.post("/<project_id>/analysis")
def run_project_analysis(project_id: str):
    body = parse_body(ProjectAnalysisRequest)
    result = container().run_project_analysis.execute(
        project_id, {"analysis_type": body.analysis_type, "params": body.params}
    )
    builder = report.project_report(body.analysis_type, result)
    return jsonify({"result": to_jsonable(result), "report": builder.report(), "figures": builder.figures})
