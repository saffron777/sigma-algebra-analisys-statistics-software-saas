"""Use cases for survey projects (market-research studies)."""

from __future__ import annotations

import uuid
from typing import Any

from sigma.application.ports import DatasetRepository, ProjectRepository
from sigma.application.use_cases.datasets import _build_manual_frame
from sigma.domain.errors import InvalidInputError
from sigma.domain.projects import Project, ProjectSummary
from sigma.domain.statistics.survey_estimation import estimate_survey
from sigma.domain.statistics.survey_sampling import sample_size_mean, sample_size_proportion
from sigma.domain.statistics.survey_tabulation import crosstab, likert_summary
from sigma.infrastructure.persistence.sql.helpers import now_iso

MAX_NAME_LENGTH = 120
ANALYSES = {"sample_size", "estimate", "crosstab", "likert"}


def _validate_name(name: str | None) -> str:
    cleaned = " ".join((name or "").split())
    if not cleaned:
        raise InvalidInputError("Escriba un nombre para el proyecto.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise InvalidInputError(f"El nombre no puede superar {MAX_NAME_LENGTH} caracteres.")
    return cleaned


def _clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise InvalidInputError("El tamaño de la población/muestra debe ser un número entero.") from None
    if number < 0:
        raise InvalidInputError("El tamaño de la población/muestra no puede ser negativo.")
    return number


class CreateProject:
    """Creates a survey project (design and data can be filled in later)."""

    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(
        self,
        name: str | None,
        population_size: Any = None,
        sample_size: Any = None,
        indicators: list[str] | None = None,
        design: list[dict[str, Any]] | None = None,
    ) -> Project:
        now = now_iso()
        project = Project(
            id=uuid.uuid4().hex,
            user_id=0,  # the repository binds it to the signed-in user
            name=_validate_name(name),
            population_size=_clean_int(population_size),
            sample_size=_clean_int(sample_size),
            indicators=[str(i) for i in (indicators or [])],
            design=[dict(q) for q in (design or [])],
            dataset_id=None,
            dataset_name=None,
            n_responses=0,
            results=None,
            created_at=now,
            updated_at=now,
        )
        return self._projects.add(project)


class ListProjects:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self) -> list[ProjectSummary]:
        return self._projects.list()


class GetProject:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self, project_id: str) -> Project:
        return self._projects.get(project_id)


class SaveProject:
    """Updates metadata/design/results; the responses are kept as a real dataset."""

    def __init__(self, projects: ProjectRepository, datasets: DatasetRepository) -> None:
        self._projects = projects
        self._datasets = datasets

    def execute(self, project_id: str, fields: dict[str, Any]) -> Project:
        project = self._projects.get(project_id)
        name = _validate_name(fields.get("name", project.name))
        project.name = name
        project.population_size = _clean_int(fields.get("population_size", project.population_size))
        project.sample_size = _clean_int(fields.get("sample_size", project.sample_size))
        project.indicators = [str(i) for i in (fields.get("indicators", project.indicators) or [])]
        project.design = [dict(q) for q in (fields.get("design", project.design) or [])]
        project.results = fields.get("results", project.results)

        if fields.get("data") is not None and project.design:
            rows = [dict(r) for r in (fields.get("data") or [])]
            frame = _build_manual_frame(project.design, rows)
            if project.dataset_id:
                info = self._datasets.replace_data(project.dataset_id, frame)
            else:
                info = self._datasets.add(name, frame, "survey", {"project_id": project.id})
            project.dataset_id = info.id
            project.dataset_name = info.name
            project.n_responses = info.n_rows
        return self._projects.update(project)


class DeleteProject:
    def __init__(self, projects: ProjectRepository) -> None:
        self._projects = projects

    def execute(self, project_id: str) -> None:
        self._projects.delete(project_id)


class RunProjectAnalysis:
    """Runs one survey analysis over the project's dataset and returns its result."""

    def __init__(self, projects: ProjectRepository, datasets: DatasetRepository) -> None:
        self._projects = projects
        self._datasets = datasets

    def execute(self, project_id: str, spec: dict[str, Any]) -> Any:
        project = self._projects.get(project_id)
        if not project.dataset_id:
            raise InvalidInputError("El proyecto aún no tiene respuestas guardadas.")
        frame = self._datasets.get(project.dataset_id).data
        kind = str(spec.get("analysis_type") or "")
        params = dict(spec.get("params") or {})
        if kind not in ANALYSES:
            raise InvalidInputError(f"Análisis inválido «{kind}».")
        if kind == "sample_size":
            return self._sample_size(project, params)
        if kind == "estimate":
            variable = str(params.get("variable") or "")
            if not variable or variable not in frame.columns:
                raise InvalidInputError("Elija la variable a estimar.")
            return estimate_survey(
                frame,
                variable,
                str(params.get("kind") or "proportion"),
                float(params.get("confidence", 0.95)),
                list(params.get("methods") or ["wilson"]),
                project.population_size,
                params.get("weight_column"),
                params.get("strata_column"),
                params.get("cluster_column"),
                bool(params.get("apply_fpc", True)),
            )
        if kind == "crosstab":
            row = str(params.get("row_variable") or "")
            col = str(params.get("col_variable") or "")
            if not row or row not in frame.columns or not col or col not in frame.columns:
                raise InvalidInputError("Elija las dos variables para la tabla cruzada.")
            return crosstab(frame, row, col)
        items = list(params.get("items") or [])
        if not items or any(i not in frame.columns for i in items):
            raise InvalidInputError("Elija los ítems de la escala Likert.")
        return likert_summary(frame, items)

    @staticmethod
    def _sample_size(project: Project, params: dict[str, Any]) -> Any:
        kind = str(params.get("kind") or "proportion")
        confidence = float(params.get("confidence", 0.95))
        margin = float(params.get("margin_of_error", 0.05))
        effect = float(params.get("design_effect", 1.0))
        if kind == "proportion":
            return sample_size_proportion(
                confidence, margin, float(params.get("proportion", 0.5)), project.population_size, effect
            )
        if kind == "mean":
            if params.get("std_dev") is None:
                raise InvalidInputError("Indique la desviación estándar esperada.")
            return sample_size_mean(confidence, margin, float(params["std_dev"]), project.population_size, effect)
        raise InvalidInputError("El tipo de cálculo debe ser «proportion» o «mean».")
