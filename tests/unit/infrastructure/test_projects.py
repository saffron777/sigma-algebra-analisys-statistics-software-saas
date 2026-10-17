"""Survey project use cases on SQL repositories."""

import pytest

from sigma.application.use_cases.projects import (
    CreateProject,
    DeleteProject,
    GetProject,
    ListProjects,
    RunProjectAnalysis,
    SaveProject,
)
from sigma.domain.errors import InvalidInputError, ProjectNotFoundError
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.project_repository import SqlProjectRepository
from tests.support import add_user


@pytest.fixture()
def repos(tmp_path):
    path = tmp_path / "projects.db"
    database = Database(path)
    uid = add_user(database)
    yield SqlProjectRepository(database, uid), SqlDatasetRepository(database, uid)
    database.dispose()


def test_project_lifecycle(repos):
    projects, datasets = repos
    design = [
        {"name": "satisfaccion", "kind": "numeric"},
        {"name": "genero", "kind": "categorical", "categories": "Hombre, Mujer"},
    ]
    created = CreateProject(projects).execute("Estudio", 10000, 385, ["estimate"], design)
    assert created.name == "Estudio" and created.sample_size == 385 and created.population_size == 10000
    assert created.dataset_id is None and created.n_responses == 0
    assert [s.name for s in ListProjects(projects).execute()] == ["Estudio"]

    # saving the responses creates the underlying dataset
    data = [{"satisfaccion": "5", "genero": "Hombre"}, {"satisfaccion": "4", "genero": "Mujer"}]
    saved = SaveProject(projects, datasets).execute(
        created.id,
        {"name": "Estudio v2", "population_size": 20000, "sample_size": 500,
         "indicators": ["estimate", "crosstab"], "design": design, "data": data},
    )
    assert saved.name == "Estudio v2" and saved.n_responses == 2
    assert saved.dataset_id and saved.dataset_name

    estimate = RunProjectAnalysis(projects, datasets).execute(
        created.id, {"analysis_type": "estimate", "params": {"variable": "satisfaccion", "kind": "mean"}}
    )
    assert estimate.variable == "satisfaccion" and estimate.n == 2

    DeleteProject(projects).execute(created.id)
    with pytest.raises(ProjectNotFoundError):
        GetProject(projects).execute(created.id)


def test_duplicate_project_name_rejected(repos):
    projects, _ = repos
    CreateProject(projects).execute("Único", design=[{"name": "a", "kind": "numeric"}])
    with pytest.raises(InvalidInputError):
        CreateProject(projects).execute("único", design=[{"name": "b", "kind": "text"}])
