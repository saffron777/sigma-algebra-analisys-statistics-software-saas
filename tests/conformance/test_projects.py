"""Survey projects behave the same on every database engine."""

import pytest

from sigma.application.use_cases.projects import CreateProject, DeleteProject, GetProject, SaveProject
from sigma.domain.errors import ProjectNotFoundError


def test_project_roundtrip(backend, uid):
    projects = backend.projects(uid)
    datasets = backend.datasets(uid)
    design = [{"name": "edad", "kind": "numeric"}]
    created = CreateProject(projects).execute("Encuesta", 1000, 278, ["estimate"], design)
    assert created.name == "Encuesta" and created.population_size == 1000 and created.sample_size == 278
    assert created.dataset_id is None

    saved = SaveProject(projects, datasets).execute(
        created.id, {"design": design, "data": [{"edad": "30"}, {"edad": "41"}]}
    )
    assert saved.n_responses == 2 and saved.dataset_id
    assert datasets.get(saved.dataset_id).data["edad"].tolist() == [30.0, 41.0]

    loaded = GetProject(projects).execute(created.id)
    assert loaded.dataset_id == saved.dataset_id
    assert [s.name for s in projects.list()] == ["Encuesta"]
    DeleteProject(projects).execute(created.id)
    with pytest.raises(ProjectNotFoundError):
        GetProject(projects).execute(created.id)
