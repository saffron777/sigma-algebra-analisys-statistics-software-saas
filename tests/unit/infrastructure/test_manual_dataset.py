"""Manual dataset creation (SPSS/Minitab-style data editor backend)."""

import itertools
from pathlib import Path

import pandas as pd
import pytest

from sigma.application.use_cases.datasets import CreateManualDataset
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from tests.support import add_user

_counter = itertools.count(1)


@pytest.fixture()
def datasets():
    path = Path(f"E:/Sigma/.test-manual-{next(_counter)}.db")
    path.unlink(missing_ok=True)
    database = Database(path)
    uid = add_user(database)
    yield SqlDatasetRepository(database, uid)
    database.dispose()
    path.unlink(missing_ok=True)


def test_create_manual_dataset_with_typed_variables(datasets):
    create = CreateManualDataset(datasets)
    info = create.execute(
        "estudio",
        [
            {"name": "edad", "kind": "numeric"},
            {"name": "grupo", "kind": "categorical"},
            {"name": "activo", "kind": "boolean"},
            {"name": "fecha", "kind": "datetime"},
        ],
        [
            {"edad": "25", "grupo": "A", "activo": "sí", "fecha": "2024-01-15"},
            {"edad": "", "grupo": "B", "activo": "no", "fecha": "2024-02-20"},
            {"edad": "30", "grupo": "A", "activo": "", "fecha": ""},
        ],
    )
    assert info.source_type == "manual"
    assert (info.n_rows, info.n_cols) == (3, 4)
    assert {c.name: c.kind for c in info.columns} == {
        "edad": "numeric", "grupo": "categorical", "activo": "boolean", "fecha": "datetime",
    }
    data = datasets.get(info.id).data
    assert data["edad"].iloc[0] == 25.0 and pd.isna(data["edad"].iloc[1])
    assert data["activo"].iloc[0] == True and data["activo"].iloc[1] == False and pd.isna(data["activo"].iloc[2])  # noqa: E712
    assert data["grupo"].tolist()[:2] == ["A", "B"]
    assert data["fecha"].iloc[0] == pd.Timestamp("2024-01-15") and pd.isna(data["fecha"].iloc[2])


def test_create_manual_dataset_validations(datasets):
    create = CreateManualDataset(datasets)
    with pytest.raises(InvalidInputError):
        create.execute("x", [], [])  # no variables
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "a", "kind": "numeric"}, {"name": "A", "kind": "text"}], [])  # duplicate names
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "a", "kind": "no-tipo"}], [])  # invalid kind
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "  ", "kind": "numeric"}], [])  # blank name
    info = create.execute("vacio", [{"name": "a", "kind": "numeric"}], [])  # empty is allowed with one variable
    assert info.n_rows == 0 and info.n_cols == 1
