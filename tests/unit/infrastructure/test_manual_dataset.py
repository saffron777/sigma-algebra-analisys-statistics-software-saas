"""Manual dataset creation (SPSS/Minitab-style data editor backend)."""

import pandas as pd
import pytest

from sigma.application.use_cases.datasets import CreateManualDataset
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from tests.support import add_user

CATEGORIES = [{"code": "1", "label": "Masculino"}, {"code": "2", "label": "Femenino"}]


@pytest.fixture()
def datasets(tmp_path):
    path = tmp_path / "manual.db"
    database = Database(path)
    uid = add_user(database)
    yield SqlDatasetRepository(database, uid)
    database.dispose()


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


def test_update_dataset_data(datasets):
    from sigma.application.use_cases.datasets import UpdateDatasetData

    columns = [{"name": "edad", "kind": "numeric"}, {"name": "grupo", "kind": "categorical"}]
    info = CreateManualDataset(datasets).execute(
        "estudio", columns, [{"edad": "25", "grupo": "A"}, {"edad": "30", "grupo": "B"}]
    )
    updated = UpdateDatasetData(datasets).execute(
        info.id, columns, [{"edad": "40", "grupo": "C"}, {"edad": "45", "grupo": "D"}]
    )
    assert (updated.n_rows, updated.n_cols) == (2, 2)
    data = datasets.get(info.id).data
    assert data["edad"].tolist() == [40.0, 45.0]
    assert data["grupo"].astype(object).tolist() == ["C", "D"]


def test_empty_rows_are_dropped(datasets):
    columns = [{"name": "a", "kind": "numeric"}, {"name": "b", "kind": "text"}]
    info = CreateManualDataset(datasets).execute(
        "x", columns,
        [
            {"a": "1", "b": "x"},
            {"a": "", "b": ""},   # fully empty -> dropped
            {"a": "2", "b": ""},  # partially filled -> kept
            {"a": "", "b": ""},   # fully empty -> dropped
            {"a": "", "b": "y"},  # partially filled -> kept
        ],
    )
    assert info.n_rows == 3
    data = datasets.get(info.id).data
    assert data["a"].tolist()[:2] == [1.0, 2.0] and pd.isna(data["a"].iloc[2])
    assert data["b"].tolist() == ["x", "", "y"]


def test_create_manual_dataset_keeps_value_labels(datasets):
    create = CreateManualDataset(datasets)
    info = create.execute(
        "estudio",
        [
            {"name": "sexo", "kind": "categorical", "categories": CATEGORIES},
            {"name": "edad", "kind": "numeric"},
        ],
        [
            {"sexo": "1", "edad": "30"},
            {"sexo": "Femenino", "edad": "25"},
            {"sexo": 1, "edad": "40"},
        ],
    )
    assert info.source_meta["variables"] == [
        {"name": "sexo", "kind": "categorical", "categories": CATEGORIES}
    ]
    # The data stores the labels; a typed code (as text or number) is translated before it reaches the frame.
    assert datasets.get(info.id).data["sexo"].astype(object).tolist() == [
        "Masculino", "Femenino", "Masculino"
    ]


def test_manual_value_label_validations(datasets):
    create = CreateManualDataset(datasets)
    invalid = [
        ([{"code": "1", "label": ""}], "código y una etiqueta"),
        ([{"code": "1", "label": "A"}, {"code": "1", "label": "B"}], "repetido"),
        ([{"code": "1", "label": "A"}, {"code": "2", "label": "a"}], "repetida"),
        ([{"code": "1", "label": "A"}, {"code": "A", "label": "B"}], "coincide"),
        ([{"code": "1", "label": "A;B"}], "no pueden contener"),
    ]
    for categories, message in invalid:
        with pytest.raises(InvalidInputError, match=message):
            create.execute("x", [{"name": "sexo", "kind": "categorical", "categories": categories}], [])


def test_update_dataset_data_updates_value_labels(datasets):
    from sigma.application.use_cases.datasets import UpdateDatasetData

    create = CreateManualDataset(datasets)
    info = create.execute(
        "estudio", [{"name": "sexo", "kind": "categorical", "categories": CATEGORIES}], [{"sexo": "1"}]
    )
    updated = UpdateDatasetData(datasets).execute(
        info.id,
        [{"name": "sexo", "kind": "categorical", "categories": CATEGORIES[:1]}],
        [{"sexo": "Masculino"}],
    )
    assert updated.source_meta["source_type"] == "manual"
    assert updated.source_meta["variables"][0]["categories"] == CATEGORIES[:1]
    assert datasets.get(info.id).data["sexo"].astype(object).tolist() == ["Masculino"]


def test_updating_imported_data_keeps_the_rest_of_source_meta(datasets):
    from sigma.application.use_cases.datasets import UpdateDatasetData

    info = datasets.add("importado", pd.DataFrame({"a": [1]}), "csv", {"file_name": "x.csv"})
    updated = UpdateDatasetData(datasets).execute(info.id, [{"name": "a", "kind": "numeric"}], [{"a": "2"}])
    assert updated.source_meta == {"file_name": "x.csv"}
    assert "variables" not in updated.source_meta
