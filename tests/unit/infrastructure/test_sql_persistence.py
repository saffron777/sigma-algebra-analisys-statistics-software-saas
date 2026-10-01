import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

from sigma.application.dto import SaveResultCommand
from sigma.application.use_cases.results import (
    DeleteResult,
    ExportResult,
    GetResult,
    ListResults,
    SaveResult,
    UpdateResult,
)
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError, ResultNotFoundError
from sigma.infrastructure.persistence.sql.dataset_repository import SqlDatasetRepository
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.result_repository import SqlResultRepository
from sigma.infrastructure.persistence.sql.schema import SCHEMA_VERSION
from tests.support import add_user, scalar, table_names


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "sigma.db")


@pytest.fixture
def user_id(db):
    return add_user(db)


@pytest.fixture
def datasets(db, user_id):
    return SqlDatasetRepository(db, user_id)


@pytest.fixture
def results(db, user_id):
    return SqlResultRepository(db, user_id)


def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entero": np.arange(5, dtype="int64"),
            "decimal": [1.5, np.nan, 3.25, 4.0, -2.0],
            "texto": ["a", None, "c", "ñandú", "e,f"],
            "flag": [True, False, True, True, False],
            "fecha": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-02-29 13:45:10", None, "2025-12-31 00:00:00", "2000-01-01 00:00:00"]
            ),
            "cat": pd.Categorical(["x", "y", "x", None, "z"]),
            "nulable": pd.array([1, None, 3, 4, 5], dtype="Int64"),
        }
    )


def test_schema_version_and_wal(db):
    assert scalar(db, "SELECT version FROM sigma_schema") == SCHEMA_VERSION
    assert scalar(db, "PRAGMA journal_mode") == "wal"
    assert scalar(db, "PRAGMA foreign_keys") == 1


def test_dataset_roundtrip_preserves_dtypes_and_values(datasets):
    original = mixed_frame()
    info = datasets.add("mixto", original, "csv", {"file_name": "x.csv"})
    assert (info.n_rows, info.n_cols) == (5, 7)
    loaded = datasets.get(info.id).data
    assert list(loaded.columns) == list(original.columns)
    assert loaded["entero"].dtype == original["entero"].dtype
    assert str(loaded["fecha"].dtype).startswith("datetime64")
    assert isinstance(loaded["cat"].dtype, pd.CategoricalDtype)
    assert str(loaded["nulable"].dtype) == "Int64"
    assert loaded["flag"].dtype == bool
    pd.testing.assert_series_equal(loaded["decimal"], original["decimal"])
    assert loaded["texto"].tolist()[3] == "ñandú" and pd.isna(loaded["texto"].iloc[1])
    assert loaded["fecha"].iloc[1] == pd.Timestamp("2024-02-29 13:45:10") and pd.isna(loaded["fecha"].iloc[2])
    assert loaded["cat"].astype(object).tolist()[:3] == ["x", "y", "x"]
    assert loaded["nulable"].tolist()[0] == 1 and pd.isna(loaded["nulable"].iloc[1])


def test_info_lists_column_kinds(datasets):
    info = datasets.add("kinds", mixed_frame(), "csv", {})
    kinds = {c.name: c.kind for c in info.columns}
    assert kinds["entero"] == "numeric" and kinds["flag"] == "boolean" and kinds["fecha"] == "datetime"
    assert kinds["cat"] == "categorical"


def test_data_survives_a_new_connection(tmp_path):
    path = tmp_path / "persist.db"
    database = Database(path)
    owner = add_user(database)
    first = SqlDatasetRepository(database, owner)
    created = first.add("persistente", pd.DataFrame({"a": [1, 2, 3]}), "csv", {})
    second = SqlDatasetRepository(Database(path), owner)  # simulates restarting the app
    assert second.get_by_name("PERSISTENTE").data["a"].tolist() == [1, 2, 3]
    assert [d.id for d in second.list()] == [created.id]


def test_names_are_unique_case_insensitive(datasets):
    datasets.add("Ventas", pd.DataFrame({"a": [1]}), "csv", {})
    with pytest.raises(InvalidInputError):
        datasets.add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    other = datasets.add("otro", pd.DataFrame({"a": [1]}), "csv", {})
    with pytest.raises(InvalidInputError):
        datasets.rename(other.id, "VENTAS")
    assert datasets.rename(other.id, "renombrado").name == "renombrado"
    assert len(datasets.list()) == 2  # the failed add left nothing behind


def test_preview_pagination_and_replace(datasets):
    info = datasets.add("grande", pd.DataFrame({"n": range(100)}), "csv", {})
    assert datasets.preview(info.id, 10, 5)["n"].tolist() == list(range(5, 15))
    updated = datasets.replace_data(info.id, pd.DataFrame({"n": [1.0, 2.0], "m": ["a", "b"]}))
    assert (updated.n_rows, updated.n_cols) == (2, 2)
    assert datasets.get(info.id).data["m"].tolist() == ["a", "b"]


def test_delete_drops_the_data_table(datasets, db):
    info = datasets.add("temporal", pd.DataFrame({"a": [1]}), "csv", {})
    physical = scalar(db, "SELECT table_name FROM datasets WHERE id = :i", i=info.id)
    assert physical in table_names(db, "ds_")
    datasets.delete(info.id)
    assert physical not in table_names(db)
    with pytest.raises(DatasetNotFoundError):
        datasets.get(info.id)
    with pytest.raises(DatasetNotFoundError):
        datasets.get_info("not-a-valid-id'; DROP TABLE datasets;--")


def test_columns_with_awkward_names(datasets):
    frame = pd.DataFrame({'col "comillas"': [1, 2], "con espacio": [3, 4], "ñ-ü": [5, 6]})
    info = datasets.add("raros", frame, "csv", {})
    assert list(datasets.get(info.id).data.columns) == list(frame.columns)


def test_large_dataset_smoke(datasets):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(200_000, 10)), columns=[f"v{i}" for i in range(10)])
    started = time.perf_counter()
    info = datasets.add("grande", frame, "csv", {})
    loaded = datasets.get(info.id).data
    assert len(loaded) == 200_000
    assert loaded["v3"].sum() == pytest.approx(frame["v3"].sum())
    assert time.perf_counter() - started < 60


# ------------------------------------------------------------------------- saved results
@pytest.fixture
def save(results, datasets):
    return SaveResult(results, datasets)


def command(**overrides):
    base = {
        "module": "regression",
        "analysis_type": "ols_multiple",
        "title": "Regresión ventas",
        "notes": "primer intento",
        "params": {"y": "ventas"},
        "result": {"r2": 0.91},
        "report": {"title": "x", "blocks": []},
        "figures": [{"data": [], "layout": {}}],
    }
    return SaveResultCommand(**{**base, **overrides})


def test_save_get_and_export(save, results, datasets):
    info = datasets.add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    saved = save.execute(command(dataset_id=info.id))
    assert saved.dataset_name == "ventas"
    fetched = GetResult(results).execute(saved.id)
    assert fetched.result == {"r2": 0.91} and fetched.figures == [{"data": [], "layout": {}}]
    exported = ExportResult(results).execute(saved.id)
    assert exported["title"] == "Regresión ventas" and exported["params"] == {"y": "ventas"}


def test_list_filters_search_and_paging(save, results):
    save.execute(command(title="Regresión ventas"))
    save.execute(command(module="anova", title="ANOVA región", notes="100% real_"))
    save.execute(command(module="anova", title="Otro ANOVA"))
    lister = ListResults(results)
    items, total = lister.execute()
    assert total == 3
    assert [i.module for i in lister.execute(module="anova")[0]] == ["anova", "anova"]
    assert lister.execute(search="reg")[1] == 2  # "Regresión ventas" (case-insensitive) + "ANOVA región"
    assert lister.execute(search="100%")[1] == 1  # LIKE wildcard is escaped
    assert lister.execute(search="%")[1] == 1
    assert len(lister.execute(limit=2)[0]) == 2 and lister.execute(limit=2, offset=2)[1] == 3
    with pytest.raises(InvalidInputError):
        lister.execute(module="nope")


def test_update_and_delete(save, results):
    saved = save.execute(command())
    updated = UpdateResult(results).execute(saved.id, "Nuevo título", "nota")
    assert (updated.title, updated.notes) == ("Nuevo título", "nota")
    with pytest.raises(InvalidInputError):
        UpdateResult(results).execute(saved.id, "   ", None)
    DeleteResult(results).execute(saved.id)
    with pytest.raises(ResultNotFoundError):
        GetResult(results).execute(saved.id)


def test_dataset_deletion_keeps_results_with_a_snapshot_of_its_name(save, results, datasets):
    info = datasets.add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    saved = save.execute(command(dataset_id=info.id))
    datasets.delete(info.id)
    kept = GetResult(results).execute(saved.id)
    assert kept.dataset_id is None and kept.dataset_name == "ventas"


def test_renaming_a_dataset_updates_the_name_shown_on_results(save, results, datasets):
    info = datasets.add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    saved = save.execute(command(dataset_id=info.id))
    datasets.rename(info.id, "ventas 2024")
    assert GetResult(results).execute(saved.id).dataset_name == "ventas 2024"


def test_save_validation(save):
    with pytest.raises(InvalidInputError):
        save.execute(command(module="otro"))
    with pytest.raises(InvalidInputError):
        save.execute(command(title=" "))
    with pytest.raises(InvalidInputError):
        save.execute(command(result={}, report={}))
    with pytest.raises(InvalidInputError):
        save.execute(command(report={"blob": "x" * (5 * 1024 * 1024 + 1)}))
    assert save.execute(command(dataset_id="0" * 32)).dataset_id is None  # unknown dataset is tolerated


def test_sql_injection_in_search_is_inert(save, results, db):
    save.execute(command())
    ListResults(results).execute(search="'; DROP TABLE saved_results; --")
    assert scalar(db, "SELECT COUNT(*) FROM saved_results") == 1


def test_concurrent_connections_can_read_while_writing(db, datasets):
    datasets.add("a", pd.DataFrame({"x": [1]}), "csv", {})
    writer = sqlite3.connect(db.settings.path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        assert len(datasets.list()) == 1  # WAL: readers are not blocked by an open writer
    finally:
        writer.rollback()
        writer.close()
