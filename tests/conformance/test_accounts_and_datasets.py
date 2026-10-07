"""Accounts and datasets behave the same on every database engine."""

import numpy as np
import pandas as pd
import pytest

from sigma.domain.errors import DatasetNotFoundError, InvalidInputError, UserNotFoundError


def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entero": np.arange(5, dtype="int64"),
            "decimal": [1.5, np.nan, 3.25, 4.0, -2.0],
            "texto": ["a", None, "c", "ñandú 日本語 🙂", "e,f"],
            "flag": [True, False, True, True, False],
            "fecha": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-02-29 13:45:10", None, "2025-12-31 00:00:00", "2000-01-01 00:00:00"]
            ),
            "cat": pd.Categorical(["x", "y", "x", None, "z"]),
            "nulable": pd.array([1, None, 3, 4, 5], dtype="Int64"),
        }
    )


# ------------------------------------------------------------------ accounts
def test_user_lifecycle_and_case_insensitive_unique_login(backend):
    users = backend.users()
    created = users.add("Ana", "Ana", "hash1", "admin", True, True)
    assert users.find_with_hash("ana")[1] == "hash1" and users.find_with_hash("ANA")[0].id == created.id
    with pytest.raises(InvalidInputError):
        users.add("ANA", "Otra", "h", "user", False, False)
    assert users.add("Aná", "Con tilde", "h", "user", False, False).id != created.id  # accents are significant
    assert users.get(created.id).auth_epoch == 1
    assert users.set_password(created.id, "hash2").auth_epoch == 2
    assert users.update(created.id, is_active=False).auth_epoch == 3
    assert users.update(created.id, display_name="Nueva").auth_epoch == 3
    assert users.count() == 2 and users.count_active_admins() == 0
    users.touch_login(created.id)
    assert users.get(created.id).last_login_at
    users.delete(created.id)
    with pytest.raises(UserNotFoundError):
        users.get(created.id)
    assert users.find_with_hash("ana") is None and [u.username for u in users.list()] == ["Aná"]


def test_ids_grow_and_are_not_reused_in_a_run(backend):
    users = backend.users()
    first = users.add("a", "A", "x", "user", False, False).id
    second = users.add("b", "B", "x", "user", False, False).id
    users.delete(second)
    third = users.add("c", "C", "x", "user", False, False).id
    assert first < second < third


# ------------------------------------------------------------------ datasets
def test_dataset_roundtrip_preserves_dtypes_and_values(backend, uid):
    repo = backend.datasets(uid)
    original = mixed_frame()
    info = repo.add("mixto", original, "csv", {"file_name": "x.csv"})
    assert (info.n_rows, info.n_cols) == (5, 7)
    loaded = repo.get(info.id).data
    assert list(loaded.columns) == list(original.columns)
    assert loaded["entero"].dtype == original["entero"].dtype
    assert str(loaded["fecha"].dtype).startswith("datetime64")
    assert isinstance(loaded["cat"].dtype, pd.CategoricalDtype) and str(loaded["nulable"].dtype) == "Int64"
    assert loaded["flag"].dtype == bool
    pd.testing.assert_series_equal(loaded["decimal"], original["decimal"])
    assert loaded["texto"].tolist()[3] == "ñandú 日本語 🙂" and pd.isna(loaded["texto"].iloc[1])
    assert loaded["fecha"].iloc[1] == pd.Timestamp("2024-02-29 13:45:10") and pd.isna(loaded["fecha"].iloc[2])
    assert loaded["cat"].astype(object).tolist()[:3] == ["x", "y", "x"] and pd.isna(loaded["nulable"].iloc[1])


def test_row_order_paging_and_replacement(backend, uid):
    repo = backend.datasets(uid)
    info = repo.add("grande", pd.DataFrame({"n": range(100)}), "csv", {})
    assert repo.get(info.id).data["n"].tolist() == list(range(100))
    assert repo.preview(info.id, 10, 5)["n"].tolist() == list(range(5, 15))
    assert repo.preview(info.id, 10, 95)["n"].tolist() == list(range(95, 100))
    updated = repo.replace_data(info.id, pd.DataFrame({"n": [1.0, 2.0], "m": ["a", "b"]}))
    assert (updated.n_rows, updated.n_cols) == (2, 2) and repo.get(info.id).data["m"].tolist() == ["a", "b"]


def test_names_are_unique_per_user_and_case_insensitive(backend, two_users):
    ana, luis = two_users
    frame = pd.DataFrame({"a": [1]})
    repo = backend.datasets(ana)
    repo.add("Ventas", frame, "csv", {})
    with pytest.raises(InvalidInputError):
        repo.add("ventas", frame, "csv", {})
    assert repo.get_by_name("VENTAS").info.name == "Ventas"
    other = repo.add("otro", frame, "csv", {})
    with pytest.raises(InvalidInputError):
        repo.rename(other.id, "VENTAS")
    assert repo.rename(other.id, "renombrado").name == "renombrado" and len(repo.list()) == 2
    backend.datasets(luis).add("VENTAS", frame, "csv", {})  # the same name under another account is fine


def test_a_foreign_dataset_looks_missing(backend, two_users):
    ana, luis = two_users
    info = backend.datasets(ana).add("privado", pd.DataFrame({"x": [1]}), "csv", {})
    intruder = backend.datasets(luis)
    for call in (
        lambda: intruder.get_info(info.id),
        lambda: intruder.get(info.id),
        lambda: intruder.preview(info.id, 5, 0),
        lambda: intruder.rename(info.id, "robado"),
        lambda: intruder.delete(info.id),
        lambda: intruder.get_by_name("privado"),
        lambda: intruder.replace_data(info.id, pd.DataFrame({"x": [9]})),
    ):
        with pytest.raises(DatasetNotFoundError):
            call()
    assert intruder.list() == [] and backend.datasets(ana).get(info.id).data["x"].tolist() == [1]


def test_delete_and_malformed_ids(backend, uid):
    repo = backend.datasets(uid)
    info = repo.add("temporal", pd.DataFrame({"a": [1]}), "csv", {})
    repo.delete(info.id)
    for bad in (info.id, "not-a-valid-id'; DROP TABLE datasets;--", ""):
        with pytest.raises(DatasetNotFoundError):
            repo.get(bad)
    assert repo.list() == []


def test_awkward_column_names_and_text(backend, uid):
    frame = pd.DataFrame(
        {'col "comillas"': [1, 2], "con espacio": [3, 4], "ñ-ü": [5, 6], "x" * 200: [7, 8], "SELECT": [9, 10]}
    )
    frame["largo"] = ["z" * 100_000, ""]
    repo = backend.datasets(uid)
    info = repo.add("raros", frame, "csv", {})
    back = repo.get(info.id).data
    assert list(back.columns) == list(frame.columns)
    assert back["largo"].tolist()[0] == "z" * 100_000
    assert pd.isna(back["largo"].iloc[1]) or back["largo"].iloc[1] == ""  # Oracle cannot tell '' from NULL


def test_empty_and_single_cell_datasets(backend, uid):
    repo = backend.datasets(uid)
    empty = repo.add("vacio", pd.DataFrame({"a": pd.Series([], dtype="float64")}), "csv", {})
    assert repo.get(empty.id).data.shape == (0, 1) and empty.n_rows == 0
    one = repo.add("uno", pd.DataFrame({"a": [1]}), "csv", {})
    assert repo.get(one.id).data.shape == (1, 1)


def test_non_finite_numbers_become_missing(backend, uid):
    repo = backend.datasets(uid)
    info = repo.add("inf", pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan]}), "csv", {})
    assert repo.get(info.id).data["x"].isna().tolist() == [False, True, True, True]


def test_wide_datasets_and_the_engine_limit(backend, uid):
    repo = backend.datasets(uid)
    wide = pd.DataFrame(np.arange(3 * 250).reshape(3, 250), columns=[f"v{i}" for i in range(250)])
    info = repo.add("ancho", wide, "csv", {})
    assert repo.get(info.id).data.shape == (3, 250)
    if backend.settings.is_relational:
        too_wide = pd.DataFrame(np.zeros((1, 2500)), columns=[f"w{i}" for i in range(2500)])
        with pytest.raises(InvalidInputError):
            repo.add("demasiado", too_wide, "csv", {})
        assert [d.name for d in repo.list()] == ["ancho"]  # the failed add left nothing behind


def test_many_rows_cross_the_batch_boundary(backend, uid):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=12_345), "b": rng.integers(0, 10, size=12_345), "t": "x"})
    repo = backend.datasets(uid)
    info = repo.add("lote", frame, "csv", {})
    loaded = repo.get(info.id).data
    assert len(loaded) == 12_345 and loaded["a"].sum() == pytest.approx(frame["a"].sum())
    assert repo.preview(info.id, 5, 12_340)["b"].tolist() == frame["b"].iloc[12_340:].tolist()


class NoKernels:
    def shutdown_owner(self, owner: str) -> None:
        pass


def test_deleting_an_account_removes_its_datasets(backend, uid, tmp_path):
    repo = backend.datasets(uid)
    repo.add("uno", pd.DataFrame({"x": [1]}), "csv", {})
    repo.add("dos", pd.DataFrame({"x": [2]}), "csv", {})
    backend.purger(tmp_path / "nb", NoKernels()).purge(uid)
    backend.users().delete(uid)
    assert backend.describe()["counts"]["datasets"] == 0
    if backend.settings.is_relational:
        from sqlalchemy import inspect

        assert [t for t in inspect(backend.database.engine).get_table_names() if t.startswith("ds_")] == []


def test_describe_reports_engine_and_counts(backend, uid):
    backend.datasets(uid).add("uno", pd.DataFrame({"x": [1]}), "csv", {})
    info = backend.describe()
    assert info["engine"] == backend.settings.engine and info["counts"]["users"] == 1
    assert info["counts"]["datasets"] == 1 and info["server_version"]


# ------------------------------------------------------------------ manual dataset creation
def test_create_manual_dataset_with_typed_variables(backend, uid):
    from sigma.application.use_cases.datasets import CreateManualDataset

    create = CreateManualDataset(backend.datasets(uid))
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
    kinds = {c.name: c.kind for c in info.columns}
    assert kinds == {"edad": "numeric", "grupo": "categorical", "activo": "boolean", "fecha": "datetime"}
    data = backend.datasets(uid).get(info.id).data
    assert data["edad"].iloc[0] == 25.0 and pd.isna(data["edad"].iloc[1])
    assert data["activo"].iloc[0] is True and data["activo"].iloc[1] is False and pd.isna(data["activo"].iloc[2])
    assert data["grupo"].tolist()[:2] == ["A", "B"]
    assert data["fecha"].iloc[0] == pd.Timestamp("2024-01-15") and pd.isna(data["fecha"].iloc[2])


def test_create_manual_dataset_validations(backend, uid):
    from sigma.application.use_cases.datasets import CreateManualDataset

    create = CreateManualDataset(backend.datasets(uid))
    with pytest.raises(InvalidInputError):
        create.execute("x", [], [])  # no variables
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "a", "kind": "numeric"}, {"name": "A", "kind": "text"}], [])  # duplicate names
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "a", "kind": "no-tipo"}], [])  # invalid kind
    with pytest.raises(InvalidInputError):
        create.execute("x", [{"name": "  ", "kind": "numeric"}], [])  # blank name
    # empty dataset is allowed when there is at least one variable
    info = create.execute("vacio", [{"name": "a", "kind": "numeric"}], [])
    assert info.n_rows == 0 and info.n_cols == 1
