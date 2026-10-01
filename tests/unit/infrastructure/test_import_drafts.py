from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sigma.application.ports import ImportRequest, StagedImport
from sigma.application.use_cases.datasets import PreviewImport, safe_options
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.importers.registry import DefaultImporterRegistry
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.import_draft_store import SqlImportDraftStore, purge_stale_drafts
from tests.support import add_user, execute, fetch_all, scalar, table_names


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "drafts.db")


@pytest.fixture
def drafts(db):
    return SqlImportDraftStore(db, add_user(db))


def staged(frame=None, source="csv", name="datos"):
    frame = pd.DataFrame({"x": [1, 2, 3]}) if frame is None else frame
    return StagedImport(frame, source, {"source_type": source, "file_name": "a.csv"}, name, {"delimiter": "comma"})


def tables(db):
    return table_names(db, "stg_")


# ------------------------------------------------------------------ file
def test_one_draft_per_source_and_a_new_file_replaces_the_old_one(db, drafts):
    first = drafts.attach_file("csv", "uno.csv", b"a\n1\n")
    token = drafts.put(staged(), first.id, {"delimiter": "comma"})
    assert token == first.id and drafts.info(token).has_preview and len(tables(db)) == 1
    again = drafts.attach_file("csv", "dos.csv", b"b\n2\n")
    assert again.id == first.id and again.file_name == "dos.csv" and again.file_size == 4
    assert not again.has_preview and again.options == {} and tables(db) == []  # the old preview is stale
    drafts.attach_file("excel", "libro.xlsx", b"PK")
    assert {d.source_type for d in drafts.list()} == {"csv", "excel"}
    assert drafts.file(first.id)[1] == b"b\n2\n"


def test_sources_without_a_file_have_a_draft_without_blob(drafts):
    token = drafts.put(staged(source="sqlserver"), None, {"server": "srv", "query": "SELECT 1"})
    info, blob = drafts.file(token)
    assert blob is None and info.file_name is None and info.options == {"server": "srv", "query": "SELECT 1"}
    assert drafts.put(staged(source="sqlserver"), None, {}) == token  # still one per source type


# ------------------------------------------------------------------ staged table
def test_dtypes_survive_the_round_trip(drafts):
    frame = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-05", "2024-02-10", None]),
            "grupo": pd.Categorical(["a", "b", "a"]),
            "n": pd.array([1, None, 3], dtype="Int64"),
            "x": [1.5, np.nan, 3.25],
            "ok": [True, False, True],
        }
    )
    token = drafts.put(staged(frame), None, {})
    back = drafts.take(token)
    assert back.frame["fecha"].dtype.kind == "M" and back.frame["fecha"].isna().tolist() == [False, False, True]
    assert str(back.frame["grupo"].dtype) == "category" and str(back.frame["n"].dtype) == "Int64"
    assert back.frame["x"].tolist()[0] == 1.5 and back.frame["ok"].tolist() == [True, False, True]
    assert back.source_type == "csv" and back.suggested_name == "datos" and back.detected == {"delimiter": "comma"}
    assert drafts.take(token).frame.shape == (3, 5)  # reading does not consume the draft


def test_preview_returns_columns_first_rows_and_detected_options(drafts):
    token = drafts.put(staged(pd.DataFrame({"x": range(100), "g": ["a"] * 100})), None, {})
    info, columns, rows, detected = drafts.preview(token, 10)
    assert info.n_rows == 100 and info.n_cols == 2 and [c.name for c in columns] == ["x", "g"]
    assert len(rows) == 10 and detected == {"delimiter": "comma"}


def test_discarding(db, drafts):
    token = drafts.put(staged(), drafts.attach_file("csv", "a.csv", b"a\n1").id, {})
    drafts.discard_preview(token)
    info = drafts.info(token)
    assert not info.has_preview and info.file_name == "a.csv" and tables(db) == []
    with pytest.raises(InvalidInputError):
        drafts.take(token)  # nothing staged any more
    drafts.discard(token)
    with pytest.raises(InvalidInputError):
        drafts.info(token)


# ------------------------------------------------------------------ ownership and bad ids
def test_somebody_elses_draft_looks_expired(db, drafts):
    other = SqlImportDraftStore(db, add_user(db, "otro"))
    token = drafts.put(staged(), None, {})
    for call in (
        lambda: other.take(token),
        lambda: other.info(token),
        lambda: other.file(token),
        lambda: other.discard(token),
        lambda: other.discard_preview(token),
        lambda: other.preview(token, 5),
        lambda: other.put(staged(), token, {}),
    ):
        with pytest.raises(InvalidInputError, match="ya no existe"):
            call()
    assert other.list() == [] and drafts.take(token).frame.shape == (3, 1)


@pytest.mark.parametrize("bad", ["", "../x", "x" * 32, "a'; DROP TABLE users;--", "A" * 32])
def test_malformed_ids_are_rejected_without_touching_the_database(drafts, bad):
    for call in (lambda: drafts.take(bad), lambda: drafts.info(bad), lambda: drafts.discard(bad)):
        with pytest.raises(InvalidInputError):
            call()


# ------------------------------------------------------------------ housekeeping
def test_stale_drafts_are_purged_everywhere(db, drafts):
    fresh = drafts.put(staged(), None, {})
    old = drafts.put(staged(source="excel"), None, {})
    stale = (datetime.now(UTC) - timedelta(days=20)).isoformat(timespec="seconds")
    execute(db, "UPDATE import_drafts SET updated_at = :s WHERE id = :i", s=stale, i=old)
    assert purge_stale_drafts(db, 14) == 1
    assert [d.id for d in drafts.list()] == [fresh]
    assert tables(db) == [scalar(db, "SELECT staged_table FROM import_drafts WHERE id = :i", i=fresh)]


def test_purge_removes_every_draft_and_table_of_a_user(db, drafts):
    drafts.put(staged(), None, {})
    drafts.put(staged(source="excel"), None, {})
    assert drafts.purge() == 2 and drafts.list() == [] and tables(db) == []


def test_drafts_bump_the_sync_revision(db):
    user = add_user(db)
    store = SqlImportDraftStore(db, user)

    def revision():
        return scalar(db, "SELECT revision FROM sync_state WHERE user_id = :u AND scope = 'drafts'", u=user) or 0

    assert revision() == 0
    token = store.attach_file("csv", "a.csv", b"a\n1").id
    first = revision()
    store.put(staged(), token, {})
    second = revision()
    store.discard(token)
    assert 0 < first < second < revision()


# ------------------------------------------------------------------ no secrets are kept
def test_secret_options_are_never_stored():
    sql = {"server": "srv", "password": "clave", "query": "SELECT 1"}
    assert safe_options("sqlserver", sql) == {"server": "srv", "query": "SELECT 1"}
    api = {
        "url": "https://x.com/v1?token=SECRETO#frag",
        "headers": "Authorization: Bearer x",
        "body": "{}",
        "params": "k=v",
    }
    assert safe_options("json_api", api) == {"url": "https://x.com/v1"}
    assert safe_options("csv", {"delimiter": ";"}) == {"delimiter": ";"}


def test_preview_import_stores_only_safe_options(db, drafts):
    class FakeImporter:
        source_type = "json_api"

        def read(self, request):
            return pd.DataFrame({"a": [1]}), {"url": "https://x.com/v1"}

    class Registry(DefaultImporterRegistry):
        def get(self, source_type):
            return FakeImporter() if source_type == "json_api" else super().get(source_type)

    options = {"url": "https://x.com/v1?key=TOKEN", "headers": "Authorization: Bearer ZZZ", "records_path": "data"}
    preview = PreviewImport(Registry(), drafts).execute(ImportRequest("json_api", options))
    stored = drafts.info(preview.token).options
    assert stored == {"url": "https://x.com/v1", "records_path": "data"}
    dump = " ".join(str(v) for row in fetch_all(db, "SELECT * FROM import_drafts") for v in row)
    assert "TOKEN" not in dump and "ZZZ" not in dump
