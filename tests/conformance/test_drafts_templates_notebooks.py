"""Import drafts, report templates and notebooks on every engine."""

import json
import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sigma.application.ports import StagedImport
from sigma.domain.errors import (
    ConflictError,
    ImportFailedError,
    InvalidInputError,
    NotebookNotFoundError,
    TemplateNotFoundError,
)
from sigma.domain.report_templates import validate_fields


def staged(frame=None, source="csv", name="datos"):
    frame = pd.DataFrame({"x": [1, 2, 3]}) if frame is None else frame
    return StagedImport(frame, source, {"source_type": source, "file_name": "a.csv"}, name, {"delimiter": "comma"})


# ------------------------------------------------------------------ drafts
def test_one_draft_per_source_and_a_new_file_replaces_the_old_one(backend, uid):
    drafts = backend.drafts(uid)
    first = drafts.attach_file("csv", "uno.csv", b"a\n1\n")
    token = drafts.put(staged(), first.id, {"delimiter": "comma"})
    assert token == first.id and drafts.info(token).has_preview
    again = drafts.attach_file("csv", "dos.csv", b"b\n2\n")
    assert again.id == first.id and again.file_name == "dos.csv" and again.file_size == 4
    assert not again.has_preview and again.options == {}
    drafts.attach_file("excel", "libro.xlsx", b"PK")
    assert {d.source_type for d in drafts.list()} == {"csv", "excel"}
    assert drafts.file(first.id)[1] == b"b\n2\n"


def test_files_larger_than_one_chunk_round_trip(backend, uid):
    drafts = backend.drafts(uid)
    content = np.random.default_rng(1).bytes(8 * 1024 * 1024 + 123) + b"\x00\x01\xff"
    info = drafts.attach_file("csv", "grande.csv", content)
    assert info.file_size == len(content)
    assert drafts.file(info.id)[1] == content


def test_binary_content_with_nul_bytes_survives(backend, uid):
    drafts = backend.drafts(uid)
    blob = bytes(range(256)) * 4
    info = drafts.attach_file("excel", "libro.xlsx", blob)
    assert drafts.file(info.id)[1] == blob


def test_sources_without_a_file_have_a_draft_without_blob(backend, uid):
    drafts = backend.drafts(uid)
    token = drafts.put(staged(source="sqlserver"), None, {"server": "srv", "query": "SELECT 1"})
    info, blob = drafts.file(token)
    assert blob is None and info.file_name is None and info.options == {"server": "srv", "query": "SELECT 1"}
    assert drafts.put(staged(source="sqlserver"), None, {}) == token


def test_staged_dtypes_survive_and_preview_pages(backend, uid):
    drafts = backend.drafts(uid)
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
    assert back.frame["fecha"].dtype.kind == "M" and str(back.frame["grupo"].dtype) == "category"
    assert str(back.frame["n"].dtype) == "Int64" and back.frame["ok"].tolist() == [True, False, True]
    assert back.suggested_name == "datos" and back.detected == {"delimiter": "comma"}
    assert drafts.take(token).frame.shape == (3, 5)  # reading does not consume it
    big = drafts.put(staged(pd.DataFrame({"x": range(100), "g": ["a"] * 100}), source="excel"), None, {})
    info, columns, rows, detected = drafts.preview(big, 10)
    assert info.n_rows == 100 and [c.name for c in columns] == ["x", "g"] and len(rows) == 10
    assert detected == {"delimiter": "comma"}


def test_discarding_and_foreign_drafts(backend, two_users):
    ana, luis = two_users
    drafts = backend.drafts(ana)
    token = drafts.put(staged(), None, {})
    with pytest.raises(InvalidInputError):
        backend.drafts(luis).take(token)  # somebody else's draft looks expired
    with pytest.raises(InvalidInputError):
        backend.drafts(luis).info(token)
    drafts.discard_preview(token)
    assert not drafts.info(token).has_preview
    drafts.discard(token)
    assert drafts.list() == []
    for bad in ("", "../x", "x" * 32, "a'; DROP TABLE users;--"):
        with pytest.raises(InvalidInputError):
            drafts.info(bad)


def test_stale_drafts_are_purged_and_users_can_purge_theirs(backend, uid):
    drafts = backend.drafts(uid)
    old = drafts.put(staged(source="csv"), None, {})
    fresh = drafts.put(staged(source="excel"), None, {})
    stale = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    from sqlalchemy import text

    if backend.settings.is_relational:
        with backend.database.begin() as conn:
            conn.execute(text("UPDATE import_drafts SET updated_at = :s WHERE id = :i"), {"s": stale, "i": old})
        assert backend.purge_stale_drafts(14) == 1
        assert [d.id for d in drafts.list()] == [fresh]
    assert drafts.purge() >= 1 and drafts.list() == []


def test_draft_changes_bump_the_sync_revision(backend, uid):
    drafts, sync = backend.drafts(uid), backend.sync(uid)
    token = drafts.put(staged(), None, {})
    first = sync.revisions()["drafts"]
    drafts.discard(token)
    assert sync.revisions()["drafts"] > first > 0


# ------------------------------------------------------------------ templates
def fields(name="Institucional", **overrides):
    return validate_fields(name, **{"organization": "Ñandú S.A.", "primary_color": "#112233", **overrides})


def test_template_lifecycle_default_and_unique_names(backend, uid):
    repo = backend.templates(uid)
    first = repo.add(fields(), make_default=True)
    second = repo.add(fields("Otra"))
    assert repo.get_default().id == first.id and {t.name for t in repo.list()} == {"Institucional", "Otra"}
    with pytest.raises(InvalidInputError):
        repo.add(fields("INSTITUCIONAL"))
    assert repo.set_default(second.id, True).is_default and repo.get_default().id == second.id
    renamed = repo.update(first.id, fields("Nueva", footer_text="", cover_text=""))
    assert renamed.name == "Nueva" and renamed.footer_text == ""
    repo.delete(second.id)
    assert [t.id for t in repo.list()] == [first.id] and repo.get_default().id == first.id  # the heir becomes default


def test_template_logo_bytes_round_trip(backend, uid):
    repo = backend.templates(uid)
    template = repo.add(fields())
    logo = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 50
    assert repo.get_logo(template.id) is None
    assert repo.set_logo(template.id, logo).has_logo and repo.get_logo(template.id) == logo
    assert not repo.set_logo(template.id, None).has_logo and repo.get_logo(template.id) is None


def test_templates_are_private(backend, two_users):
    ana, luis = two_users
    template = backend.templates(ana).add(fields())
    assert backend.templates(luis).list() == []
    backend.templates(luis).add(fields())  # the same name under another account is fine
    with pytest.raises(TemplateNotFoundError):
        backend.templates(luis).get(template.id)


# ------------------------------------------------------------------ notebooks
@pytest.fixture
def notebooks(backend, uid, tmp_path):
    return backend.notebooks(uid, tmp_path / "notebooks")


def code_cell(source):
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None}


def test_notebook_create_save_list_and_export(notebooks):
    created = notebooks.create("Mi análisis ñ", [code_cell("print('hola')")])
    assert created.name == "Mi análisis ñ" and created.cells[0]["source"] == "print('hola')" and created.revision
    saved = notebooks.save(
        created.id, [code_cell("x = 1"), {"cell_type": "markdown", "source": "# T"}], None, "Renombrado"
    )
    assert saved.name == "Renombrado" and len(saved.cells) == 2 and saved.revision != created.revision
    summary = notebooks.list()[0]
    assert (summary.id, summary.name, summary.n_cells) == (created.id, "Renombrado", 2)
    exported = json.loads(notebooks.export_bytes(created.id))
    assert exported["nbformat"] == 4 and "".join(exported["cells"][1]["source"]) == "# T"


def test_notebook_conflicts_use_the_base_revision(notebooks):
    created = notebooks.create("a")
    first = notebooks.save(created.id, [code_cell("1")], None, None, created.revision)
    with pytest.raises(ConflictError):
        notebooks.save(created.id, [code_cell("2")], None, None, created.revision)  # stale
    assert notebooks.save(created.id, [code_cell("3")], None, None, first.revision).cells[0]["source"] == "3"
    assert notebooks.save(created.id, [code_cell("4")], None, None).cells[0]["source"] == "4"  # unconditional


def test_notebook_import_delete_signature_and_missing(notebooks):
    before = notebooks.signature()
    document = notebooks.create("origen", [code_cell("a = 1")])
    imported = notebooks.import_bytes("copia.ipynb", notebooks.export_bytes(document.id))
    assert imported.id != document.id and imported.cells[0]["source"] == "a = 1"
    assert notebooks.signature() != before and len(notebooks.list()) == 2
    middle = notebooks.signature()
    notebooks.delete(document.id)
    assert notebooks.signature() != middle and [n.id for n in notebooks.list()] == [imported.id]
    for call in (lambda: notebooks.get(document.id), lambda: notebooks.delete(document.id), lambda: notebooks.get("x")):
        with pytest.raises(NotebookNotFoundError):
            call()
    with pytest.raises(ImportFailedError):
        notebooks.import_bytes("malo.ipynb", b"no es un notebook")


def test_notebooks_are_private(backend, two_users, tmp_path):
    ana, luis = two_users
    mine = backend.notebooks(ana, tmp_path / "nb").create("mío")
    theirs = backend.notebooks(luis, tmp_path / "nb")
    assert theirs.list() == []
    with pytest.raises(NotebookNotFoundError):
        theirs.get(mine.id)
    with pytest.raises(NotebookNotFoundError):
        theirs.delete(mine.id)


def test_concurrent_saves_with_the_same_base_revision_only_one_wins(backend, uid, tmp_path):
    if backend.settings.is_sqlite:
        pytest.skip("file notebooks serialise saves with a process lock; the row version is for server databases")
    store = backend.notebooks(uid, tmp_path / "nb")
    created = store.create("carrera")
    outcomes = []
    barrier = threading.Barrier(4)

    def worker(n):
        barrier.wait()
        try:
            store.save(created.id, [code_cell(str(n))], None, None, created.revision)
            outcomes.append("ok")
        except ConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ok") == 1 and outcomes.count("conflict") == 3
