"""Saved results, workspace, sync revisions and server-wide settings on every engine."""

import threading
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from sigma.application.dto import SaveResultCommand
from sigma.application.use_cases.results import DeleteResult, GetResult, ListResults, SaveResult, UpdateResult
from sigma.domain.errors import InvalidInputError, ResultNotFoundError
from sigma.domain.workspace import UNSET


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


@pytest.fixture
def results(backend, uid):
    return backend.results(uid)


@pytest.fixture
def save(backend, uid, results):
    return SaveResult(results, backend.datasets(uid))


# ------------------------------------------------------------------ saved results
def test_save_get_and_export_round_trip(save, results, backend, uid):
    info = backend.datasets(uid).add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    saved = save.execute(command(dataset_id=info.id))
    assert saved.dataset_name == "ventas"
    fetched = GetResult(results).execute(saved.id)
    assert fetched.result == {"r2": 0.91} and fetched.figures == [{"data": [], "layout": {}}]
    assert fetched.title == "Regresión ventas" and fetched.params == {"y": "ventas"}


def test_empty_notes_and_large_payloads(save, results):
    saved = save.execute(command(notes="", report={"blob": "ñ" * 1_000_000}))
    fetched = GetResult(results).execute(saved.id)
    assert fetched.notes == "" and len(fetched.report["blob"]) == 1_000_000


def test_list_filters_search_and_paging(save, results):
    save.execute(command(title="Regresión ventas"))
    save.execute(command(module="anova", title="ANOVA región", notes="100% real_"))
    save.execute(command(module="anova", title="Otro ANOVA"))
    lister = ListResults(results)
    assert lister.execute()[1] == 3
    assert [i.module for i in lister.execute(module="anova")[0]] == ["anova", "anova"]
    assert lister.execute(search="reg")[1] == 2
    assert lister.execute(search="100%")[1] == 1 and lister.execute(search="%")[1] == 1  # wildcards are escaped
    assert lister.execute(search="real_")[1] == 1 and lister.execute(search="_")[1] == 1
    assert len(lister.execute(limit=2)[0]) == 2 and lister.execute(limit=2, offset=2)[1] == 3
    assert len(lister.execute(limit=2, offset=2)[0]) == 1
    with pytest.raises(InvalidInputError):
        lister.execute(module="nope")
    assert lister.execute(search="'; DROP TABLE saved_results; --")[1] == 0
    assert lister.execute()[1] == 3


def test_update_delete_and_isolation(save, results, backend, two_users):
    saved = save.execute(command())
    updated = UpdateResult(results).execute(saved.id, "Nuevo título", "nota")
    assert (updated.title, updated.notes) == ("Nuevo título", "nota")
    with pytest.raises(InvalidInputError):
        UpdateResult(results).execute(saved.id, "   ", None)
    ana, luis = two_users
    foreign = backend.results(luis)
    own = backend.results(ana)
    own_saved = SaveResult(own, backend.datasets(ana)).execute(command())
    with pytest.raises(ResultNotFoundError):
        GetResult(foreign).execute(own_saved.id)
    DeleteResult(results).execute(saved.id)
    with pytest.raises(ResultNotFoundError):
        GetResult(results).execute(saved.id)


def test_dataset_deletion_keeps_results_with_a_snapshot_of_its_name(save, results, backend, uid):
    datasets = backend.datasets(uid)
    info = datasets.add("ventas", pd.DataFrame({"a": [1]}), "csv", {})
    saved = save.execute(command(dataset_id=info.id))
    datasets.rename(info.id, "ventas 2024")
    assert GetResult(results).execute(saved.id).dataset_name == "ventas 2024"
    datasets.delete(info.id)
    kept = GetResult(results).execute(saved.id)
    assert kept.dataset_id is None and kept.dataset_name == "ventas 2024"


def test_interpretation_can_be_set_and_cleared(results, save):
    saved = save.execute(command())
    results.update(saved.id, None, None, {"text": "hola ñ"})
    assert results.get(saved.id).interpretation == {"text": "hola ñ"}
    results.update(saved.id, None, None, None)
    assert results.get(saved.id).interpretation is None


# ------------------------------------------------------------------ workspace
def test_workspace_partial_saves_and_growing_revisions(backend, uid):
    workspace = backend.workspace(uid)
    assert workspace.get("regression").revision == 0
    first = workspace.save("regression", state={"y": "a"})
    second = workspace.save("regression", payload={"r2": 0.5})
    assert (first.revision, second.revision) == (1, 2)
    assert second.state == {"y": "a"} and second.payload == {"r2": 0.5}  # the untouched field is kept
    third = workspace.save("regression", state=None)
    assert third.state is None and third.payload == {"r2": 0.5} and third.revision == 3
    cleared = workspace.clear("regression")
    assert cleared == 4 and workspace.get("regression").state is None and workspace.revisions() == {"regression": 4}
    assert workspace.save("regression", state=UNSET).revision == 5


def test_workspace_is_private_and_clear_all(backend, two_users):
    ana, luis = two_users
    backend.workspace(ana).save("regression", state={"a": 1})
    backend.workspace(ana).save("anova", state={"a": 2})
    assert backend.workspace(luis).revisions() == {}
    assert set(backend.workspace(ana).clear_all()) == {"regression", "anova"}
    assert backend.workspace(ana).get("anova").state is None


def test_concurrent_workspace_saves_never_lose_a_revision(backend, uid):
    workspace = backend.workspace(uid)
    workspace.save("regression", state={"n": 0})
    errors = []

    def worker():
        try:
            for i in range(6):
                workspace.save("regression", state={"n": i})
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and workspace.get("regression").revision == 1 + 36


def test_first_writes_racing_on_a_new_row_both_succeed(backend, uid):
    workspace = backend.workspace(uid)
    errors = []
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait()
            workspace.save("anova", state={"x": 1})
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and workspace.get("anova").revision == 4


# ------------------------------------------------------------------ sync revisions
def test_sync_revisions_grow_with_every_change(backend, uid):
    datasets, sync = backend.datasets(uid), backend.sync(uid)
    assert sync.revisions() == {}
    info = datasets.add("uno", pd.DataFrame({"x": [1]}), "csv", {})
    after_add = sync.revisions()["datasets"]
    datasets.rename(info.id, "dos")
    after_rename = sync.revisions()["datasets"]
    datasets.delete(info.id)
    assert 0 < after_add < after_rename < sync.revisions()["datasets"]


def test_concurrent_dataset_changes_count_every_revision(backend, uid):
    frame = pd.DataFrame({"x": [1]})
    errors = []

    def worker(n):
        try:
            backend.datasets(uid).add(f"d{n}", frame, "csv", {})
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and backend.sync(uid).revisions()["datasets"] == 5


# ------------------------------------------------------------------ server settings and AI usage
def test_settings_are_upserted(backend):
    settings = backend.settings_store()
    assert settings.get("ai_model") is None and settings.get("ai_model", "x") == "x"
    settings.set("ai_model", "uno")
    settings.set("ai_model", "dos ñ")
    assert settings.get("ai_model") == "dos ñ"


def test_ai_usage_counts_and_summaries(backend, two_users):
    ana, luis = two_users
    usage = backend.ai_usage()
    now = datetime.now(UTC)
    since = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    usage.add(ana, "regression", "modelo", 100, 50)
    usage.add(ana, "anova", "modelo", 10, 5)
    usage.add(luis, "anova", "modelo", 1, 1)
    assert usage.count_since(ana, since) == 2 and usage.count_since(luis, since) == 1
    assert usage.count_since(ana, (now + timedelta(hours=1)).isoformat()) == 0
    assert usage.oldest_since(ana, since) is not None and usage.oldest_since(ana, "9999") is None
    rows = {row["user_id"]: row for row in usage.summary(since)}
    assert rows[ana]["requests"] == 2 and rows[ana]["input_tokens"] == 110 and rows[luis]["output_tokens"] == 1
