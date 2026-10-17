"""Copying a whole Sigma Algebra database to another engine and back: every record arrives intact, ids included."""

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sigma.application.ports import StagedImport
from sigma.application.use_cases.migration import MigrateData
from sigma.domain.database import MIGRATION_ENTITIES
from sigma.domain.errors import InvalidInputError, MigrationError
from sigma.domain.report_templates import validate_fields
from sigma.infrastructure.persistence.backend import create_backend
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql.import_draft_store import CHUNK_BYTES
from tests.conformance.test_accounts_and_datasets import mixed_frame

LOGO = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 20


def populate(backend, root: Path) -> None:
    """Two accounts with something of every kind, created through the real repositories."""
    users = backend.users()
    ana = users.add("ana", "Ana Ñ", "hash-ana", "admin", True, True).id
    luis = users.add("luis", "Luis", "hash-luis", "user", False, False).id
    users.set_password(luis, "hash-luis-2")  # auth_epoch 2: sessions must stay valid after the move
    users.touch_login(ana)
    backend.settings_store().set("ai_model", "claude-opus-5-5")

    templates = backend.templates(ana)
    template = templates.add(validate_fields("Institucional", organization="Ñandú S.A.", primary_color="#112233"), True)
    templates.set_logo(template.id, LOGO)
    backend.templates(luis).add(validate_fields("Mía"))

    frames = backend.datasets(ana)
    sales = frames.add("ventas", mixed_frame(), "csv", {"file_name": "v.csv"})
    backend.datasets(ana).add("grande", pd.DataFrame({"a": np.arange(7000), "t": "x"}), "excel", {})
    backend.datasets(luis).add("Ventas", pd.DataFrame({"z": [1.5, None]}), "csv", {})

    results = backend.results(ana)
    for title, dataset_id in (("Regresión", sales.id), ("Sin datos", None)):
        results.add(
            _result(title, dataset_id, sales.name if dataset_id else None),
        )
    backend.workspace(ana).save("regression", state={"y": "ventas"}, payload={"r2": 0.9})
    backend.workspace(luis).save("anova", state={"f": 1})

    drafts = backend.drafts(ana)
    drafts.attach_file("csv", "pendiente.csv", np.random.default_rng(3).bytes(CHUNK_BYTES + 77))
    drafts.put(
        StagedImport(pd.DataFrame({"x": [1]}), "sqlserver", {"source_type": "sqlserver"}, "q", {}),
        None,
        {"server": "s"},
    )
    usage = backend.ai_usage()
    usage.add(ana, "regression", "modelo", 100, 50)
    usage.add(luis, "anova", "modelo", 7, 3)

    backend.notebooks(ana, root).create(
        "Mi análisis ñ",
        [{"cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": [], "execution_count": None}],
    )
    backend.notebooks(luis, root).create("De Luis")


def _result(title, dataset_id, dataset_name):
    from sigma.domain.saved_results import SavedResult

    return SavedResult(
        id=uuid.uuid4().hex,
        module="regression",
        analysis_type="ols_multiple",
        title=title,
        notes="",
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        created_at="2026-10-01T10:00:00+00:00",
        updated_at="2026-10-01T10:00:00+00:00",
        params={"y": "ventas"},
        result={"r2": 0.91},
        report={"title": title, "blocks": []},
        figures=[{"data": [], "layout": {}}],
    )


def snapshot(backend, root: Path) -> tuple[dict, dict]:
    """Everything observable through the repositories: plain data to compare, and the dataset frames."""
    plain: dict = {"settings": backend.settings_store().get("ai_model")}
    frames: dict = {}
    users = backend.users().list()
    plain["users"] = sorted(
        (
            u.id,
            u.username,
            u.display_name,
            u.role,
            u.can_use_console,
            u.can_use_ai,
            u.is_active,
            u.auth_epoch,
            u.created_at,
            u.last_login_at,
        )
        for u in users
    )
    for user in users:
        uid = user.id
        datasets = backend.datasets(uid)
        infos = sorted(datasets.list(), key=lambda d: d.id)
        plain[f"datasets:{uid}"] = [
            (d.id, d.name, d.source_type, d.n_rows, d.n_cols, d.created_at, d.source_meta) for d in infos
        ]
        for info in infos:
            frames[(uid, info.id)] = datasets.get(info.id).data
        results = backend.results(uid)
        listed = sorted(results.list(None, None, 100, 0)[0], key=lambda r: r.id)
        plain[f"results:{uid}"] = [
            (r.id, r.title, r.dataset_id, r.dataset_name, results.get(r.id).result, results.get(r.id).figures)
            for r in listed
        ]
        workspace = backend.workspace(uid)
        plain[f"workspace:{uid}"] = {
            m: (workspace.get(m).state, workspace.get(m).payload, e) for m, e in workspace.revisions().items()
        }
        plain[f"sync:{uid}"] = backend.sync(uid).revisions()
        drafts = backend.drafts(uid)
        plain[f"drafts:{uid}"] = sorted(
            (d.id, d.source_type, d.file_name, d.file_size, json.dumps(d.options, sort_keys=True), drafts.file(d.id)[1])
            for d in drafts.list()
        )
        templates = backend.templates(uid)
        plain[f"templates:{uid}"] = sorted(
            (
                t.id,
                t.name,
                t.organization,
                t.primary_color,
                t.is_default,
                t.has_logo,
                templates.get_logo(t.id),
                t.created_at,
            )
            for t in templates.list()
        )
        notebooks = backend.notebooks(uid, root)
        plain[f"notebooks:{uid}"] = sorted(
            (n.id, n.name, n.n_cells, json.dumps(notebooks.get(n.id).cells, sort_keys=True)) for n in notebooks.list()
        )
    plain["usage"] = sorted(
        (row["user_id"], row["requests"], row["input_tokens"], row["output_tokens"])
        for row in backend.ai_usage().summary("0")
    )
    return plain, frames


def assert_same(left: tuple[dict, dict], right: tuple[dict, dict]) -> None:
    assert left[0] == right[0]
    assert left[1].keys() == right[1].keys()
    for key, frame in left[1].items():
        pd.testing.assert_frame_equal(frame, right[1][key])


@pytest.fixture
def sqlite_peer(tmp_path):
    peer = create_backend(DatabaseSettings.sqlite(tmp_path / "peer.db"))
    yield peer
    peer.close()


def test_sqlite_to_the_engine_and_back(backend, sqlite_peer, tmp_path):
    source_root, target_root, back_root = tmp_path / "nb_src", tmp_path / "nb_dst", tmp_path / "nb_back"
    populate(sqlite_peer, source_root)
    expected = snapshot(sqlite_peer, source_root)
    assert expected[0]["users"] and any(expected[0][k] for k in expected[0] if k.startswith("notebooks:"))

    seen: list[str] = []
    report = MigrateData().execute(
        sqlite_peer.transfer(source_root),
        backend.transfer(target_root),
        lambda key, label, done, total: seen.append(key) if done == total else None,
    )
    assert [key for key, _ in MIGRATION_ENTITIES] == list(dict.fromkeys(seen))
    assert report.copied["users"] == 2 and report.copied["datasets"] == 3 and report.copied["notebooks"] == 2
    assert_same(expected, snapshot(backend, target_root))
    highest = max(u.id for u in backend.users().list())
    assert backend.users().add("nuevo", "N", "x", "user", False, False).id > highest  # the id generator moved on

    # and back into an empty SQLite file: still identical
    fresh = create_backend(DatabaseSettings.sqlite(tmp_path / "back.db"))
    try:
        backend.users().delete(backend.users().find_with_hash("nuevo")[0].id)
        MigrateData().execute(backend.transfer(target_root), fresh.transfer(back_root))
        assert_same(expected, snapshot(fresh, back_root))
    finally:
        fresh.close()


def test_a_target_with_data_is_refused_untouched(backend, sqlite_peer, tmp_path):
    populate(sqlite_peer, tmp_path / "nb_src")
    backend.users().add("ocupado", "O", "x", "user", False, False)
    before = backend.transfer(tmp_path / "nb_dst").counts()
    with pytest.raises(InvalidInputError):
        MigrateData().execute(sqlite_peer.transfer(tmp_path / "nb_src"), backend.transfer(tmp_path / "nb_dst"))
    assert backend.transfer(tmp_path / "nb_dst").counts() == before


def test_a_failed_copy_leaves_the_target_empty_and_can_be_retried(backend, sqlite_peer, tmp_path):
    root = tmp_path / "nb_src"
    populate(sqlite_peer, root)
    real = sqlite_peer.transfer(root)

    class Flaky:
        def counts(self):
            return real.counts()

        def read(self, entity):
            if entity == "saved_results":
                raise RuntimeError("el origen se cayó")
            return real.read(entity)

    with pytest.raises(MigrationError):
        MigrateData().execute(Flaky(), backend.transfer(tmp_path / "nb_dst"))
    target = backend.transfer(tmp_path / "nb_dst")
    assert target.is_empty()
    MigrateData().execute(real, target)
    assert_same(snapshot(sqlite_peer, root), snapshot(backend, tmp_path / "nb_dst"))
