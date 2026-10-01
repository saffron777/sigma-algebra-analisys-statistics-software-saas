"""Console tests: a real ipykernel is started, so these are slower (a few seconds for the first kernel)."""

import io
import json
import time

import nbformat
import pandas as pd
import pytest

from sigma.config import PROJECT_ROOT
from sigma.domain.errors import ImportFailedError, InvalidInputError, NotebookNotFoundError
from sigma.infrastructure.kernel.jupyter_executor import JupyterKernelPool
from sigma.infrastructure.persistence.file_notebook_store import FileNotebookStore
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository
from tests.support import add_user


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("console")
    db_path = root / "sigma.db"
    database = Database(db_path)
    owner_id = add_user(database)
    assert str(owner_id) == OWNER
    repo = SqliteDatasetRepository(database, owner_id)
    repo.add("ventas", pd.DataFrame({"x": [1, 2, 3], "y": [2.0, 4.0, 6.5]}), "csv", {})
    pool = JupyterKernelPool(PROJECT_ROOT, db_path, root / "work", idle_minutes=30)
    yield pool, repo
    pool.shutdown_all()


OWNER = "1"  # the first account created in a fresh database


def run(pool, code, timeout=60, session="s1"):
    return pool.execute(session, code, timeout, OWNER)


def test_state_persists_between_cells_and_counts_executions(env):
    pool, _ = env
    first = run(pool, "x = 21")
    second = run(pool, "x * 2")
    assert first.status == "ok" and first.outputs == []
    assert second.outputs[0]["output_type"] == "execute_result"
    assert second.outputs[0]["data"]["text/plain"] == "42"
    assert second.execution_count == first.execution_count + 1


def test_stdout_and_stderr_are_captured_and_merged(env):
    pool, _ = env
    result = run(pool, "import sys\nprint('hola')\nprint('mundo')\nprint('aviso', file=sys.stderr)")
    streams = {o["name"]: o["text"] for o in result.outputs if o["output_type"] == "stream"}
    assert streams["stdout"] == "hola\nmundo\n" and streams["stderr"] == "aviso\n"


def test_errors_are_reported_with_traceback(env):
    pool, _ = env
    result = run(pool, "1 / 0")
    assert result.status == "error"
    error = result.outputs[-1]
    assert error["output_type"] == "error" and error["ename"] == "ZeroDivisionError" and error["traceback"]
    assert run(pool, "'sigue funcionando'").status == "ok"  # the kernel survives errors


def test_matplotlib_figures_come_back_as_png(env):
    pool, _ = env
    result = run(pool, "plt.plot([1, 2, 3], [3, 1, 2]); plt.show()")
    images = [o for o in result.outputs if "image/png" in o.get("data", {})]
    assert images and len(images[0]["data"]["image/png"]) > 500


def test_dataframes_come_back_as_html_tables(env):
    pool, _ = env
    result = run(pool, "pd.DataFrame({'a': [1, 2]})")
    assert "<table" in result.outputs[0]["data"]["text/html"]


def test_sigma_helper_loads_and_saves_datasets(env):
    pool, repo = env
    loaded = run(pool, "df = sigma.load('ventas'); int(df['x'].sum())")
    assert loaded.outputs[0]["data"]["text/plain"] == "6"
    run(pool, "df['z'] = df['x'] * 10\nsigma.save(df, 'ventas_z')")
    assert repo.get_by_name("ventas_z").data["z"].tolist() == [10, 20, 30]
    clash = run(pool, "sigma.save(df, 'ventas_z')")
    assert clash.status == "error" and "replace=True" in clash.outputs[-1]["evalue"]
    assert run(pool, "sigma.save(df, 'ventas_z', replace=True)").status == "ok"
    missing = run(pool, "sigma.load('nada')")
    assert "Datasets disponibles" in missing.outputs[-1]["evalue"]
    listing = run(pool, "sigma.datasets()['nombre'].tolist()")
    assert "ventas" in listing.outputs[0]["data"]["text/plain"]


def test_infinite_loop_is_interrupted_by_timeout_and_kernel_keeps_state(env):
    pool, _ = env
    run(pool, "keep = 'vivo'", session="loop")
    started = time.monotonic()
    result = run(pool, "while True:\n    pass", timeout=2, session="loop")
    assert result.status == "timeout" and time.monotonic() - started < 20
    assert any("Tiempo máximo" in o.get("text", "") for o in result.outputs)
    assert run(pool, "keep", session="loop").outputs[0]["data"]["text/plain"] == "'vivo'"


def test_endless_output_loop_still_times_out(env):
    pool, _ = env
    result = run(pool, "while True:\n    print('x' * 1000)", timeout=2, session="chatty")
    assert result.status == "timeout"
    assert sum(len(o.get("text", "")) for o in result.outputs) < 2.2 * 1024 * 1024  # output cap holds


def test_kernel_crash_is_reported_and_next_run_starts_a_fresh_kernel(env):
    pool, _ = env
    run(pool, "survivor = 1", session="crash")
    result = run(pool, "import os\nos._exit(1)", session="crash")
    assert result.status == "error" and result.kernel_restarted
    fresh = run(pool, "'survivor' in globals()", session="crash")
    assert fresh.outputs[0]["data"]["text/plain"] == "False"


def test_restart_clears_state_and_sessions_are_isolated(env):
    pool, _ = env
    run(pool, "a = 1", session="one")
    run(pool, "a = 2", session="two")
    pool.restart("one", OWNER)
    assert run(pool, "'a' in globals()", session="one").outputs[0]["data"]["text/plain"] == "False"
    assert run(pool, "a", session="two").outputs[0]["data"]["text/plain"] == "2"
    assert pool.status("two").running and not pool.status("never-started").running


def test_interrupt_from_another_thread(env):
    import threading

    pool, _ = env
    run(pool, "pass", session="irq")
    threading.Timer(1.5, pool.interrupt, args=("irq",)).start()
    result = run(pool, "import time\nwhile True:\n    time.sleep(0.02)", timeout=60, session="irq")
    assert result.status == "error" and result.outputs[-1]["ename"] == "KeyboardInterrupt"


def test_uninterruptible_call_ends_with_a_kernel_restart(env):
    """On Windows a long time.sleep() ignores the interrupt; after the grace period the kernel is restarted."""
    pool, _ = env
    run(pool, "marker = 1", session="stuck")
    started = time.monotonic()
    result = run(pool, "import time\ntime.sleep(120)", timeout=2, session="stuck")
    assert result.status == "timeout" and time.monotonic() - started < 30
    if result.kernel_restarted:
        assert run(pool, "'marker' in globals()", session="stuck").outputs[0]["data"]["text/plain"] == "False"


# ------------------------------------------------------------------------------ notebook files
@pytest.fixture
def store(tmp_path):
    return FileNotebookStore(tmp_path / "nbs")


def make_ipynb(cells=None) -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells or [nbformat.v4.new_markdown_cell("# Título"), nbformat.v4.new_code_cell("print(1)")]
    notebook.metadata["title"] = "Mi análisis"
    return nbformat.writes(notebook).encode("utf-8")


def test_create_save_reload_roundtrip(store):
    doc = store.create("Primero")
    assert doc.name == "Primero" and len(doc.cells) == 1 and doc.cells[0]["cell_type"] == "code"
    cells = [
        {"cell_type": "markdown", "source": ["# Hola\n", "mundo"], "id": "abc12345", "ui_only": True},
        {
            "cell_type": "code",
            "source": "print('x')",
            "outputs": [{"output_type": "stream", "name": "stdout", "text": "x\n"}],
            "execution_count": 3,
            "running": True,
        },
    ]
    saved = store.save(doc.id, cells, None, "Renombrado")
    assert saved.name == "Renombrado"
    reloaded = store.get(doc.id)
    assert reloaded.cells[0]["source"] == "# Hola\nmundo" and "ui_only" not in reloaded.cells[0]
    assert reloaded.cells[1]["execution_count"] == 3 and "running" not in reloaded.cells[1]
    assert all(c.get("id") for c in reloaded.cells) and len({c["id"] for c in reloaded.cells}) == 2
    assert [s.id for s in store.list()] == [doc.id]


def test_saved_files_are_valid_notebooks(store):
    doc = store.create("Valido")
    store.save(doc.id, [{"cell_type": "code", "source": "1+1"}], None, None)
    notebook = nbformat.reads(store.export_bytes(doc.id).decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells[0].source == "1+1"


def test_import_ipynb_and_name_resolution(store):
    doc = store.import_bytes("analisis_ventas.ipynb", make_ipynb())
    assert doc.name == "Mi análisis" and len(doc.cells) == 2 and all(c["id"] for c in doc.cells)
    nameless = nbformat.v4.new_notebook()
    named = store.import_bytes("otro_archivo.ipynb", nbformat.writes(nameless).encode())
    assert named.name == "otro_archivo"


def test_import_rejects_garbage(store):
    for content in (b"not json", b'{"cells": "x"}', b"", b"\xff\xfe\x00"):
        with pytest.raises(ImportFailedError):
            store.import_bytes("x.ipynb", content)


def test_old_nbformat_versions_are_upgraded(store):
    legacy = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {},
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["print(1)"], "outputs": [], "execution_count": None}
        ],
    }
    doc = store.import_bytes("legacy.ipynb", json.dumps(legacy).encode())
    assert doc.cells[0]["source"] == "print(1)" and doc.cells[0]["id"]


def test_invalid_ids_and_missing_notebooks(store):
    for bad in ("../../etc/passwd", "x", "0" * 31, ""):
        with pytest.raises(NotebookNotFoundError):
            store.get(bad)
    with pytest.raises(NotebookNotFoundError):
        store.get("0" * 32)
    with pytest.raises(NotebookNotFoundError):
        store.delete("0" * 32)
    doc = store.create("x")
    with pytest.raises(InvalidInputError):
        store.save(doc.id, [{"cell_type": "weird", "source": ""}], None, None)


# ------------------------------------------------------------------------------ HTTP API
def test_notebook_api_flow(api):
    created = api.post("/api/notebooks", {"name": "Demo"})
    assert created.status_code == 201
    nb = created.parsed["notebook"]
    listing = api.get("/api/notebooks").parsed["notebooks"]
    assert [n["id"] for n in listing] == [nb["id"]]

    kernel = api.post(f"/api/notebooks/{nb['id']}/kernel/start").parsed["kernel"]
    assert kernel["running"] is True
    run = api.post(f"/api/notebooks/{nb['id']}/execute", {"source": "print('desde la API')\n40 + 2"})
    assert run.status_code == 200 and run.parsed["status"] == "ok"
    assert run.parsed["outputs"][0]["text"] == "desde la API\n"
    assert run.parsed["outputs"][1]["data"]["text/plain"] == "42"

    cells = nb["cells"]
    cells[0].update(
        source="print('desde la API')", outputs=run.parsed["outputs"], execution_count=run.parsed["execution_count"]
    )
    assert api.put(f"/api/notebooks/{nb['id']}", {"cells": cells}).status_code == 200
    reopened = api.get(f"/api/notebooks/{nb['id']}").parsed["notebook"]
    assert reopened["cells"][0]["outputs"][0]["text"] == "desde la API\n"

    assert api.patch(f"/api/notebooks/{nb['id']}", {"name": "Renombrado"}).parsed["notebook"]["name"] == "Renombrado"
    export = api.get(f"/api/notebooks/{nb['id']}/export")
    assert "Renombrado.ipynb" in export.headers["Content-Disposition"]
    nbformat.validate(nbformat.reads(export.get_data(as_text=True), as_version=4))

    assert api.post(f"/api/notebooks/{nb['id']}/kernel/restart").parsed["restarted"] is True
    assert api.delete(f"/api/notebooks/{nb['id']}").status_code == 200
    assert api.get(f"/api/notebooks/{nb['id']}").status_code == 404
    assert api.get(f"/api/notebooks/{nb['id']}/kernel").status_code == 404


def test_notebook_api_import_and_validation(api):
    imported = api.post(
        "/api/notebooks/import",
        data={"file": (io.BytesIO(make_ipynb()), "demo.ipynb")},
        content_type="multipart/form-data",
    )
    assert imported.status_code == 201 and imported.parsed["notebook"]["name"] == "Mi análisis"
    bad = api.post(
        "/api/notebooks/import", data={"file": (io.BytesIO(b"nope"), "x.ipynb")}, content_type="multipart/form-data"
    )
    assert bad.status_code == 422
    assert api.post("/api/notebooks/import", data={}, content_type="multipart/form-data").status_code == 422
    nb_id = imported.parsed["notebook"]["id"]
    assert api.post(f"/api/notebooks/{nb_id}/execute", {"source": "1", "timeout": 0}).status_code == 422
    assert api.post("/api/notebooks/nope/execute", {"source": "1"}).status_code == 404
    api.delete(f"/api/notebooks/{nb_id}")


def test_execute_requires_csrf(app):
    response = app.test_client().post("/api/notebooks/" + "0" * 32 + "/execute", json={"source": "1"})
    assert response.status_code == 403
