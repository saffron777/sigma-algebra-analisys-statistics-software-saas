"""Tests for the R console infrastructure (language routing and the Rscript fallback executor)."""

import shutil
from pathlib import Path

import pytest

from sigma.application.use_cases.notebooks import kernel_name_for
from sigma.infrastructure.kernel.rscript_executor import RSCRIPT, RscriptExecutor


def test_language_maps_to_kernel_name():
    assert kernel_name_for("python") == "python3"
    assert kernel_name_for("r") == "ir"


def test_rscript_script_bridges_sigma_and_persists_workspace():
    ex = RscriptExecutor(Path("E:/Sigma"), "sqlite:///:memory:", Path("E:/Sigma/_nb"))
    script = ex._script("u1:nb1", "x <- 1")
    assert 'reticulate::import("sigma.infrastructure.kernel.helpers")' in script
    assert "x <- 1" in script
    assert 'save(list = setdiff(ls(), "sigma")' in script
    # session id is sanitised into the workspace file name (':' -> '_')
    assert "u1_nb1.RData" in script


def test_rscript_restart_clears_the_workspace():
    ex = RscriptExecutor(Path("E:/Sigma"), "sqlite:///:memory:", Path("E:/Sigma/_nb"))
    ws = ex._workspace("u1:nb1")
    ws.parent.mkdir(parents=True, exist_ok=True)
    ws.write_text("x", encoding="utf-8")
    ex.restart("u1:nb1", owner="u1", kernel_name="ir")
    assert not ws.exists()


def test_rscript_executor_status_is_running():
    ex = RscriptExecutor(Path("E:/Sigma"), "sqlite:///:memory:", Path("E:/Sigma/_nb"))
    assert ex.status("s").running is True


@pytest.mark.skipif(not Path(RSCRIPT).is_file(), reason="R no está instalado")
def test_rscript_executes_r_with_persistence():
    work = Path("E:/Sigma/.test-r-console")
    shutil.rmtree(work, ignore_errors=True)
    try:
        ex = RscriptExecutor(Path("E:/Sigma"), "sqlite:///:memory:", work)
        ex.execute("s1", "x <- 40\ny <- 2", 90, "u1", "ir")
        res = ex.execute("s1", "cat(x + y, '\\n')", 90, "u1", "ir")
        assert res.status == "ok"
        assert any("42" in o.get("text", "") for o in res.outputs)
        # sigma bridge (reticulate) is available in the R kernel
        res2 = ex.execute("s1", "cat('m', length(names(sigma)), '\\n')", 90, "u1", "ir")
        assert res2.status == "ok"
    finally:
        shutil.rmtree(work, ignore_errors=True)
