"""Heavy analyses run in a separate process with a time limit; light ones run here."""

import time

import pandas as pd
import pytest

import tests.analysis_fixtures  # noqa: F401 - registers the internal test analyses in this process
from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.infrastructure.compute.process_runner import AnalysisRunnerImpl, InlineRunner, ProcessRunner


@pytest.fixture(autouse=True)
def extra_analyses(monkeypatch):
    """The worker process is a fresh interpreter: it must find the test analyses too."""
    monkeypatch.setenv("SIGMA_EXTRA_ANALYSIS_MODULES", "tests.analysis_fixtures")


def run(runner, analysis_id, frame=None, roles=None, params=None, module="series"):
    return runner.run(module, analysis_id, frame if frame is not None else pd.DataFrame(), roles or {}, params or {})


def test_a_light_analysis_runs_in_this_process():
    out = run(
        InlineRunner(),
        "acf_pacf",
        pd.DataFrame({"s": range(40)}).astype(float),
        {"series": "s"},
        {"lags": 5, "alpha": 0.05, "pacf_method": "ywm"},
    )
    assert out.result["n"] == 40


def test_unexpected_exceptions_become_clear_computation_errors():
    with pytest.raises(ComputationError, match="KeyError"):
        run(InlineRunner(), "_boom")


def test_a_heavy_analysis_returns_its_output_from_the_other_process():
    runner = ProcessRunner(timeout_seconds=60)
    out = run(runner, "_warns", pd.DataFrame({"a": [1, 2, 3]}))
    assert out.title == "Con advertencias"
    alert = out.blocks[0]
    assert alert["type"] == "alert" and "optimizador no convergió" in alert["message"]
    assert "divide by zero" not in alert["message"]  # numerical noise is not worth a warning to the user


def test_errors_cross_the_process_boundary_with_their_type():
    runner = ProcessRunner(timeout_seconds=60)
    with pytest.raises(InvalidInputError, match="imposibles"):
        run(runner, "_invalid")
    with pytest.raises(ComputationError, match="KeyError"):
        run(runner, "_boom")


def test_a_heavy_analysis_that_overruns_is_killed():
    runner = ProcessRunner(timeout_seconds=2)
    started = time.monotonic()
    with pytest.raises(ComputationError, match="tiempo máximo"):
        run(runner, "_sleep", params={"seconds": 60})
    assert time.monotonic() - started < 25  # it did not wait for the 60 s


def test_the_runner_picks_the_process_only_for_heavy_analyses():
    runner = AnalysisRunnerImpl(timeout_seconds=60)
    assert run(runner, "_warns").blocks[0]["type"] == "alert"  # heavy: warnings are collected in the worker
    light = run(
        runner,
        "t_power",
        module="potencia",
        params={
            "design": "one_sample",
            "solve_for": "n",
            "effect_size": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "alternative": "two-sided",
            "ratio": 1.0,
        },
    )
    assert light.result["n"] > 20


def test_when_every_slot_is_busy_the_wait_ends_with_a_clear_message():
    runner = ProcessRunner(timeout_seconds=30, max_parallel=1, wait_seconds=1)
    import threading

    holder = threading.Thread(target=lambda: run(runner, "_sleep", params={"seconds": 4}))
    holder.start()
    time.sleep(1.5)  # the first one is running now
    with pytest.raises(ComputationError, match="ocupado"):
        run(runner, "_sleep", params={"seconds": 0.1})
    holder.join()
