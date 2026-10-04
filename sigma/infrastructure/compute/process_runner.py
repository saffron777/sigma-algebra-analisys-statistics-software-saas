"""Runs catalogue analyses: light ones in this process, heavy ones in a separate process with a time limit.

A heavy analysis (model grids, forests, bootstrap…) must not hold a web thread for minutes nor be unstoppable, so it
runs in its own process: if it exceeds the limit the process is terminated and the user gets a clear message. At most
``max_parallel`` heavy analyses run at once; the rest wait for a slot for a while.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import warnings
from typing import Any

import pandas as pd

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import AnalysisOutput
from sigma.domain.analysis.spec import AnalysisSpec
from sigma.domain.errors import ComputationError, SigmaError

log = logging.getLogger(__name__)
_KNOWN_ERRORS = {cls.__name__: cls for cls in SigmaError.__subclasses__()}
_NOISE = ("divide by zero", "invalid value encountered", "overflow encountered", "Mean of empty slice")
_MAX_WARNINGS = 3


def _short(error: BaseException) -> str:
    text = " ".join(str(error).split())[:240]
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _compute(spec: AnalysisSpec, frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]) -> AnalysisOutput:
    """Run the analysis; a failure of the numerical code becomes a clear ``ComputationError``."""
    try:
        return spec.compute(frame, roles, params)
    except SigmaError:
        raise
    except Exception as error:  # noqa: BLE001 - statsmodels, sklearn, arch… each raise their own types
        log.warning("analysis %s/%s failed", spec.module, spec.id, exc_info=True)
        raise ComputationError(f"No se pudo completar el cálculo ({_short(error)}).") from error


class InlineRunner:
    def run(
        self, module: str, analysis_id: str, frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]
    ) -> AnalysisOutput:
        return _compute(registry.get(module, analysis_id), frame, roles, params)


# ------------------------------------------------------------------ the separate process
def _child_main(conn: Any, module: str, analysis_id: str, frame: Any, roles: Any, params: Any) -> None:
    """Entry point of the worker process: computes and sends back the output (or the error)."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = _compute(registry.get(module, analysis_id), frame, roles, params)
        seen: list[str] = []
        for item in caught:
            message = " ".join(str(item.message).split())[:200]
            if message and not any(noise in message for noise in _NOISE) and message not in seen:
                seen.append(message)
        conn.send(("ok", output, seen[:_MAX_WARNINGS]))
    except SigmaError as error:
        conn.send(("sigma", type(error).__name__, error.message, error.details))
    except BaseException as error:  # noqa: BLE001 - nothing may escape without telling the parent
        conn.send(("error", _short(error)))
    finally:
        conn.close()


class ProcessRunner:
    def __init__(self, timeout_seconds: int = 120, max_parallel: int = 2, wait_seconds: int = 30) -> None:
        self._timeout = timeout_seconds
        self._wait = wait_seconds
        self._slots = threading.BoundedSemaphore(max_parallel)

    def run(
        self, module: str, analysis_id: str, frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]
    ) -> AnalysisOutput:
        if not self._slots.acquire(timeout=self._wait):
            raise ComputationError(
                "El servidor está ocupado con otros análisis pesados. Inténtelo de nuevo en un momento."
            )
        try:
            return self._run_in_process(module, analysis_id, frame, roles, params)
        finally:
            self._slots.release()

    def _run_in_process(
        self, module: str, analysis_id: str, frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]
    ) -> AnalysisOutput:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_child_main, args=(sender, module, analysis_id, frame, roles, params), daemon=True
        )
        process.start()
        sender.close()
        try:
            if not receiver.poll(self._timeout):
                raise ComputationError(
                    f"El análisis superó el tiempo máximo de {self._timeout} s y se canceló. "
                    "Use menos filas o columnas, o parámetros más simples."
                )
            try:
                message = receiver.recv()
            except EOFError:
                raise ComputationError("El cálculo terminó de forma inesperada (¿falta de memoria?).") from None
        finally:
            if process.is_alive():
                process.terminate()
            process.join(5)
            receiver.close()
        return self._unpack(message)

    @staticmethod
    def _unpack(message: tuple) -> AnalysisOutput:
        kind = message[0]
        if kind == "ok":
            output: AnalysisOutput = message[1]
            if message[2]:
                output.blocks.insert(
                    0,
                    {
                        "type": "alert",
                        "kind": "warning",
                        "message": "Advertencias del cálculo: " + " · ".join(message[2]),
                    },
                )
            return output
        if kind == "sigma":
            raise _KNOWN_ERRORS.get(message[1], ComputationError)(message[2], message[3])
        raise ComputationError(f"No se pudo completar el cálculo ({message[1]}).")


class AnalysisRunnerImpl:
    """Light analyses here, ``heavy`` ones in a separate process with a time limit."""

    def __init__(self, timeout_seconds: int = 120, max_parallel: int = 2) -> None:
        self._inline = InlineRunner()
        self._process = ProcessRunner(timeout_seconds, max_parallel)

    def run(
        self, module: str, analysis_id: str, frame: pd.DataFrame, roles: dict[str, Any], params: dict[str, Any]
    ) -> AnalysisOutput:
        spec = registry.get(module, analysis_id)
        runner = self._process if spec.heavy else self._inline
        return runner.run(module, analysis_id, frame, roles, params)
