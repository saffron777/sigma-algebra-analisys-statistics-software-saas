"""R code execution via ``Rscript`` (fallback when IRkernel/pbdZMQ is unavailable on the machine).

Each notebook has a persistent R workspace (``.RData``): a cell loads it, runs, and saves it back, so variables
defined in one cell are available in the next. ``sigma`` is re-bridged through reticulate on every cell (reticulate
objects are not R-serialisable, so they are excluded from the saved workspace).

Two environment details matter for running R headlessly from Python on Windows:

- R needs to create its own temp directory (``R_TempDir``); the sandbox/system ``TEMP`` may be read-only, so we point
  ``TEMP``/``TMP``/``TMPDIR`` at a writable folder.
- R's ``-e`` only evaluates the *first* line of a multi-line argument, so the whole cell is written to a ``.R`` script
  file and run with ``Rscript <file>`` instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sigma.domain.errors import KernelError
from sigma.domain.notebooks import EXECUTION_ERROR, EXECUTION_OK, EXECUTION_TIMEOUT, ExecutionResult, KernelStatus

RSCRIPT = os.environ.get("SIGMA_RSCRIPT", "C:/Program Files/R/R-4.5.3/bin/Rscript.exe")


class RscriptExecutor:
    """Runs R code per cell with a persistent ``.RData`` workspace per session."""

    def __init__(self, project_root: Path, database_url: str, work_dir: Path) -> None:
        self._project_root = Path(project_root)
        self._database_url = database_url
        self._work_dir = Path(work_dir)
        self._python = sys.executable

    @staticmethod
    def _safe(session_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)

    def _workspace(self, session_id: str) -> Path:
        return self._work_dir / f"{self._safe(session_id)}.RData"

    def _script(self, session_id: str, code: str) -> str:
        ws = str(self._workspace(session_id)).replace("\\", "/")
        return (
            f'if (file.exists("{ws}")) load("{ws}")\n'
            'library(reticulate)\n'
            'sigma <- reticulate::import("sigma.infrastructure.kernel.helpers")$sigma\n'
            f"{code}\n"
            f'save(list = setdiff(ls(), "sigma"), file = "{ws}")\n'
        )

    def start(self, session_id: str, owner: str = "", kernel_name: str = "python3") -> None:
        return None

    def execute(self, session_id: str, code: str, timeout_seconds: float, owner: str = "", kernel_name: str = "python3") -> ExecutionResult:
        if not Path(RSCRIPT).is_file():
            raise KernelError("No se encontró Rscript: instale R o defina SIGMA_RSCRIPT.")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        r_tmp = self._work_dir / ".r-tmp"
        r_tmp.mkdir(parents=True, exist_ok=True)
        script_file = self._work_dir / f"{self._safe(session_id)}.R"
        script_file.write_text(self._script(session_id, code), encoding="utf-8")
        env = {
            **os.environ,
            "RETICULATE_PYTHON": self._python,
            "PYTHONPATH": str(self._project_root),
            "SIGMA_DATABASE_URL": self._database_url,
            "SIGMA_USER_ID": owner,
            # R must be able to create its own temp dir; the sandbox/system TEMP may be read-only.
            "TEMP": str(r_tmp),
            "TMP": str(r_tmp),
            "TMPDIR": str(r_tmp),
        }
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [RSCRIPT, str(script_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                env=env,
                cwd=str(self._work_dir),
            )
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - started) * 1000)
            return ExecutionResult(
                EXECUTION_TIMEOUT,
                [{"output_type": "stream", "name": "stderr", "text": "\n[Se superó el tiempo límite.]\n"}],
                None,
                duration,
            )
        outputs: list[dict[str, Any]] = []
        if proc.stdout:
            outputs.append({"output_type": "stream", "name": "stdout", "text": proc.stdout})
        if proc.stderr:
            outputs.append({"output_type": "stream", "name": "stderr", "text": proc.stderr})
        status = EXECUTION_OK if proc.returncode == 0 else EXECUTION_ERROR
        duration = int((time.monotonic() - started) * 1000)
        return ExecutionResult(status, outputs, None, duration)

    def interrupt(self, session_id: str) -> None:
        return None

    def restart(self, session_id: str, owner: str = "", kernel_name: str = "python3") -> None:
        self.shutdown(session_id)

    def shutdown(self, session_id: str) -> None:
        ws = self._workspace(session_id)
        ws.unlink(missing_ok=True)
        script_file = self._work_dir / f"{self._safe(session_id)}.R"
        script_file.unlink(missing_ok=True)

    def shutdown_owner(self, owner: str) -> None:
        prefix = self._safe(owner)
        for path in self._work_dir.glob(f"{prefix}_*"):
            if path.suffix in (".RData", ".R"):
                path.unlink(missing_ok=True)

    def status(self, session_id: str) -> KernelStatus:
        return KernelStatus(running=True)
