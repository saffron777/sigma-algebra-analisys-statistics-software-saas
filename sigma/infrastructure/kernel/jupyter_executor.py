"""Code execution adapter: one real Jupyter kernel (ipykernel) per session, driven with ``jupyter_client``.

Each notebook gets its own kernel process, so a crash, ``exit()`` or infinite loop never affects the
web server. Cells run one at a time per notebook; a timeout interrupts the kernel (and restarts it if
it does not respond).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jupyter_client import BlockingKernelClient, KernelManager

from sigma.domain.ai import SECRET_KEY_ENV_VARS
from sigma.domain.errors import KernelError
from sigma.domain.notebooks import EXECUTION_ERROR, EXECUTION_OK, EXECUTION_TIMEOUT, ExecutionResult, KernelStatus
from sigma.infrastructure.kernel.bootstrap_code import BOOTSTRAP_CODE, R_BOOTSTRAP_CODE

log = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 2 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 90
INTERRUPT_GRACE_SECONDS = 8
POLL_SECONDS = 0.5
CULL_INTERVAL_SECONDS = 60


@dataclass
class _Session:
    manager: KernelManager
    client: BlockingKernelClient
    owner: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)
    busy: bool = False
    execution_count: int | None = None


class JupyterKernelPool:
    """Kernels keyed by session id; ``owner`` (the user id) scopes limits, the working folder and ``sigma.load``."""

    def __init__(
        self,
        project_root: Path,
        database_url: str,
        work_dir: Path,
        idle_minutes: float = 30,
        max_per_owner: int = 3,
        max_total: int = 8,
        rscript: Any = None,
    ) -> None:
        self._project_root = Path(project_root)
        self._database_url = database_url
        self._work_dir = Path(work_dir)
        self._rscript = rscript  # fallback executor for R notebooks when IRkernel/pbdZMQ is unavailable
        self._idle_seconds = idle_minutes * 60
        self._max_per_owner = max_per_owner
        self._max_total = max_total
        self._sessions: dict[str, _Session] = {}
        self._pool_lock = threading.Lock()
        self._culler: threading.Thread | None = None
        self._closing = threading.Event()
        atexit.register(self.shutdown_all)

    # ------------------------------------------------------------------ public API
    def start(self, session_id: str, owner: str = "", kernel_name: str = "python3") -> None:
        if kernel_name == "ir" and self._rscript is not None:
            return self._rscript.start(session_id, owner, kernel_name)
        self._ensure(session_id, owner, kernel_name)

    def execute(
        self, session_id: str, code: str, timeout_seconds: float, owner: str = "", kernel_name: str = "python3"
    ) -> ExecutionResult:
        if kernel_name == "ir" and self._rscript is not None:
            return self._rscript.execute(session_id, code, timeout_seconds, owner, kernel_name)
        session = self._ensure(session_id, owner, kernel_name)
        with session.lock:  # one cell at a time per notebook
            session.busy = True
            session.last_used = time.monotonic()
            started = time.monotonic()
            try:
                result = self._run(session, code, timeout_seconds)
            finally:
                session.busy = False
                session.last_used = time.monotonic()
            result.duration_ms = int((time.monotonic() - started) * 1000)
            session.execution_count = result.execution_count or session.execution_count
            if result.kernel_restarted or not session.manager.is_alive():
                self._discard(session_id, session)
            return result

    def interrupt(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session and session.manager.is_alive():
            session.manager.interrupt_kernel()

    def restart(self, session_id: str, owner: str = "", kernel_name: str = "python3") -> None:
        if kernel_name == "ir" and self._rscript is not None:
            return self._rscript.restart(session_id, owner, kernel_name)
        self.shutdown(session_id)
        self._ensure(session_id, owner, kernel_name)

    def shutdown(self, session_id: str) -> None:
        with self._pool_lock:
            session = self._sessions.pop(session_id, None)
        if session:
            self._stop(session)
        if self._rscript is not None:
            self._rscript.shutdown(session_id)

    def shutdown_owner(self, owner: str) -> None:
        """Stop every kernel that belongs to ``owner`` (used when the account is deleted)."""
        with self._pool_lock:
            ids = [sid for sid, s in self._sessions.items() if s.owner == owner]
            sessions = [self._sessions.pop(sid) for sid in ids]
        for session in sessions:
            self._stop(session)
        if self._rscript is not None:
            self._rscript.shutdown_owner(owner)

    def shutdown_all(self) -> None:
        self._closing.set()
        with self._pool_lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            self._stop(session)

    def status(self, session_id: str) -> KernelStatus:
        session = self._sessions.get(session_id)
        if not session or not session.manager.is_alive():
            return KernelStatus(running=False)
        return KernelStatus(running=True, busy=session.busy, execution_count=session.execution_count)

    # ------------------------------------------------------------------ kernel lifecycle
    def _ensure(self, session_id: str, owner: str = "", kernel_name: str = "python3") -> _Session:
        with self._pool_lock:
            session = self._sessions.get(session_id)
            if session and session.manager.is_alive():
                return session
            if session:
                self._stop(session)
                del self._sessions[session_id]
            self._make_room(owner)
            session = self._launch(owner, kernel_name)
            self._sessions[session_id] = session
            self._start_culler()
            return session

    def _make_room(self, owner: str) -> None:
        """Enforce the kernel limits (pool lock held): evict the owner's oldest idle kernel, or refuse."""
        mine = [(sid, s) for sid, s in self._sessions.items() if s.owner == owner]
        if len(mine) >= self._max_per_owner:
            idle = [(sid, s) for sid, s in mine if not s.busy]
            if not idle:
                raise KernelError(
                    f"Ya tiene {self._max_per_owner} kernels ejecutando código. Espere a que termine alguno."
                )
            sid, oldest = min(idle, key=lambda item: item[1].last_used)
            del self._sessions[sid]
            self._stop(oldest)  # its notebook gets a fresh kernel the next time it runs a cell
        if len(self._sessions) >= self._max_total:
            raise KernelError(
                "El servidor alcanzó el máximo de kernels simultáneos. Inténtelo de nuevo en unos minutos."
            )

    def _launch(self, owner: str = "", kernel_name: str = "python3") -> _Session:
        work_dir = self._work_dir / owner if owner else self._work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(p for p in (str(self._project_root), os.environ.get("PYTHONPATH", "")) if p),
            "SIGMA_DATABASE_URL": self._database_url,
            "SIGMA_USER_ID": owner,
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        }
        if kernel_name == "ir":
            env["RETICULATE_PYTHON"] = sys.executable  # the R kernel's reticulate uses the server's venv Python
            r_tmp = work_dir / ".r-tmp"
            r_tmp.mkdir(parents=True, exist_ok=True)
            # R must create its own temp dir; the sandbox/system TEMP may be read-only.
            env["TEMP"] = env["TMP"] = env["TMPDIR"] = str(r_tmp)
        for name in SECRET_KEY_ENV_VARS:  # code run in a kernel must never read the server's AI keys
            env.pop(name, None)
        manager = KernelManager(kernel_name=kernel_name)
        label = "R" if kernel_name == "ir" else "Python"
        try:
            manager.start_kernel(env=env, cwd=str(work_dir))
        except Exception as error:  # noqa: BLE001 - jupyter_client raises many types when the kernel cannot spawn
            raise KernelError(f"No se pudo iniciar el kernel de {label}: {error}") from error
        client = manager.client()
        client.start_channels()
        session = _Session(manager, client, owner)
        try:
            client.wait_for_ready(timeout=STARTUP_TIMEOUT_SECONDS)
            self._bootstrap(session, kernel_name)
        except Exception as error:
            self._stop(session)
            if isinstance(error, KernelError):
                raise
            raise KernelError(f"El kernel de {label} no respondió al iniciar: {error}") from error
        return session

    def _bootstrap(self, session: _Session, kernel_name: str = "python3") -> None:
        code = R_BOOTSTRAP_CODE if kernel_name == "ir" else BOOTSTRAP_CODE
        msg_id = session.client.execute(code, silent=True, store_history=False, allow_stdin=False)
        failure: str | None = None
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                msg = session.client.get_iopub_msg(timeout=POLL_SECONDS)
            except queue.Empty:
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "error":
                failure = f"{msg['content']['ename']}: {msg['content']['evalue']}"
            elif msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                break
        if failure:
            raise KernelError(f"El kernel no pudo preparar el entorno de Sigma Algebra ({failure}).")

    def _discard(self, session_id: str, session: _Session) -> None:
        with self._pool_lock:
            if self._sessions.get(session_id) is session:
                del self._sessions[session_id]
        self._stop(session)

    @staticmethod
    def _stop(session: _Session) -> None:
        try:
            session.client.stop_channels()
        except Exception:  # noqa: BLE001
            log.debug("stop_channels failed", exc_info=True)
        try:
            session.manager.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001
            log.debug("shutdown_kernel failed", exc_info=True)
        try:
            session.manager.cleanup_resources()
        except Exception:  # noqa: BLE001
            log.debug("cleanup_resources failed", exc_info=True)

    # ------------------------------------------------------------------ idle culling
    def _start_culler(self) -> None:
        if self._culler is None or not self._culler.is_alive():
            self._culler = threading.Thread(target=self._cull_loop, name="sigma-kernel-culler", daemon=True)
            self._culler.start()

    def _cull_loop(self) -> None:
        while not self._closing.wait(CULL_INTERVAL_SECONDS):
            now = time.monotonic()
            with self._pool_lock:
                stale = [
                    sid for sid, s in self._sessions.items() if not s.busy and now - s.last_used > self._idle_seconds
                ]
            for session_id in stale:
                log.info("Shutting down idle kernel %s", session_id)
                self.shutdown(session_id)

    # ------------------------------------------------------------------ running one cell
    def _run(self, session: _Session, code: str, timeout_seconds: float) -> ExecutionResult:
        client = session.client
        msg_id = client.execute(code, silent=False, store_history=True, allow_stdin=False, stop_on_error=True)
        collector = _OutputCollector()
        deadline = time.monotonic() + timeout_seconds
        status = EXECUTION_OK
        interrupted_at: float | None = None

        while True:
            now = time.monotonic()
            if interrupted_at is None and now > deadline:
                status = EXECUTION_TIMEOUT
                interrupted_at = now
                session.manager.interrupt_kernel()
            elif interrupted_at is not None and now - interrupted_at > INTERRUPT_GRACE_SECONDS:
                # the kernel ignored the interrupt: report it and let the caller discard (restart) it
                collector.add_stream("stderr", "\nEl kernel no respondió a la interrupción y será reiniciado.\n")
                return ExecutionResult(EXECUTION_TIMEOUT, collector.outputs, None, 0, kernel_restarted=True)
            try:
                msg = client.get_iopub_msg(timeout=POLL_SECONDS)
            except queue.Empty:
                if not session.manager.is_alive():
                    collector.add_stream(
                        "stderr",
                        "\nEl kernel se detuvo inesperadamente (¿memoria insuficiente o exit()?). "
                        "Se reiniciará en la próxima ejecución.\n",
                    )
                    return ExecutionResult(EXECUTION_ERROR, collector.outputs, None, 0, kernel_restarted=True)
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            if collector.consume(msg):
                break

        if status == EXECUTION_TIMEOUT:
            collector.add_stream(
                "stderr", f"\nTiempo máximo de ejecución excedido ({timeout_seconds:g} s): se interrumpió la celda.\n"
            )
        count, reply_status = self._shell_reply(client, msg_id)
        if status == EXECUTION_OK and (reply_status in ("error", "aborted") or collector.had_error):
            status = EXECUTION_ERROR
        return ExecutionResult(status, collector.outputs, count, 0)

    @staticmethod
    def _shell_reply(client: BlockingKernelClient, msg_id: str) -> tuple[int | None, str]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                reply = client.get_shell_msg(timeout=1)
            except queue.Empty:
                continue
            if reply["parent_header"].get("msg_id") == msg_id:
                content = reply["content"]
                return content.get("execution_count"), content.get("status", "ok")
        return None, "ok"


class _OutputCollector:
    """Turns iopub messages into nbformat outputs, merging stream chunks and capping the total size."""

    def __init__(self) -> None:
        self.outputs: list[dict[str, Any]] = []
        self.had_error = False
        self._size = 0
        self._truncated = False

    def consume(self, msg: dict[str, Any]) -> bool:
        """Handle one message; returns True once the kernel is idle again."""
        kind, content = msg["msg_type"], msg["content"]
        if kind == "status":
            return content["execution_state"] == "idle"
        if kind == "stream":
            self.add_stream(content["name"], content["text"])
        elif kind in ("display_data", "execute_result", "update_display_data"):
            output: dict[str, Any] = {
                "output_type": "execute_result" if kind == "execute_result" else "display_data",
                "data": content["data"],
                "metadata": content.get("metadata", {}),
            }
            if kind == "execute_result":
                output["execution_count"] = content.get("execution_count")
            self._append(output)
        elif kind == "error":
            self.had_error = True
            self._append(
                {
                    "output_type": "error",
                    "ename": content["ename"],
                    "evalue": content["evalue"],
                    "traceback": content.get("traceback", []),
                }
            )
        elif kind == "clear_output":
            self.outputs.clear()
            self._size = 0
        return False

    def add_stream(self, name: str, text: str) -> None:
        if self.outputs and self.outputs[-1]["output_type"] == "stream" and self.outputs[-1]["name"] == name:
            if self._room(len(text)):
                self.outputs[-1]["text"] += text
        else:
            self._append({"output_type": "stream", "name": name, "text": text})

    def _room(self, extra: int) -> bool:
        if self._size + extra > MAX_OUTPUT_BYTES:
            if not self._truncated:
                self._truncated = True
                self.outputs.append(
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": "\n[Salida truncada: se superó el límite de 2 MB]\n",
                    }
                )
            return False
        self._size += extra
        return True

    def _append(self, output: dict[str, Any]) -> None:
        if self._room(len(json.dumps(output, default=str))):
            self.outputs.append(output)
