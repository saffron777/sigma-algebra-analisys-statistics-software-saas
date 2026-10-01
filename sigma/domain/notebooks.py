"""Notebook and code-execution entities (cells use the nbformat v4 dict layout, so they stay Jupyter-compatible)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NOTEBOOK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

EXECUTION_OK = "ok"
EXECUTION_ERROR = "error"
EXECUTION_TIMEOUT = "timeout"


@dataclass
class NotebookSummary:
    id: str
    name: str
    n_cells: int
    created_at: str
    updated_at: str
    revision: str = ""


@dataclass
class NotebookDocument:
    id: str
    name: str
    created_at: str
    updated_at: str
    cells: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: str = ""  # changes whenever the file is written; lets a second window detect a conflicting save


@dataclass
class ExecutionResult:
    """Outcome of running one cell: nbformat-style outputs plus how the run ended."""

    status: str
    outputs: list[dict[str, Any]]
    execution_count: int | None
    duration_ms: int
    kernel_restarted: bool = False


@dataclass
class KernelStatus:
    running: bool
    busy: bool = False
    execution_count: int | None = None
