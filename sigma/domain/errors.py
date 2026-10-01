"""Domain errors. The web layer maps them to HTTP responses; the domain knows nothing about HTTP."""

from __future__ import annotations

from typing import Any


class SigmaError(Exception):
    """Base class for every error raised on purpose by Sigma."""

    code = "sigma_error"

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidInputError(SigmaError):
    """The user supplied parameters or data that cannot be analysed."""

    code = "invalid_input"


class ComputationError(SigmaError):
    """The computation is valid in principle but failed (singular matrix, no variance...)."""

    code = "computation_error"


class DatasetNotFoundError(SigmaError):
    code = "dataset_not_found"


class ResultNotFoundError(SigmaError):
    code = "result_not_found"


class NotebookNotFoundError(SigmaError):
    code = "notebook_not_found"


class ImportFailedError(SigmaError):
    """A data source could not be read."""

    code = "import_failed"


class KernelError(SigmaError):
    """The code execution kernel failed or is unavailable."""

    code = "kernel_error"
