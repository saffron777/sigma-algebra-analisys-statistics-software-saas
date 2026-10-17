"""Domain errors. The web layer maps them to HTTP responses; the domain knows nothing about HTTP."""

from __future__ import annotations

from typing import Any


class SigmaError(Exception):
    """Base class for every error raised on purpose by Sigma Algebra."""

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


class AuthenticationError(SigmaError):
    """Missing, expired or wrong credentials (the message never says which one)."""

    code = "unauthenticated"


class PermissionDeniedError(SigmaError):
    """The signed-in user is not allowed to do this."""

    code = "forbidden"


class RateLimitedError(SigmaError):
    """Too many attempts or requests; ``details`` carries ``retry_after`` seconds."""

    code = "rate_limited"


class UserNotFoundError(SigmaError):
    code = "user_not_found"


class SetupRequiredError(SigmaError):
    """No account exists yet: the first administrator must be created."""

    code = "setup_required"


class ConflictError(SigmaError):
    """The resource changed since the client last saw it (optimistic concurrency)."""

    code = "conflict"


class TemplateNotFoundError(SigmaError):
    code = "template_not_found"


class ProjectNotFoundError(SigmaError):
    code = "project_not_found"


class AiUnavailableError(SigmaError):
    """The AI service cannot be used right now (disabled, no key, network or service problem)."""

    code = "ai_unavailable"


class MigrationError(SigmaError):
    """Copying the data to another database failed (the destination is left empty)."""

    code = "migration_failed"


class MaintenanceError(SigmaError):
    """Sigma Algebra is busy with an operation that needs exclusive use of the data (e.g. a database migration)."""

    code = "maintenance"
