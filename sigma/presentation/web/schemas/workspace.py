"""Request model for the workspace endpoints."""

from __future__ import annotations

from typing import Any

from sigma.presentation.web.schemas.base import RequestModel


class SaveWorkspaceRequest(RequestModel):
    """Partial update: only the fields present in the body are changed (``null`` clears one)."""

    state: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
