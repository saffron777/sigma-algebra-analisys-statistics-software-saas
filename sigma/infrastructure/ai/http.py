"""The HTTP transport of the AI adapters: one POST that returns the status and the decoded body.

Using ``requests`` (already a dependency of Sigma) keeps every provider on the same, inspectable path and adds no
new package to install. The client is injectable, so the adapters are tested without touching the network.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import requests

from sigma.domain.errors import AiUnavailableError

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 150.0
RETRIES = 2
RETRY_STATUSES = (429, 500, 502, 503, 504)


class JsonClient:
    """A minimal POST-JSON client (the port the adapters depend on)."""

    def post(self, url: str, headers: Mapping[str, str], body: Any, timeout: float = TIMEOUT_SECONDS) -> tuple[int, Any]:
        """Return ``(status, payload)``; a payload that is not JSON comes back as ``None``."""
        raise NotImplementedError


class RequestsJsonClient(JsonClient):
    def __init__(self, session: requests.Session | None = None, retries: int = RETRIES) -> None:
        self._session = session or requests.Session()
        self._retries = retries

    def post(
        self, url: str, headers: Mapping[str, str], body: Any, timeout: float = TIMEOUT_SECONDS
    ) -> tuple[int, Any]:
        attempt = 0
        while True:
            try:
                response = self._session.post(url, headers=dict(headers), json=body, timeout=timeout)
            except requests.Timeout:
                raise AiUnavailableError(
                    "El proveedor de IA tardó demasiado en responder. Inténtelo de nuevo o pruebe otro modelo."
                ) from None
            except requests.RequestException:
                raise AiUnavailableError(
                    "No se pudo conectar con el proveedor de IA. Revise la dirección y la conexión del servidor."
                ) from None
            if response.status_code in RETRY_STATUSES and attempt < self._retries:
                attempt += 1
                log.info("El proveedor de IA respondió %s; reintento %s.", response.status_code, attempt)
                continue
            return response.status_code, _decode(response)


def _decode(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:  # includes json.JSONDecodeError: an HTML error page, a plain-text message…
        return response.text[:600] if response.text else None
