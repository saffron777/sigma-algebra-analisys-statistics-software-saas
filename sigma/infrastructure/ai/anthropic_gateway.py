"""Anthropic adapter for the interpretation feature (official SDK, Messages API).

Opus 5.5 always thinks, so ``thinking`` is omitted and depth is set with ``output_config.effort``; sampling parameters
are not sent. The request opts into server-side refusal fallbacks (to Opus 4.8); if the API rejects that beta the call
is repeated without it and the choice is remembered.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import anthropic

from sigma.domain.errors import AiUnavailableError, RateLimitedError
from sigma.domain.interpretation import InterpretationOutput, Prompt
from sigma.infrastructure.ai.settings import AiSettings

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-06-01"
FALLBACK_MODEL = "claude-opus-4-8"
MAX_TOKENS = 8000
TIMEOUT_SECONDS = 150.0


def _default_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=2)


class AnthropicGateway:
    def __init__(self, settings: AiSettings, client_factory: Callable[[str], Any] | None = None) -> None:
        self._settings = settings
        self._client_factory = client_factory or _default_client
        self._fallbacks_supported = True

    # ------------------------------------------------------------------ port
    def interpret(self, prompt: Prompt) -> InterpretationOutput:
        return self._call(prompt.system, prompt.user, MAX_TOKENS)

    def test_connection(self) -> InterpretationOutput:
        return self._call("Responde únicamente con la palabra OK.", "Prueba de conexión.", 1500)

    # ------------------------------------------------------------------ call
    def _call(self, system: str, user: str, max_tokens: int) -> InterpretationOutput:
        key = self._settings.api_key
        if not key:
            raise AiUnavailableError("La IA no está configurada: falta la clave de API del servidor.")
        client = self._client_factory(key)
        model = self._settings.model
        arguments: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": "medium"},
        }
        with_fallback = self._fallbacks_supported and model != FALLBACK_MODEL
        try:
            response = self._create(client, arguments, with_fallback)
        except anthropic.BadRequestError as error:
            if with_fallback and "fallback" in _message(error).lower():
                log.warning("La API rechazó la opción de reserva; se reintenta sin ella.")
                self._fallbacks_supported = False
                response = self._create_or_raise(client, arguments)
            else:
                raise AiUnavailableError(f"El servicio de IA rechazó la solicitud: {_message(error)}") from None
        except anthropic.AnthropicError as error:
            raise _translate(error) from None
        return self._read(response, model)

    def _create_or_raise(self, client: Any, arguments: dict[str, Any]) -> Any:
        try:
            return self._create(client, arguments, False)
        except anthropic.BadRequestError as error:
            raise AiUnavailableError(f"El servicio de IA rechazó la solicitud: {_message(error)}") from None
        except anthropic.AnthropicError as error:
            raise _translate(error) from None

    @staticmethod
    def _create(client: Any, arguments: dict[str, Any], with_fallback: bool) -> Any:
        extra: dict[str, Any] = {}
        if with_fallback:
            extra = {"betas": [FALLBACK_BETA], "fallbacks": [{"model": FALLBACK_MODEL}]}
        return client.beta.messages.create(**arguments, **extra)

    @staticmethod
    def _read(response: Any, requested_model: str) -> InterpretationOutput:
        if response.stop_reason == "refusal":
            raise AiUnavailableError("El modelo declinó interpretar este contenido. Pruebe con otro resultado.")
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        if not text:
            raise AiUnavailableError("El servicio de IA no devolvió texto. Inténtelo de nuevo.")
        usage = getattr(response, "usage", None)
        fallback_used = any(getattr(block, "type", "") == "fallback" for block in response.content) or any(
            getattr(entry, "type", "") == "fallback_message" for entry in (getattr(usage, "iterations", None) or [])
        )
        return InterpretationOutput(
            text=text,
            model=getattr(response, "model", None) or requested_model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            fallback_used=fallback_used,
            truncated=response.stop_reason == "max_tokens",
        )


def _message(error: Exception) -> str:
    return str(getattr(error, "message", None) or error)[:300]


def _translate(error: anthropic.AnthropicError) -> Exception:
    """SDK errors → errors the web layer shows in Spanish (never anything that could contain the key)."""
    if isinstance(error, anthropic.AuthenticationError):
        return AiUnavailableError(
            "La clave de API de IA del servidor no es válida o fue revocada. Avise al administrador."
        )
    if isinstance(error, anthropic.PermissionDeniedError):
        return AiUnavailableError("La clave de API de IA no tiene permiso para este modelo. Avise al administrador.")
    if isinstance(error, anthropic.RateLimitError):
        return RateLimitedError(
            "El servicio de IA está saturado. Espere un momento y vuelva a intentarlo.", {"retry_after": 30}
        )
    if isinstance(error, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return AiUnavailableError(
            "No se pudo conectar con el servicio de IA. Revise la conexión a internet del servidor."
        )
    if isinstance(error, anthropic.APIStatusError) and error.status_code >= 500:
        return AiUnavailableError("El servicio de IA no está disponible en este momento. Inténtelo más tarde.")
    return AiUnavailableError("El servicio de IA devolvió un error inesperado.")
