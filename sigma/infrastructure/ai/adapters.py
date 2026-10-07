"""The provider adapters: one class per HTTP family, all behind the same tiny interface.

Every adapter receives an already-resolved ``AiModel`` (its key included) and returns an ``InterpretationOutput``.
The request and response *shapes* live in ``providers.py`` as pure functions; here live the transport, the retries
and each family's quirks.
"""

from __future__ import annotations

from typing import Any, Protocol

from sigma.domain.ai import ANTHROPIC_KIND, GEMINI_KIND, OPENAI_KIND
from sigma.domain.ai import AiModel
from sigma.domain.errors import AiUnavailableError
from sigma.domain.interpretation import InterpretationOutput, Prompt
from sigma.infrastructure.ai.http import JsonClient
from sigma.infrastructure.ai.providers import (
    Answer,
    anthropic_prepared,
    auth_headers,
    error_detail,
    error_for_status,
    gemini_prepared,
    max_tokens_for,
    openai_prepared,
    parse_for,
    test_prompt,
)


class ProviderAdapter(Protocol):
    """One configured model, ready to answer."""

    def interpret(self, prompt: Prompt) -> InterpretationOutput: ...

    def test_connection(self) -> InterpretationOutput: ...


class _BaseAdapter:
    kind = ""

    def __init__(self, model: AiModel, client: JsonClient) -> None:
        self._model = model
        self._client = client

    # ------------------------------------------------------------------ helpers
    def _key(self) -> str:
        key = self._model.stored_key or ""
        if not key and self._model.spec.key_env:
            import os

            key = (os.environ.get(self._model.spec.key_env) or "").strip()
        return key

    def _guard(self) -> None:
        if self._model.spec.key_required and not self._key():
            raise AiUnavailableError(
                f"El modelo «{self._model.label}» no tiene clave de API. "
                "Añádala en Usuarios → Interpretación con IA."
            )

    def _send(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, Any]:
        return self._client.post(url, {**headers, **auth_headers(self.kind, self._key())}, body)

    def _answer(self, status: int, payload: Any, requested_model: str) -> Answer:
        if status >= 400:
            raise error_for_status(status, payload, self._model.spec.label, self._key())
        if not isinstance(payload, dict):
            raise AiUnavailableError(f"{self._model.spec.label} devolvió una respuesta que no se pudo leer.")
        return parse_for(self.kind, payload, requested_model)

    def _output(self, answer: Answer) -> InterpretationOutput:
        if answer.blocked and not answer.text:
            raise AiUnavailableError(
                f"El modelo «{self._model.label}» declinó interpretar este contenido. Pruebe con otro resultado."
            )
        if not answer.text:
            raise AiUnavailableError("El servicio de IA no devolvió texto. Inténtelo de nuevo.")
        return InterpretationOutput(
            text=answer.text,
            model=answer.model or self._model.model,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            fallback_used=answer.fallback_used,
            truncated=answer.truncated,
            provider=self._model.provider,
            server_label=self._model.label,
        )

    def _complete(self, prompt: Prompt, max_tokens: int) -> Answer:
        raise NotImplementedError

    # ------------------------------------------------------------------ interface
    def interpret(self, prompt: Prompt) -> InterpretationOutput:
        self._guard()
        return self._output(self._complete(prompt, max_tokens_for(False)))

    def test_connection(self) -> InterpretationOutput:
        self._guard()
        return self._output(self._complete(test_prompt(), max_tokens_for(True)))


class OpenAICompatibleAdapter(_BaseAdapter):
    """``/chat/completions`` with a bearer token: OpenAI, Azure, DeepSeek, Groq, Mistral, xAI, Ollama…"""

    kind = OPENAI_KIND

    def _complete(self, prompt: Prompt, max_tokens: int) -> Answer:
        prepared = openai_prepared(self._model.model, self._model.base_url, prompt, max_tokens)
        return self._answer(*self._send(prepared.url, prepared.headers, prepared.body), prepared.model)


class GeminiAdapter(_BaseAdapter):
    """Google Generative Language: the model goes in the path and the key in ``x-goog-api-key``."""

    kind = GEMINI_KIND

    def _complete(self, prompt: Prompt, max_tokens: int) -> Answer:
        prepared = gemini_prepared(self._model.model, self._model.base_url, prompt, max_tokens)
        return self._answer(*self._send(prepared.url, prepared.headers, prepared.body), prepared.model)


class AnthropicAdapter(_BaseAdapter):
    """Messages API: Opus always thinks, and the request opts into server-side fallbacks to Opus 4.8.

    If the API rejects the fallback option the call is repeated without it and the choice is remembered for the
    lifetime of this adapter, so a deployment that has not enabled the beta pays the surprise only once.
    """

    kind = ANTHROPIC_KIND

    def __init__(self, model: AiModel, client: JsonClient) -> None:
        super().__init__(model, client)
        self._fallbacks_supported = True

    def _complete(self, prompt: Prompt, max_tokens: int) -> Answer:
        prepared = anthropic_prepared(
            self._model.model, self._model.base_url, prompt, max_tokens, self._fallbacks_supported
        )
        status, payload = self._send(prepared.url, prepared.headers, prepared.body)
        if status >= 400 and self._fallbacks_supported and "fallback" in error_detail(payload).lower():
            self._fallbacks_supported = False
            prepared = anthropic_prepared(self._model.model, self._model.base_url, prompt, max_tokens, False)
            status, payload = self._send(prepared.url, prepared.headers, prepared.body)
        return self._answer(status, payload, prepared.model)


def build_adapter(model: AiModel, client: JsonClient) -> ProviderAdapter:
    """The adapter that understands this model's provider family."""
    kind = model.spec.kind
    if kind == ANTHROPIC_KIND:
        return AnthropicAdapter(model, client)
    if kind == GEMINI_KIND:
        return GeminiAdapter(model, client)
    if kind == OPENAI_KIND:
        return OpenAICompatibleAdapter(model, client)
    raise AiUnavailableError(f"Tipo de servicio de IA no soportado: «{kind}».")
