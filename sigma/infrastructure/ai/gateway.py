"""The AI gateway: picks the configured model, builds its adapter and answers.

One place decides *which* model answers (the one asked for, else the server default) and one place keeps the
adapters, so the rest of Sigma only knows the ``AiGateway`` port.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from sigma.domain.ai import AiModel
from sigma.domain.errors import AiUnavailableError
from sigma.domain.interpretation import InterpretationOutput, Prompt
from sigma.infrastructure.ai.adapters import build_adapter
from sigma.infrastructure.ai.http import JsonClient, RequestsJsonClient
from sigma.infrastructure.ai.settings import AiSettings


class ConfiguredGateway:
    def __init__(self, settings: AiSettings, client: JsonClient | None = None) -> None:
        self._settings = settings
        self._client = client or RequestsJsonClient()
        self._adapters: dict[tuple, Any] = {}

    # ------------------------------------------------------------------ configuration
    @property
    def configuration(self) -> AiSettings:
        return self._settings

    def models(self) -> list[dict[str, Any]]:
        """The models the *users* may ask for: enabled and with a key."""
        return [model.describe() for model in self._settings.usable_models()]

    def default_model(self) -> AiModel | None:
        return self._settings.default_model()

    def default_model_id(self) -> str | None:
        model = self._settings.default_model()
        return model.id if model else None

    def describe(self) -> dict[str, Any]:
        return self._settings.describe()

    # ------------------------------------------------------------------ the port
    def interpret(self, prompt: Prompt, model_id: str | None = None) -> InterpretationOutput:
        return self._adapter(model_id).interpret(prompt)

    def test_connection(self, model_id: str | None = None) -> InterpretationOutput:
        return self._adapter(model_id).test_connection()

    # ------------------------------------------------------------------ internals
    def _adapter(self, model_id: str | None):
        model = self._settings.resolve(model_id)
        if model is None:
            raise AiUnavailableError(
                "La IA no está configurada: añada al menos un modelo con su clave de API en "
                "Usuarios → Interpretación con IA."
            )
        signature = (
            model.id,
            model.provider,
            model.model,
            model.base_url,
            sha256((model.stored_key or "").encode("utf-8")).hexdigest()[:12],
        )
        adapter = self._adapters.get(signature)
        if adapter is None:
            adapter = build_adapter(model, self._client)
            self._adapters[signature] = adapter
        return adapter
