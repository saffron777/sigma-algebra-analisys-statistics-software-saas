"""AI interpretation use cases (per user) and AI administration (administrator only)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sigma.application.ports import (
    AiConfiguration,
    AiGateway,
    AiUsageRepository,
    DatasetRepository,
)
from sigma.domain.ai import providers_for_ui
from sigma.domain.dataset_report import dataset_section
from sigma.domain.errors import AiUnavailableError, InvalidInputError, PermissionDeniedError, RateLimitedError
from sigma.domain.interpretation import MODULES, build_prompt, report_to_text
from sigma.domain.users import User

MAX_CODE_CHARS = 6000
MAX_LISTED_VALUES = 20  # a column with more distinct values than this is probably free text or an identifier


def _lists_raw_values(block: dict[str, Any], many_values: set[str]) -> bool:
    """The ficha's «first rows» table and the frequency tables of identifier-like columns show raw values."""
    title = str(block.get("title") or "")
    if title.startswith("Primeras"):
        return True
    return any(title == f"Frecuencias de «{name}»" for name in many_values)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


class GetAiStatus:
    """Whether the signed-in user can ask for interpretations right now, and why not when they cannot."""

    def __init__(self, config: AiConfiguration, usage: AiUsageRepository, user: User, per_hour: int) -> None:
        self._config = config
        self._usage = usage
        self._user = user
        self._per_hour = per_hour

    def execute(self) -> dict[str, Any]:
        described = self._config.describe()
        models = [model for model in described.get("models", []) if model.get("available")]
        reason = None
        if not self._config.enabled:
            reason = "La IA está desactivada por el administrador."
        elif not models:
            reason = "La IA no está configurada: el administrador debe añadir al menos un modelo con su clave de API."
        elif not self._user.can_use_ai:
            reason = "Su cuenta no tiene permiso para usar la IA."
        since = _iso(datetime.now(UTC) - timedelta(hours=1))
        used = self._usage.count_since(self._user.id, since) if reason is None else 0
        preferred = next((m for m in models if m["id"] == described.get("default_server_id")), None)
        preferred = preferred or (models[0] if models else {})
        return {
            "available": reason is None,
            "reason": reason,
            "model": preferred.get("model", self._config.model),
            "model_label": preferred.get("label", ""),
            "provider": preferred.get("provider", ""),
            "server_id": preferred.get("id"),
            "models": [
                {"id": m["id"], "label": m["label"], "model": m["model"], "provider": m["provider"]}
                for m in models
            ],
            "requests_this_hour": used,
            "requests_per_hour": self._per_hour,
        }


class InterpretResult:
    """Sends the *report* of a result (never the data) to the model and returns its explanation."""

    def __init__(
        self,
        gateway: AiGateway,
        config: AiConfiguration,
        usage: AiUsageRepository,
        datasets: DatasetRepository,
        user: User,
        per_hour: int,
    ) -> None:
        self._gateway = gateway
        self._config = config
        self._usage = usage
        self._datasets = datasets
        self._user = user
        self._per_hour = per_hour

    def execute(
        self,
        module: str,
        report: dict[str, Any] | None = None,
        title: str = "",
        focus: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_allowed()
        if module not in MODULES:
            raise InvalidInputError(f"Módulo inválido «{module}».")
        if module == "dataset":
            if not dataset_id:
                raise InvalidInputError("Indique el dataset que desea interpretar.")
            dataset = self._datasets.get(dataset_id)  # a foreign id is a 404, like everywhere else
            section = dataset_section(dataset.info, dataset.data)
            # the model gets the summary (types, gaps, statistics), not rows of the data
            many_values = {c.name for c in dataset.info.columns if c.n_unique > MAX_LISTED_VALUES}
            blocks = [b for b in section.blocks if not _lists_raw_values(b, many_values)]
            report = {"title": section.title, "subtitle": section.subtitle, "blocks": blocks}
            title, dataset_name = f"Ficha del dataset {dataset.info.name}", dataset.info.name
        if not isinstance(report, dict) or not report.get("blocks"):
            raise InvalidInputError("No hay ningún resultado que interpretar.")
        text = report_to_text(report)
        self._check_model(model_id)
        self._check_rate()
        prompt = build_prompt(module, text, title or str(report.get("title", "")), focus, dataset_name)
        output = self._gateway.interpret(prompt, model_id)
        self._usage.add(
            self._user.id, module, output.model, output.input_tokens, output.output_tokens, output.provider
        )
        return {
            "text": output.text,
            "model": output.model,
            "provider": output.provider,
            "model_label": output.server_label,
            "created_at": _iso(datetime.now(UTC)),
            "fallback_used": output.fallback_used,
            "truncated": output.truncated,
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
        }

    def _check_allowed(self) -> None:
        if not self._user.can_use_ai:
            raise PermissionDeniedError("Su cuenta no tiene permiso para usar la IA.")
        if not self._config.enabled:
            raise AiUnavailableError("La IA está desactivada por el administrador.")
        if not self._config.configured:
            raise AiUnavailableError(
                "La IA no está configurada: añada al menos un modelo con su clave de API en "
                "Usuarios → Interpretación con IA."
            )

    def _check_model(self, model_id: str | None) -> None:
        """An explicitly chosen model must exist and be enabled; a wrong id never reaches a provider."""
        if not model_id:
            return
        chosen = next((item for item in self._config.describe().get("models", []) if item["id"] == model_id), None)
        if chosen is None:
            raise InvalidInputError(f"El modelo «{model_id}» no está configurado.")
        if not chosen.get("enabled"):
            raise InvalidInputError(f"El modelo «{chosen.get('label') or model_id}» está desactivado.")

    def _check_rate(self) -> None:
        since = datetime.now(UTC) - timedelta(hours=1)
        if self._usage.count_since(self._user.id, _iso(since)) >= self._per_hour:
            oldest = self._usage.oldest_since(self._user.id, _iso(since))
            wait = 60
            if oldest:
                wait = max(
                    1, int((datetime.fromisoformat(oldest) + timedelta(hours=1) - datetime.now(UTC)).total_seconds())
                )
            raise RateLimitedError(
                f"Alcanzó el límite de {self._per_hour} interpretaciones por hora. Inténtelo de nuevo más tarde.",
                {"retry_after": wait},
            )


class GetAiAdmin:
    """What the administrator's AI page shows: settings (never a key) and usage per account."""

    def __init__(self, config: AiConfiguration, usage: AiUsageRepository, per_hour: int) -> None:
        self._config = config
        self._usage = usage
        self._per_hour = per_hour

    def execute(self) -> dict[str, Any]:
        since = _iso(datetime.now(UTC) - timedelta(days=30))
        return {**self._config.describe(), "requests_per_hour": self._per_hour, "usage": self._usage.summary(since)}


class UpdateAiSettings:
    """The switch, and the shortcut to the Claude entry that older versions of the page used to edit."""

    def __init__(self, config: AiConfiguration) -> None:
        self._config = config

    def execute(
        self, key: str | None = None, clear_key: bool = False, model: str | None = None, enabled: bool | None = None
    ) -> dict[str, Any]:
        if key:
            self._config.set_key(key)
        elif clear_key:
            self._config.clear_key()
        if model:
            self._config.set_model(model)
        if enabled is not None:
            self._config.set_enabled(enabled)
        return self._config.describe()


class AiServers:
    """The administration of the configured models: add, edit, remove, test, and the provider catalogue.

    Every change returns the whole administration view (settings, catalogue and usage), which is exactly what the
    page needs to redraw itself. The gateway is read from the container on each call, so a test or a future
    replacement never goes stale.
    """

    def __init__(self, owner: Any, config: Any, view: GetAiAdmin) -> None:
        self._owner = owner  # the Container: holds the gateway
        self._config = config
        self._view = view

    def _gateway(self) -> AiGateway:
        return self._owner.ai_gateway

    # ------------------------------------------------------------------ views
    def list_models(self) -> dict[str, Any]:
        return {**self._view.execute(), "models": self._config.describe()["models"], "providers": providers_for_ui()}

    def providers(self) -> list[dict[str, Any]]:
        return providers_for_ui()

    # ------------------------------------------------------------------ changes
    def add_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        entry = self._config.add_model(
            str(payload.get("provider") or ""),
            str(payload.get("model") or ""),
            str(payload.get("label") or ""),
            str(payload.get("base_url") or ""),
            str(payload.get("key") or ""),
            True if payload.get("enabled") is None else bool(payload["enabled"]),
        )
        return {**self.list_models(), "created": entry.id}

    def update_model(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._config.update_model(model_id, payload)
        return self.list_models()

    def remove_model(self, model_id: str) -> dict[str, Any]:
        self._config.remove_model(model_id)
        return {**self.list_models(), "deleted": model_id}

    def test_model(self, model_id: str | None) -> dict[str, Any]:
        model = self._config.get(model_id) if model_id else self._config.default_model()
        if model is None:
            raise AiUnavailableError("No hay ningún modelo configurado que probar.")
        if model.missing_key:
            raise AiUnavailableError(f"El modelo «{model.label}» todavía no tiene clave de API.")
        output = self._gateway().test_connection(model.id)
        return {
            "ok": True,
            "model": output.model,
            "provider": output.provider,
            "server_label": output.server_label,
            "fallback_used": output.fallback_used,
            "reply": output.text[:80],
        }


class TestAiConnection:
    """One tiny request, so the administrator can check a model's key and address.

    ``owner`` is the container: the gateway is read on each call, so replacing it in a test always takes effect.
    """

    __test__ = False  # not a pytest class

    def __init__(self, owner: Any, config: AiConfiguration) -> None:
        self._owner = owner
        self._config = config

    def execute(self, model_id: str | None = None) -> dict[str, Any]:
        if not self._config.configured:
            raise AiUnavailableError("Primero guarde la clave de API.")
        output = self._owner.ai_gateway.test_connection(model_id)
        return {
            "ok": True,
            "model": output.model,
            "provider": output.provider,
            "server_label": output.server_label,
            "fallback_used": output.fallback_used,
            "reply": output.text[:80],
        }
