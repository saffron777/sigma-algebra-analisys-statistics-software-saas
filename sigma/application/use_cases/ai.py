"""AI interpretation use cases (per user) and AI administration (administrator only)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sigma.application.ports import (
    AiConfiguration,
    AiGateway,
    AiUsageRepository,
    DatasetRepository,
)
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
        reason = None
        if not self._config.enabled:
            reason = "La IA está desactivada por el administrador."
        elif not self._config.configured:
            reason = "La IA no está configurada: el administrador debe añadir la clave de API."
        elif not self._user.can_use_ai:
            reason = "Su cuenta no tiene permiso para usar la IA."
        since = _iso(datetime.now(UTC) - timedelta(hours=1))
        used = self._usage.count_since(self._user.id, since) if reason is None else 0
        return {
            "available": reason is None,
            "reason": reason,
            "model": self._config.model,
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
        self._check_rate()
        prompt = build_prompt(module, text, title or str(report.get("title", "")), focus, dataset_name)
        output = self._gateway.interpret(prompt)
        self._usage.add(self._user.id, module, output.model, output.input_tokens, output.output_tokens)
        return {
            "text": output.text,
            "model": output.model,
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
            raise AiUnavailableError("La IA no está configurada: el administrador debe añadir la clave de API.")

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
    """What the administrator's AI page shows: settings (never the key) and usage per account."""

    def __init__(self, config: AiConfiguration, usage: AiUsageRepository, per_hour: int) -> None:
        self._config = config
        self._usage = usage
        self._per_hour = per_hour

    def execute(self) -> dict[str, Any]:
        since = _iso(datetime.now(UTC) - timedelta(days=30))
        return {**self._config.describe(), "requests_per_hour": self._per_hour, "usage": self._usage.summary(since)}


class UpdateAiSettings:
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


class TestAiConnection:
    """One tiny request, so the administrator can check the key and the model."""

    __test__ = False  # not a pytest class

    def __init__(self, gateway: AiGateway, config: AiConfiguration) -> None:
        self._gateway = gateway
        self._config = config

    def execute(self) -> dict[str, Any]:
        if not self._config.configured:
            raise AiUnavailableError("Primero guarde la clave de API.")
        output = self._gateway.test_connection()
        return {"ok": True, "model": output.model, "fallback_used": output.fallback_used, "reply": output.text[:80]}
