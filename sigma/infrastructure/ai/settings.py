"""The administrator's AI settings: the configured models, which one is the default, and the global on/off switch.

Compatibility matters here: an installation that used a single Claude model (``ANTHROPIC_API_KEY`` or
``instance/ai_key.txt`` plus a saved model name) keeps working — the first read migrates it into
``instance/ai.json``. A key is never returned by any endpoint: only whether it exists, where it comes from and its
last four characters.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigma.application.ports import SettingsRepository
from sigma.domain.ai import (
    DEFAULT_ANTHROPIC_MODEL,
    LEGACY_PROVIDER,
    PROVIDERS,
    AiModel,
    clean_base_url,
    clean_label,
    clean_model_id,
    default_label,
    provider_of,
    providers_for_ui,
)
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.ai.servers import (
    LEGACY_KEY_FILE,
    StoredConfiguration,
    StoredModel,
    config_path,
    environment_model,
    find,
    legacy_entry,
    load_configuration,
    new_model_id,
    require_room,
    stored_models,
)

DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL
ENABLED_KEY = "ai_enabled"
DEFAULT_KEY = "ai_default_server"
MODEL_KEY = "ai_model"  # what older versions of Sigma Algebra wrote and read
CONFIG_KEY = "ai_models"  # the whole model list (with keys), persisted in the database


class AiSettings:
    def __init__(
        self,
        settings: SettingsRepository,
        instance_dir: Path,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._instance_dir = Path(instance_dir)
        self._path = config_path(instance_dir)
        self._environ = os.environ if environ is None else environ

    # ------------------------------------------------------------------ environment
    def _env(self, name: str) -> str:
        return (self._environ.get(name) or "").strip() if name else ""

    def _env_keys(self) -> dict[str, bool]:
        return {spec.key_env: bool(self._env(spec.key_env)) for spec in PROVIDERS if spec.key_env}

    def _legacy_key(self) -> str:
        try:
            return (self._instance_dir / LEGACY_KEY_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    # ------------------------------------------------------------------ configuration (database)
    @property
    def stored_configuration(self) -> StoredConfiguration:
        """The configured models, read from the database; an older on-disk ``ai.json`` is migrated on first read."""
        raw = self._settings.get(CONFIG_KEY)
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, Mapping):
                return StoredConfiguration.from_dict(data)
        file_config = load_configuration(self._path)
        if file_config.servers:
            self._settings.set(CONFIG_KEY, json.dumps(file_config.to_dict(), ensure_ascii=False))
            return file_config
        return StoredConfiguration()

    def _save(self, configuration: StoredConfiguration) -> None:
        self._settings.set(CONFIG_KEY, json.dumps(configuration.to_dict(), ensure_ascii=False))

    def _migrate_legacy(self, configuration: StoredConfiguration) -> StoredConfiguration:
        """Keep an installation working: the old key and model become the first configured entry."""
        key = self._legacy_key()
        env_key = self._env("ANTHROPIC_API_KEY")
        saved_model = self._settings.get(MODEL_KEY)
        if configuration.servers or not key and not (env_key and saved_model):
            # a key that only lives in the environment needs no migration: it is offered as a read-only entry
            return configuration
        model = saved_model or self._env("SIGMA_AI_MODEL") or DEFAULT_ANTHROPIC_MODEL
        # a key in the environment is *not* copied into the file: the entry keeps reading it from the environment
        entry = legacy_entry(model, "" if env_key and not key else key, default=True)
        configuration.servers = [entry]
        configuration.default_id = entry.id
        self._save(configuration)
        self._settings.set(DEFAULT_KEY, entry.id)
        return configuration

    # ------------------------------------------------------------------ models
    @property
    def models(self) -> list[AiModel]:
        """Every configured model, plus one read-only entry per provider whose key is in the environment."""
        configuration = self._migrate_legacy(self.stored_configuration)
        configured = stored_models(configuration, self._env_keys())
        covered = {model.provider for model in configured}
        for spec in PROVIDERS:
            if spec.key in covered or not self._env(spec.key_env):
                continue
            # each provider is offered with its own example model
            suggested = spec.suggested_models[0] if spec.suggested_models else DEFAULT_ANTHROPIC_MODEL
            configured.append(environment_model(spec.key, suggested, self._env(spec.key_env)))
        return configured

    def usable_models(self) -> list[AiModel]:
        return [model for model in self.models if model.usable]

    def find(self, model_id: str | None) -> AiModel | None:
        if not model_id:
            return None
        return next((model for model in self.models if model.id == model_id), None)

    def get(self, model_id: str) -> AiModel:
        model = self.find(model_id)
        if model is None:
            raise InvalidInputError(f"El modelo «{model_id}» no está configurado.")
        return model

    def default_model(self) -> AiModel | None:
        usable = {model.id: model for model in self.models if model.usable}
        preferred = self._settings.get(DEFAULT_KEY)
        if preferred and preferred in usable:
            return usable[preferred]
        return next(iter(usable.values()), None)

    def resolve(self, model_id: str | None = None) -> AiModel | None:
        """The model a request should use: the one asked for (if it can answer), else the default of the server."""
        if not model_id:
            return self.default_model()
        model = self.find(model_id)
        if model is None:
            raise InvalidInputError(f"El modelo «{model_id}» no está configurado.")
        if not model.enabled:
            raise InvalidInputError(f"El modelo «{model.label}» está desactivado.")
        return model

    # ------------------------------------------------------------------ management
    def add_model(
        self,
        provider: str,
        model: str,
        label: str = "",
        base_url: str = "",
        key: str = "",
        enabled: bool = True,
    ) -> StoredModel:
        spec = provider_of(provider)
        configuration = self._stored_for_update()
        require_room(configuration)
        chosen = clean_model_id(model, spec)
        entry = StoredModel(
            id=new_model_id(),
            provider=spec.key,
            label=clean_label(label, default_label(spec, chosen)),
            model=chosen,
            base_url=clean_base_url(base_url, spec),
            api_key=(key or "").strip(),
            enabled=enabled,
        )
        configuration.servers.append(entry)
        if configuration.default_id is None:
            configuration.default_id = entry.id
        self._settings.set(DEFAULT_KEY, configuration.default_id or "")
        self._save(configuration)
        return entry

    def update_model(self, model_id: str, changes: Mapping[str, Any]) -> StoredModel:
        configuration = self._stored_for_update()
        entry = find(configuration, model_id)
        spec = provider_of(str(changes.get("provider") or entry.provider))
        if spec.key != entry.provider:
            entry.provider = spec.key
            entry.base_url = spec.default_base_url
        if changes.get("model"):
            entry.model = clean_model_id(str(changes["model"]), spec)
        if changes.get("label") is not None:
            entry.label = clean_label(str(changes["label"]), default_label(spec, entry.model))
        if changes.get("base_url"):
            entry.base_url = clean_base_url(str(changes["base_url"]), spec)
        if changes.get("clear_key"):
            entry.api_key = ""
        elif changes.get("key"):
            entry.api_key = str(changes["key"]).strip()
        if changes.get("enabled") is not None:
            entry.enabled = bool(changes["enabled"])
        if changes.get("make_default"):
            configuration.default_id = entry.id
        if configuration.default_id is None and configuration.servers:
            configuration.default_id = configuration.servers[0].id
        self._settings.set(DEFAULT_KEY, configuration.default_id or "")
        self._save(configuration)
        return entry

    def remove_model(self, model_id: str) -> None:
        configuration = self._stored_for_update()
        entry = find(configuration, model_id)
        configuration.servers.remove(entry)
        ids = {item.id for item in configuration.servers}
        if configuration.default_id not in ids:
            configuration.default_id = configuration.servers[0].id if configuration.servers else None
        self._settings.set(DEFAULT_KEY, configuration.default_id or "")
        self._save(configuration)

    def _stored_for_update(self) -> StoredConfiguration:
        configuration = self._migrate_legacy(self.stored_configuration)
        return StoredConfiguration(default_id=configuration.default_id, servers=list(configuration.servers))

    # ------------------------------------------------------------------ compatibility (the single Anthropic entry)
    def _legacy_target(self) -> tuple[StoredConfiguration, StoredModel]:
        configuration = self._stored_for_update()
        target = next((entry for entry in configuration.servers if entry.provider == LEGACY_PROVIDER), None)
        if target is None:
            require_room(configuration)
            spec = provider_of(LEGACY_PROVIDER)
            model = clean_model_id(
                self._settings.get(MODEL_KEY) or self._env("SIGMA_AI_MODEL") or DEFAULT_ANTHROPIC_MODEL, spec
            )
            target = legacy_entry(model, "", default=True)
            configuration.servers.append(target)
            if configuration.default_id is None:
                configuration.default_id = target.id
        return configuration, target

    def set_key(self, key: str) -> None:
        key = (key or "").strip()
        if not key:
            raise InvalidInputError("Indique la clave de API.")
        configuration, target = self._legacy_target()
        target.api_key = key
        self._settings.set(DEFAULT_KEY, configuration.default_id or "")
        self._save(configuration)

    def clear_key(self) -> None:
        configuration = self._stored_for_update()
        target = next((entry for entry in configuration.servers if entry.provider == LEGACY_PROVIDER), None)
        if target is not None:
            target.api_key = ""
            self._save(configuration)

    @property
    def model(self) -> str:
        preferred = self._settings.get(MODEL_KEY)
        if preferred:
            return preferred
        default = self.default_model()
        return default.model if default is not None else (self._env("SIGMA_AI_MODEL") or DEFAULT_ANTHROPIC_MODEL)

    def set_model(self, model: str) -> None:
        cleaned = clean_model_id(model, provider_of(LEGACY_PROVIDER))
        configuration, target = self._legacy_target()
        target.model = cleaned
        target.label = default_label(provider_of(LEGACY_PROVIDER), cleaned)
        self._settings.set(MODEL_KEY, cleaned)
        self._settings.set(DEFAULT_KEY, configuration.default_id or "")
        self._save(configuration)

    @property
    def configured(self) -> bool:
        """True when at least one configured model can actually answer."""
        return bool(self.usable_models())

    @property
    def enabled(self) -> bool:
        return self._settings.get(ENABLED_KEY, "1") != "0"

    def set_enabled(self, enabled: bool) -> None:
        self._settings.set(ENABLED_KEY, "1" if enabled else "0")

    # ------------------------------------------------------------------ views
    def describe(self) -> dict[str, Any]:
        models = self.models
        preferred = self.default_model() or next(iter(models), None)
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "key_source": preferred.key_source if preferred else None,
            "key_hint": (preferred.key_tail or preferred.spec.key_hint) if preferred else None,
            "model": preferred.model if preferred else self.model,
            "default_model": DEFAULT_MODEL,
            "default_server_id": preferred.id if preferred else None,
            "models": [model.describe() for model in models],
            "providers": providers_for_ui(),
        }


__all__ = ["AiSettings", "DEFAULT_MODEL", "DEFAULT_ANTHROPIC_MODEL", "LEGACY_KEY_FILE", "new_model_id"]
