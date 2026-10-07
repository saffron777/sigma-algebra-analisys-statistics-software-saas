"""The configured AI models, kept on disk as JSON.

Unlike the database choice (``instance/database.json``) this file holds API keys, so it is written atomically with
mode 0600 and is **never** read by the web layer directly: the administration page only learns whether a key exists
and its last four characters.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigma.domain.ai import (
    DEFAULT_ANTHROPIC_MODEL,
    LEGACY_PROVIDER,
    MAX_AI_MODELS,
    AiModel,
    clean_base_url,
    clean_label,
    clean_model_id,
    default_label,
    new_model_id,
    provider_of,
)
from sigma.domain.errors import InvalidInputError

FILE_NAME = "ai.json"
FILE_VERSION = 1
LEGACY_KEY_FILE = "ai_key.txt"


def _flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", ""}


@dataclass
class StoredModel:
    """One entry exactly as it is written to ``ai.json`` (the key included)."""

    id: str
    provider: str
    label: str
    model: str
    base_url: str
    api_key: str = ""
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StoredModel:
        spec = provider_of(str(data.get("provider") or ""))
        model = clean_model_id(str(data.get("model") or ""), spec)
        return cls(
            id=str(data.get("id") or new_model_id()),
            provider=spec.key,
            label=clean_label(str(data.get("label") or ""), default_label(spec, model)),
            model=model,
            base_url=clean_base_url(str(data.get("base_url") or ""), spec),
            api_key=str(data.get("api_key") or ""),
            enabled=_flag(data.get("enabled"), True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "enabled": self.enabled,
        }


@dataclass
class StoredConfiguration:
    default_id: str | None = None
    servers: list[StoredModel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StoredConfiguration:
        servers: list[StoredModel] = []
        ids: set[str] = set()
        for raw in data.get("servers") or []:
            if not isinstance(raw, Mapping):
                continue
            entry = StoredModel.from_dict(raw)
            while entry.id in ids:  # a hand-edited file must not create two models with the same id
                entry.id = new_model_id()
            ids.add(entry.id)
            servers.append(entry)
        default_id = data.get("default")
        return cls(default_id=str(default_id) if default_id in ids else None, servers=servers[:MAX_AI_MODELS])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FILE_VERSION,
            "default": self.default_id,
            "servers": [entry.to_dict() for entry in self.servers],
        }


def config_path(instance_dir: Path) -> Path:
    return Path(instance_dir) / FILE_NAME


def load_configuration(path: Path) -> StoredConfiguration:
    """The file, or an empty configuration when it does not exist yet (or a hand-edit broke it)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StoredConfiguration()
    return StoredConfiguration.from_dict(data) if isinstance(data, Mapping) else StoredConfiguration()


def save_configuration(path: Path, configuration: StoredConfiguration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(configuration.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    with contextlib.suppress(OSError):
        temporary.chmod(0o600)  # best effort: it holds API keys
    os.replace(temporary, path)


def stored_models(configuration: StoredConfiguration, env_keys: Mapping[str, bool]) -> list[AiModel]:
    """The file's entries as domain models: the key travels inside, but ``describe()`` never shows it."""
    return [
        AiModel(
            id=entry.id,
            provider=entry.provider,
            label=entry.label,
            model=entry.model,
            base_url=entry.base_url,
            enabled=entry.enabled,
            has_env_key=bool(env_keys.get(provider_of(entry.provider).key_env)) and not entry.api_key,
            stored_key=entry.api_key or None,
        )
        for entry in configuration.servers
    ]


def environment_model(provider_key: str, model: str, env_key: str) -> AiModel:
    """A read-only entry backed by an environment variable, so a deployment needs no configuration at all."""
    spec = provider_of(provider_key)
    return AiModel(
        id=f"env_{spec.key}",
        provider=spec.key,
        label=f"{spec.label} (variable de entorno)",
        model=clean_model_id(model, spec) if model else DEFAULT_ANTHROPIC_MODEL,
        base_url=spec.default_base_url,
        enabled=True,
        has_env_key=bool(env_key),
        read_only=True,
    )


def entries_of(configuration: StoredConfiguration) -> list[StoredModel]:
    return configuration.servers


def legacy_entry(model: str, api_key: str, default: bool) -> StoredModel:
    """The entry that keeps an installation working after the update (same key, same model)."""
    spec = provider_of(LEGACY_PROVIDER)
    chosen = clean_model_id(model, spec) if model else DEFAULT_ANTHROPIC_MODEL
    return StoredModel(
        id=new_model_id(),
        provider=spec.key,
        label=default_label(spec, chosen),
        model=chosen,
        base_url=spec.default_base_url,
        api_key=api_key,
        enabled=True,
    )


def require_room(configuration: StoredConfiguration) -> None:
    if len(configuration.servers) >= MAX_AI_MODELS:
        raise InvalidInputError(f"Ya hay {MAX_AI_MODELS} modelos configurados. Elimine uno antes de añadir otro.")


def find(configuration: StoredConfiguration, model_id: str) -> StoredModel:
    for entry in configuration.servers:
        if entry.id == model_id:
            return entry
    raise InvalidInputError(f"El modelo «{model_id}» no está configurado.")
