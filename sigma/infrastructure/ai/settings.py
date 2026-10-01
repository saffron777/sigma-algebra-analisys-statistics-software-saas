"""The administrator's AI settings: API key (environment or ``instance/ai_key.txt``), model and the on/off switch.

The key is never returned by any endpoint: only whether it exists, where it comes from and its last four characters.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigma.application.ports import SettingsRepository
from sigma.domain.errors import InvalidInputError

DEFAULT_MODEL = "claude-opus-5-5"
_MODEL_PATTERN = re.compile(r"^claude-[a-z0-9][a-z0-9.-]{2,60}$")
_KEY_PATTERN = re.compile(r"^sk-ant-[A-Za-z0-9_\-]{16,}$")


class AiSettings:
    def __init__(
        self, settings: SettingsRepository, instance_dir: Path, environ: Mapping[str, str] | None = None
    ) -> None:
        self._settings = settings
        self._key_file = Path(instance_dir) / "ai_key.txt"
        self._environ = os.environ if environ is None else environ

    # ------------------------------------------------------------------ key
    def _env_key(self) -> str | None:
        return (self._environ.get("ANTHROPIC_API_KEY") or "").strip() or None

    def _file_key(self) -> str | None:
        try:
            return self._key_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    @property
    def api_key(self) -> str | None:
        return self._env_key() or self._file_key()

    @property
    def key_source(self) -> str | None:
        return "env" if self._env_key() else ("file" if self._file_key() else None)

    @property
    def configured(self) -> bool:
        return self.api_key is not None

    def set_key(self, key: str) -> None:
        key = (key or "").strip()
        if not _KEY_PATTERN.match(key):
            raise InvalidInputError(
                "La clave no tiene el formato de una clave de API de Anthropic (empieza por «sk-ant-»)."
            )
        self._key_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._key_file.with_suffix(".tmp")
        temporary.write_text(key, encoding="utf-8")
        with contextlib.suppress(OSError):
            temporary.chmod(0o600)  # best effort: a no-op for most of Windows
        os.replace(temporary, self._key_file)

    def clear_key(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._key_file.unlink()

    # ------------------------------------------------------------------ model and switch
    @property
    def model(self) -> str:
        return self._settings.get("ai_model") or self._environ.get("SIGMA_AI_MODEL") or DEFAULT_MODEL

    def set_model(self, model: str) -> None:
        model = (model or "").strip()
        if not _MODEL_PATTERN.match(model):
            raise InvalidInputError("Nombre de modelo inválido (por ejemplo, claude-opus-5-5).")
        self._settings.set("ai_model", model)

    @property
    def enabled(self) -> bool:
        return self._settings.get("ai_enabled", "1") != "0"

    def set_enabled(self, enabled: bool) -> None:
        self._settings.set("ai_enabled", "1" if enabled else "0")

    def describe(self) -> dict[str, Any]:
        key = self.api_key
        return {
            "enabled": self.enabled,
            "configured": key is not None,
            "key_source": self.key_source,
            "key_hint": f"…{key[-4:]}" if key and len(key) >= 8 else None,
            "model": self.model,
            "default_model": DEFAULT_MODEL,
        }
