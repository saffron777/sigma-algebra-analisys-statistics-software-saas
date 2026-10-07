"""Which AI models the administrator can configure, and how their configuration is validated.

Pure data and rules: the catalogue of providers, the limits, and the normalization of one configured model. The
secret itself lives elsewhere (``instance/ai.json``, read only by the infrastructure layer); here a model is
described by *where* its key comes from, never by the key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sigma.domain.errors import InvalidInputError

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"  # the model older versions of Sigma used unless told otherwise
MAX_AI_MODELS = 12  # a configured model per provider is plenty; more is usually a mistake
DEFAULT_MAX_TOKENS = 8000
TEST_MAX_TOKENS = 1500

# The three HTTP shapes on the market today, as implemented by ``infrastructure.ai.providers``:
#   anthropic          → Messages API (always-on thinking + server-side fallbacks)
#   openai_compatible  → Chat Completions (OpenAI, Azure, DeepSeek, Groq, Mistral, xAI, OpenRouter, Ollama…)
#   gemini             → Google Generative Language generateContent
ANTHROPIC_KIND = "anthropic"
OPENAI_KIND = "openai_compatible"
GEMINI_KIND = "gemini"
KINDS = (ANTHROPIC_KIND, OPENAI_KIND, GEMINI_KIND)

# the provider whose entry is described by the pre-catalogue settings (ANTHROPIC_API_KEY, the saved ai_model)
LEGACY_PROVIDER = "anthropic"


@dataclass(frozen=True)
class ProviderSpec:
    """One provider «family» the page offers: only its connection defaults and example model names."""

    key: str
    label: str
    kind: str
    default_base_url: str
    path: str  # appended to the base URL (the model goes before it, for Gemini)
    key_required: bool
    key_env: str
    key_hint: str
    suggested_models: tuple[str, ...]
    note: str = ""
    placeholder: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        key="anthropic",
        label="Anthropic (Claude)",
        kind=ANTHROPIC_KIND,
        default_base_url="https://api.anthropic.com",
        path="/v1/messages",
        key_required=True,
        key_env="ANTHROPIC_API_KEY",
        key_hint="sk-ant-…",
        suggested_models=(
            DEFAULT_ANTHROPIC_MODEL,
            "claude-sonnet-5-5",
            "claude-haiku-4-5",
            "claude-opus-4-8",
        ),
        note="Opus piensa siempre: el esfuerzo se fija en «medio» y, si el modelo declina, se reintenta con Opus 4.8.",
    ),
    ProviderSpec(
        key="openai",
        label="OpenAI (ChatGPT)",
        kind=OPENAI_KIND,
        default_base_url="https://api.openai.com/v1",
        path="/chat/completions",
        key_required=True,
        key_env="OPENAI_API_KEY",
        key_hint="sk-…",
        suggested_models=("gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o", "o4-mini"),
        placeholder="Escriba el identificador del modelo, p. ej. gpt-5",
    ),
    ProviderSpec(
        key="azure_openai",
        label="Azure OpenAI",
        kind=OPENAI_KIND,
        default_base_url="",
        path="/chat/completions",
        key_required=True,
        key_env="AZURE_OPENAI_API_KEY",
        key_hint="clave del recurso",
        suggested_models=("gpt-5", "gpt-4o", "gpt-4.1", "o4-mini"),
        note="Pegue la URL del despliegue, incluido «?api-version=…», y elija el nombre del despliegue.",
        placeholder="nombre del despliegue, p. ej. gpt-5",
    ),
    ProviderSpec(
        key="gemini",
        label="Google Gemini",
        kind=GEMINI_KIND,
        default_base_url="https://generativelanguage.googleapis.com",
        path="/v1beta/models",
        key_required=True,
        key_env="GEMINI_API_KEY",
        key_hint="AIza…",
        suggested_models=("gemini-2.5-pro", "gemini-2.5-flash"),
        placeholder="Escriba el identificador del modelo, p. ej. gemini-2.5-pro",
    ),
    ProviderSpec(
        key="deepseek",
        label="DeepSeek",
        kind=OPENAI_KIND,
        default_base_url="https://api.deepseek.com/v1",
        path="/chat/completions",
        key_required=True,
        key_env="DEEPSEEK_API_KEY",
        key_hint="sk-…",
        suggested_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    ProviderSpec(
        key="groq",
        label="Groq",
        kind=OPENAI_KIND,
        default_base_url="https://api.groq.com/openai/v1",
        path="/chat/completions",
        key_required=True,
        key_env="GROQ_API_KEY",
        key_hint="gsk_…",
        suggested_models=("llama-3.3-70b-versatile", "mixtral-8x7b-32768"),
    ),
    ProviderSpec(
        key="mistral",
        label="Mistral AI",
        kind=OPENAI_KIND,
        default_base_url="https://api.mistral.ai/v1",
        path="/chat/completions",
        key_required=True,
        key_env="MISTRAL_API_KEY",
        key_hint="clave de Mistral",
        suggested_models=("mistral-large-latest", "mistral-small-latest"),
    ),
    ProviderSpec(
        key="xai",
        label="xAI (Grok)",
        kind=OPENAI_KIND,
        default_base_url="https://api.x.ai/v1",
        path="/chat/completions",
        key_required=True,
        key_env="XAI_API_KEY",
        key_hint="xai-…",
        suggested_models=("grok-4", "grok-3"),
    ),
    ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        kind=OPENAI_KIND,
        default_base_url="https://openrouter.ai/api/v1",
        path="/chat/completions",
        key_required=True,
        key_env="OPENROUTER_API_KEY",
        key_hint="sk-or-…",
        suggested_models=("anthropic/claude-sonnet-4", "deepseek/deepseek-chat", "meta-llama/llama-3.3-70b-instruct"),
        note="Un solo catálogo con modelos de muchos fabricantes.",
    ),
    ProviderSpec(
        key="together",
        label="Together AI",
        kind=OPENAI_KIND,
        default_base_url="https://api.together.xyz/v1",
        path="/chat/completions",
        key_required=True,
        key_env="TOGETHER_API_KEY",
        key_hint="clave de Together",
        suggested_models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
    ),
    ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        kind=OPENAI_KIND,
        default_base_url="http://127.0.0.1:11434/v1",
        path="/chat/completions",
        key_required=False,
        key_env="OLLAMA_API_KEY",
        key_hint="(sin clave)",
        suggested_models=("llama3.1", "qwen2.5", "mistral"),
        note="Se ejecuta en su propia máquina: no hace falta clave y no sale nada a internet.",
        placeholder="modelo instalado, p. ej. llama3.1",
    ),
    ProviderSpec(
        key="custom_openai",
        label="OpenAI-compatible personalizado",
        kind=OPENAI_KIND,
        default_base_url="",
        path="/chat/completions",
        key_required=False,
        key_env="",
        key_hint="(según el servicio)",
        suggested_models=(),
        note="LM Studio, vLLM, una puerta de enlace propia… Indique la URL base y el modelo a mano.",
        placeholder="identificador del modelo en ese servicio",
    ),
)

PROVIDERS_BY_KEY: dict[str, ProviderSpec] = {spec.key: spec for spec in PROVIDERS}

# environment variables a kernel must never see: they would let any code on the console spend the server's AI quota
SECRET_KEY_ENV_VARS: tuple[str, ...] = tuple(dict.fromkeys(spec.key_env for spec in PROVIDERS if spec.key_env))

_BASE_URL = re.compile(r"^https?://[^\s\x00-\x1f\x7f]{1,300}$")


def provider_of(key: str) -> ProviderSpec:
    try:
        return PROVIDERS_BY_KEY[str(key or "").strip().lower()]
    except KeyError:
        raise InvalidInputError(f"Proveedor de IA no soportado: «{key}».") from None


def model_example(provider: ProviderSpec | None) -> str:
    """A model name to show in an error message: the provider's own example, or a well-known one."""
    if provider is None:
        return "gpt-5"
    if provider.suggested_models:
        return provider.suggested_models[0]
    if provider.placeholder:
        for marker in ("p. ej.", "por ejemplo", "ej."):
            if marker in provider.placeholder:
                return provider.placeholder.split(marker, 1)[1].strip(" .")
    return "gpt-5"


def clean_model_id(value: str, provider: ProviderSpec | None = None) -> str:
    """The identifier of the model, with a message that says what to type when it is not one.

    A real identifier names a family and a version («gpt-5», «deepseek-chat», «llama3.1», «gemini-2.5-pro»), so a
    bare word like «deepseek» or «claude» is refused: it is almost always a truncated paste. Each reason has its own
    message, because «write a model name» and «write it differently» are different problems — and the offending value
    is echoed back, which is what usually reveals the typo.
    """
    model = " ".join((value or "").split())
    example = model_example(provider)
    if not model:
        raise InvalidInputError(f"Escriba el identificador del modelo, por ejemplo «{example}».")
    if len(model) > 200 or any(ord(char) < 32 for char in model):
        raise InvalidInputError("El nombre del modelo tiene caracteres no válidos.")
    if " " in model:
        raise InvalidInputError("El nombre del modelo no puede llevar espacios; escríbalo tal cual lo da el proveedor.")
    if model.startswith(("http://", "https://")):
        raise InvalidInputError(
            "En «Modelo» va el nombre del modelo, no la dirección del servicio: esa va en «Dirección (URL)»."
        )
    if not re.search(r"[-.]", model):
        raise InvalidInputError(
            f"«{model}» no parece un identificador de modelo: escríbalo completo tal como lo da el proveedor, "
            f"por ejemplo «{example}». Si el nombre es correcto, pruébelo con «OpenAI-compatible personalizado»."
        )
    return model


def clean_label(value: str, fallback: str) -> str:
    text = " ".join((value or "").split())
    return (text[:60] or fallback[:60]).strip()


def clean_base_url(value: str, provider: ProviderSpec) -> str:
    url = (value or "").strip() or provider.default_base_url
    if not url:
        raise InvalidInputError("Indique la dirección (URL) del servicio de IA.")
    if not _BASE_URL.match(url):
        raise InvalidInputError("La dirección del servicio debe empezar por «http://» o «https://».")
    return url.rstrip()


def providers_for_ui() -> list[dict[str, Any]]:
    """The provider list the administration page offers (never a secret: only defaults and examples)."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "kind": spec.kind,
            "default_base_url": spec.default_base_url,
            "path": spec.path,
            "key_required": spec.key_required,
            "key_hint": spec.key_hint,
            "suggested_models": list(spec.suggested_models),
            "note": spec.note,
            "placeholder": spec.placeholder,
        }
        for spec in PROVIDERS
    ]


@dataclass(frozen=True)
class AiModel:
    """One model the server can use, already fully specified (with the *origin* of its key, never the key).

    ``key_tail`` is the last four characters of a stored key, the most the page is ever allowed to show.
    """

    id: str
    provider: str
    label: str
    model: str
    base_url: str
    enabled: bool = True
    has_env_key: bool = False  # the environment provides a key for this provider
    stored_key: str | None = None  # never serialized: the adapters use it, ``describe()`` does not
    read_only: bool = False  # comes from the environment, not from the configuration file

    @property
    def spec(self) -> ProviderSpec:
        return provider_of(self.provider)

    @property
    def has_key(self) -> bool:
        return bool(self.stored_key) or self.has_env_key

    @property
    def key_source(self) -> str | None:
        return "file" if self.stored_key else ("env" if self.has_env_key else None)

    @property
    def key_tail(self) -> str | None:
        key = self.stored_key or ""
        return f"…{key[-4:]}" if len(key) >= 8 else None

    @property
    def missing_key(self) -> bool:
        return self.spec.key_required and not self.has_key

    @property
    def usable(self) -> bool:
        return self.enabled and not self.missing_key

    def describe(self) -> dict[str, Any]:
        """Everything the administration page may know about this model."""
        return {
            "id": self.id,
            "provider": self.provider,
            "provider_label": self.spec.label,
            "kind": self.spec.kind,
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "enabled": self.enabled,
            "has_key": self.has_key,
            "key_source": self.key_source,
            "key_required": self.spec.key_required,
            "key_hint": self.key_tail or (self.spec.key_hint if self.has_env_key or not self.stored_key else None),
            "read_only": self.read_only,
            "available": self.usable,
        }


def new_model_id() -> str:
    return f"ai_{uuid4().hex[:10]}"


def default_label(provider: ProviderSpec, model: str) -> str:
    return f"{provider.label} · {model}"[:60]
