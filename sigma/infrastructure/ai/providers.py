"""The three HTTP shapes of the market: how a prompt becomes a request, and a response becomes an answer.

Pure functions over plain dictionaries, so every provider rule is tested without a network. The adapters in this
package only add the transport (``requests``) and the retries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sigma.domain.ai import (
    ANTHROPIC_KIND,
    DEFAULT_MAX_TOKENS,
    GEMINI_KIND,
    OPENAI_KIND,
    TEST_MAX_TOKENS,
)
from sigma.domain.errors import AiUnavailableError, InvalidInputError, RateLimitedError
from sigma.domain.interpretation import Prompt

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-06-01"
ANTHROPIC_FALLBACK_MODEL = "claude-opus-4-8"
ANTHROPIC_EFFORT = "medium"
GEMINI_TRUNCATED_REASONS = ("MAX_TOKENS",)
GEMINI_BLOCKED_REASONS = ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "OTHER")

TEST_SYSTEM = "Responde únicamente con la palabra OK."
TEST_USER = "Prueba de conexión."


@dataclass(frozen=True)
class Prepared:
    """One HTTP call: where to send it, with which headers and body, and which model it uses."""

    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    model: str


def join_url(base_url: str, path: str) -> str:
    """«https://api.openai.com/v1» + «/chat/completions» → the endpoint; a query string is preserved."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise InvalidInputError("Indique la dirección (URL) del servicio de IA.")
    return f"{base}{path if path.startswith('/') else '/' + path}"


def anthropic_prepared(model: str, base_url: str, prompt: Prompt, max_tokens: int, fallback: bool) -> Prepared:
    """Messages API. Opus thinks by default, takes no sampling parameters, and supports server-side fallbacks."""
    use_fallback = fallback and model != ANTHROPIC_FALLBACK_MODEL
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user}],
        "output_config": {"effort": ANTHROPIC_EFFORT},
    }
    headers = {"anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    if use_fallback:
        body["fallbacks"] = [{"model": ANTHROPIC_FALLBACK_MODEL}]
        headers["anthropic-beta"] = ANTHROPIC_FALLBACK_BETA
    return Prepared(url=join_url(base_url, "/v1/messages"), headers=headers, body=body, model=model)


def openai_prepared(model: str, base_url: str, prompt: Prompt, max_tokens: int) -> Prepared:
    body = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    }
    return Prepared(
        url=join_url(base_url, "/chat/completions"),
        headers={"content-type": "application/json"},
        body=body,
        model=model,
    )


def gemini_prepared(model: str, base_url: str, prompt: Prompt, max_tokens: int) -> Prepared:
    name = model if model.startswith("models/") else f"models/{model}"
    body = {
        "systemInstruction": {"parts": [{"text": prompt.system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
        "generationConfig": {"maxOutputTokens": int(max_tokens)},
    }
    return Prepared(
        url=join_url(base_url, f"/v1beta/{name}:generateContent"),
        headers={"content-type": "application/json"},
        body=body,
        model=model,
    )


def prepared_for(
    kind: str, model: str, base_url: str, prompt: Prompt, max_tokens: int, fallback: bool = True
) -> Prepared:
    if kind == ANTHROPIC_KIND:
        return anthropic_prepared(model, base_url, prompt, max_tokens, fallback)
    if kind == OPENAI_KIND:
        return openai_prepared(model, base_url, prompt, max_tokens)
    if kind == GEMINI_KIND:
        return gemini_prepared(model, base_url, prompt, max_tokens)
    raise InvalidInputError(f"Tipo de servicio de IA no soportado: «{kind}».")


def test_prompt() -> Prompt:
    return Prompt(system=TEST_SYSTEM, user=TEST_USER)


def max_tokens_for(testing: bool) -> int:
    return TEST_MAX_TOKENS if testing else DEFAULT_MAX_TOKENS


def auth_headers(kind: str, key: str) -> dict[str, str]:
    """How each family proves who it is. A local service (Ollama, LM Studio) may need nothing at all."""
    if not key:
        return {}
    if kind == ANTHROPIC_KIND:
        return {"x-api-key": key}
    if kind == GEMINI_KIND:
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


# ------------------------------------------------------------------ response parsing
@dataclass(frozen=True)
class Answer:
    """The text the model produced, what it cost, and whether it was cut off or refused."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_used: bool = False
    truncated: bool = False
    blocked: str = ""  # a refusal / safety stop: no text was produced on purpose


def _text_parts(value: Any) -> str:
    """OpenAI returns ``content`` as a string or as a list of parts; Gemini nests ``parts[].text``."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _text_parts(value.get("text"))
    if isinstance(value, list):
        return "".join(_text_parts(item) for item in value)
    return ""


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def parse_anthropic(payload: Mapping[str, Any], requested_model: str) -> Answer:
    stop = str(payload.get("stop_reason") or "")
    if stop == "refusal":
        return Answer(text="", model=requested_model, blocked="refusal")
    blocks = payload.get("content") or []
    text = "".join(_text_parts(block.get("text")) for block in blocks if isinstance(block, Mapping)).strip()
    usage = _mapping(payload.get("usage"))
    iterations = usage.get("iterations") or []
    fallback_used = any(isinstance(b, Mapping) and b.get("type") == "fallback" for b in blocks) or any(
        isinstance(i, Mapping) and i.get("type") == "fallback_message" for i in iterations
    )
    return Answer(
        text=text,
        model=str(payload.get("model") or requested_model),
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        fallback_used=fallback_used,
        truncated=stop == "max_tokens",
    )


def parse_openai(payload: Mapping[str, Any], requested_model: str) -> Answer:
    choices = payload.get("choices") or []
    first = _mapping(choices[0]) if choices else {}
    message = _mapping(first.get("message"))
    text = _text_parts(message.get("content")).strip()
    finish = str(first.get("finish_reason") or "")
    usage = _mapping(payload.get("usage"))
    return Answer(
        text=text,
        model=str(payload.get("model") or requested_model),
        input_tokens=_int(usage.get("prompt_tokens")),
        output_tokens=_int(usage.get("completion_tokens")),
        truncated=finish == "length",
        blocked="refusal" if finish == "content_filter" else "",
    )


def parse_gemini(payload: Mapping[str, Any], requested_model: str) -> Answer:
    candidates = payload.get("candidates") or []
    first = _mapping(candidates[0]) if candidates else {}
    reason = str(first.get("finishReason") or "")
    content = _mapping(first.get("content"))
    text = _text_parts(content.get("parts")).strip()
    usage = _mapping(payload.get("usageMetadata"))
    return Answer(
        text=text,
        model=requested_model,  # Gemini echoes only the model version, not the requested name
        input_tokens=_int(usage.get("promptTokenCount")),
        output_tokens=_int(usage.get("candidatesTokenCount")),
        truncated=reason in GEMINI_TRUNCATED_REASONS,
        blocked="" if text else (reason if reason in GEMINI_BLOCKED_REASONS else ""),
    )


def parse_for(kind: str, payload: Mapping[str, Any], requested_model: str) -> Answer:
    if kind == ANTHROPIC_KIND:
        return parse_anthropic(payload, requested_model)
    if kind == OPENAI_KIND:
        return parse_openai(payload, requested_model)
    if kind == GEMINI_KIND:
        return parse_gemini(payload, requested_model)
    raise InvalidInputError(f"Tipo de servicio de IA no soportado: «{kind}».")


# ------------------------------------------------------------------ errors
def error_detail(payload: Any) -> str:
    """The readable part of an error body, whatever the provider calls it."""
    if isinstance(payload, str):
        return " ".join(payload.split())[:300]
    if not isinstance(payload, Mapping):
        return ""
    for key in ("message", "detail", "error_description", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:300]
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return " ".join(error.split())[:300]
    if isinstance(error, (Mapping, list)) and error:
        return error_detail(error[0] if isinstance(error, list) else error)
    return ""


def safe_message(message: str, *secrets: str) -> str:
    """Never let a key (or a long token) travel back to the browser inside an error message."""
    text = " ".join(str(message or "").split())[:300]
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "«clave oculta»")
    return text


def error_for_status(status: int, payload: Any, provider_label: str = "El proveedor de IA", key: str = "") -> Exception:
    """An HTTP failure of a provider → the error the web layer already knows how to show."""
    detail = safe_message(error_detail(payload), key)
    tail = f" Detalle: {detail}" if detail else ""
    if status in (401, 403):
        return AiUnavailableError(
            f"{provider_label} rechazó la clave de API (no es válida o no tiene permiso para este modelo)."
            " Revise la configuración del modelo."
        )
    if status == 404:
        return AiUnavailableError(f"{provider_label} no encontró el modelo o la dirección indicada.{tail}")
    if status == 429:
        return RateLimitedError(
            f"{provider_label} está saturado. Espere un momento y vuelva a intentarlo.", {"retry_after": 30}
        )
    if status in (400, 422):
        return AiUnavailableError(f"{provider_label} rechazó la solicitud.{tail}")
    if status >= 500:
        return AiUnavailableError(f"{provider_label} no está disponible en este momento. Inténtelo más tarde.")
    return AiUnavailableError(f"{provider_label} devolvió un error inesperado ({status}).{tail}")
