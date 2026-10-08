"""The configurable AI models: settings, storage, the provider adapters and the gateway."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from sigma.domain.ai import MAX_AI_MODELS, AiModel, ProviderSpec, provider_of
from sigma.domain.errors import AiUnavailableError, InvalidInputError, RateLimitedError
from sigma.domain.interpretation import Prompt
from sigma.infrastructure.ai.adapters import AnthropicAdapter, GeminiAdapter, OpenAICompatibleAdapter
from sigma.infrastructure.ai.gateway import ConfiguredGateway
from sigma.infrastructure.ai.http import JsonClient
from sigma.infrastructure.ai.providers import ANTHROPIC_FALLBACK_BETA, ANTHROPIC_FALLBACK_MODEL
from sigma.infrastructure.ai.settings import DEFAULT_MODEL, AiSettings, config_path
from sigma.infrastructure.persistence.sql.ai_repositories import SqlAiUsageRepository, SqlSettingsRepository
from sigma.infrastructure.persistence.sql.engine import Database
from tests.support import add_user

KEY = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
OPENAI_KEY = "sk-openai-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
PROMPT = Prompt(system="sistema", user="usuario")
LongKey = "sk-proj-" + "Z" * 40


@pytest.fixture
def tmp_path():
    """A scratch directory next to the tests, so the suite also runs where the system TEMP is not writable."""
    path = Path(__file__).resolve().parents[2] / ".ai-tmp" / uuid4().hex[:10]
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def settings(tmp_path, environ=None, name="s.db") -> AiSettings:
    return AiSettings(
        SqlSettingsRepository(Database(tmp_path / name)), tmp_path, environ if environ is not None else {}
    )


def file_of(tmp_path) -> dict[str, Any]:
    return json.loads(config_path(tmp_path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ storage and migration
def test_a_fresh_installation_has_no_models(tmp_path):
    ai = settings(tmp_path)
    assert ai.models == [] and not ai.configured and ai.enabled
    assert ai.default_model() is None and ai.describe()["key_hint"] is None


def test_the_old_key_and_model_are_migrated_to_one_claude_entry(tmp_path):
    (tmp_path / "ai_key.txt").write_text(f"  {KEY}\n", encoding="utf-8")
    database = Database(tmp_path / "s.db")
    SqlSettingsRepository(database).set("ai_model", "claude-sonnet-5-5")
    ai = AiSettings(SqlSettingsRepository(database), tmp_path, {})

    models = ai.models
    assert len(models) == 1 and models[0].provider == "anthropic" and models[0].model == "claude-sonnet-5-5"
    assert models[0].stored_key == KEY and ai.configured and ai.default_model().id == models[0].id
    stored = SqlSettingsRepository(database).get("ai_models")
    assert stored and KEY in stored  # the migrated entry now lives in the database
    assert (tmp_path / "ai_key.txt").is_file()  # left alone: the entry is the source of truth now


def test_a_key_in_the_environment_is_read_but_never_copied_to_the_file(tmp_path):
    ai = settings(tmp_path, {"ANTHROPIC_API_KEY": KEY})
    models = ai.models
    assert len(models) == 1 and models[0].has_env_key and models[0].stored_key is None
    assert ai.describe()["key_source"] == "env" and ai.configured
    assert not config_path(tmp_path).is_file()  # nothing to persist: the entry is read-only
    assert KEY not in json.dumps(ai.describe())


def test_key_format_is_validated_and_never_returned(tmp_path):
    ai = settings(tmp_path)
    ai.set_key(f"  {KEY}\n")
    described = ai.describe()
    assert described["configured"] and described["key_source"] == "file"
    assert described["key_hint"] == "…1234" and KEY not in json.dumps(described)
    with pytest.raises(InvalidInputError):
        ai.set_key("   ")
    ai.clear_key()
    ai.clear_key()  # clearing twice is fine
    assert not ai.configured


def test_model_and_switch_are_persisted(tmp_path):
    ai = settings(tmp_path)
    assert ai.model == DEFAULT_MODEL == "claude-opus-5-5" and ai.enabled
    ai.set_model("claude-sonnet-5-5")
    ai.set_enabled(False)
    again = settings(tmp_path)
    assert again.model == "claude-sonnet-5-5" and not again.enabled
    for bad in ("", "claude", "http://api.example.com", "con espacio"):
        with pytest.raises(InvalidInputError):
            ai.set_model(bad)
    assert settings(tmp_path, {"SIGMA_AI_MODEL": "claude-haiku-4-5"}, name="other.db").model == "claude-haiku-4-5"


def test_adding_editing_and_removing_models(tmp_path):
    ai = settings(tmp_path)
    first = ai.add_model("anthropic", "claude-opus-5-5", "Claude de producción", key=KEY)
    second = ai.add_model("ollama", "llama3.1", base_url="http://127.0.0.1:11434/v1")
    assert ai.default_model().id == first.id  # the first one added becomes the default

    described = {model["id"]: model for model in ai.describe()["models"]}
    assert described[first.id]["provider_label"] == "Anthropic (Claude)" and described[first.id]["available"]
    assert described[second.id]["key_required"] is False and described[second.id]["available"]
    assert described[second.id]["base_url"] == "http://127.0.0.1:11434/v1"

    ai.update_model(second.id, {"label": "Ollama local", "make_default": True, "model": "qwen2.5"})
    assert ai.default_model().id == second.id
    assert ai.get(second.id).label == "Ollama local" and ai.get(second.id).model == "qwen2.5"

    ai.update_model(first.id, {"clear_key": True})
    assert ai.get(first.id).stored_key is None and not ai.get(first.id).usable

    ai.remove_model(second.id)
    assert [model.id for model in ai.models] == [first.id]
    assert ai.default_model() is None  # the remaining entry has no key
    with pytest.raises(InvalidInputError):
        ai.remove_model("no-existe")


def test_a_model_can_be_disabled_and_a_remote_provider_needs_a_key(tmp_path):
    ai = settings(tmp_path)
    entry = ai.add_model("openai", "gpt-5", key=OPENAI_KEY)
    assert ai.get(entry.id).usable
    ai.update_model(entry.id, {"enabled": False})
    assert not ai.get(entry.id).usable and ai.default_model() is None
    ai.update_model(entry.id, {"enabled": True, "base_url": "https://api.openai.com/v1"})
    assert ai.get(entry.id).usable
    without_key = ai.add_model("groq", "llama-3.3-70b-versatile")
    assert not ai.get(without_key.id).usable and ai.get(without_key.id).missing_key


def test_the_number_of_models_is_limited(tmp_path):
    ai = settings(tmp_path)
    for index in range(MAX_AI_MODELS):
        ai.add_model("ollama", f"modelo-{index}")
    with pytest.raises(InvalidInputError, match="modelos configurados"):
        ai.add_model("ollama", "otro-modelo")


def test_bad_urls_providers_and_models_are_rejected(tmp_path):
    ai = settings(tmp_path)
    with pytest.raises(InvalidInputError):
        ai.add_model("no-existe", "algo")
    with pytest.raises(InvalidInputError):
        ai.add_model("ollama", "llama3.1", base_url="ftp://127.0.0.1")
    with pytest.raises(InvalidInputError):
        ai.add_model("azure_openai", "despliegue")  # Azure has no default address
    with pytest.raises(InvalidInputError):
        ai.add_model("ollama", "")


def test_the_model_error_explains_what_to_write(tmp_path):
    """The message must tell apart «empty» from «this does not look like a model», and name the offender."""
    ai = settings(tmp_path)
    with pytest.raises(InvalidInputError, match="Escriba el identificador del modelo"):
        ai.add_model("deepseek", "")
    with pytest.raises(InvalidInputError, match="«deepseek» no parece un identificador"):
        ai.add_model("deepseek", "deepseek")  # the family without the version: a truncated paste
    with pytest.raises(InvalidInputError, match="deepseek-chat"):  # the message suggests the real example
        ai.add_model("deepseek", "deepseek")
    with pytest.raises(InvalidInputError, match="no puede llevar espacios"):
        ai.add_model("deepseek", "deepseek chat")
    with pytest.raises(InvalidInputError, match="no la dirección del servicio"):
        ai.add_model("deepseek", "https://api.deepseek.com/v1")
    assert ai.get(ai.add_model("deepseek", "  deepseek-chat  ").id).model == "deepseek-chat"


def test_a_missing_file_or_a_broken_one_is_an_empty_configuration(tmp_path):
    ai = settings(tmp_path)
    config_path(tmp_path).write_text("{no es json", encoding="utf-8")
    assert ai.models == []
    config_path(tmp_path).write_text("[]", encoding="utf-8")
    assert ai.models == []


# ------------------------------------------------------------------ the adapters (fake transport)
class RecordingClient(JsonClient):
    def __init__(self, *responses) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses)

    def post(self, url, headers, body, timeout=150.0):
        self.calls.append({"url": url, "headers": dict(headers), "body": body})
        return self.responses.pop(0) if self.responses else (200, {})


def model_of(provider: str, model: str, key: str = "", base_url: str = "", **kwargs) -> AiModel:
    spec: ProviderSpec = provider_of(provider)
    return AiModel(
        id=f"ai_{provider}",
        provider=spec.key,
        label=f"{spec.label} · {model}",
        model=model,
        base_url=base_url or spec.default_base_url,
        stored_key=key or None,
        **kwargs,
    )


def openai_payload(text="Interpretación.", model="gpt-5", finish="stop", usage=(11, 7)):
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def gemini_payload(text="Interpretación.", finish="STOP", usage=(9, 5)):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1]},
    }


def test_openai_compatible_request_and_answer():
    client = RecordingClient((200, openai_payload()))
    adapter = OpenAICompatibleAdapter(model_of("openai", "gpt-5", OPENAI_KEY), client)
    output = adapter.interpret(PROMPT)

    call = client.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {OPENAI_KEY}"
    assert call["body"]["model"] == "gpt-5" and call["body"]["max_tokens"] == 8000
    assert call["body"]["messages"][0] == {"role": "system", "content": "sistema"}
    assert output.text == "Interpretación." and output.model == "gpt-5"
    assert (output.input_tokens, output.output_tokens) == (11, 7)
    assert output.provider == "openai" and not output.fallback_used and not output.truncated


def test_openai_accepts_a_list_of_content_parts_and_flags_truncation():
    payload = openai_payload(
        [{"type": "text", "text": "Hola"}, {"type": "text", "text": " mundo"}], finish="length"
    )
    adapter = OpenAICompatibleAdapter(model_of("openai", "gpt-5", OPENAI_KEY), RecordingClient((200, payload)))
    output = adapter.interpret(PROMPT)
    assert output.text == "Hola mundo" and output.truncated


def test_a_local_provider_sends_no_authorization_header_and_honours_the_base_url():
    client = RecordingClient((200, openai_payload(text="Ok", model="llama3.1")))
    adapter = OpenAICompatibleAdapter(model_of("ollama", "llama3.1", base_url="http://127.0.0.1:11434/v1"), client)
    assert adapter.interpret(PROMPT).text == "Ok"
    assert "Authorization" not in client.calls[0]["headers"]
    assert client.calls[0]["url"] == "http://127.0.0.1:11434/v1/chat/completions"


def test_gemini_request_and_answer():
    client = RecordingClient((200, gemini_payload()))
    output = GeminiAdapter(model_of("gemini", "gemini-2.5-pro", "AIza-test-key"), client).interpret(PROMPT)

    call = client.calls[0]
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    assert call["headers"]["x-goog-api-key"] == "AIza-test-key"
    assert call["body"]["systemInstruction"]["parts"][0]["text"] == "sistema"
    assert call["body"]["generationConfig"]["maxOutputTokens"] == 8000
    assert output.text == "Interpretación." and output.model == "gemini-2.5-pro"
    assert (output.input_tokens, output.output_tokens) == (9, 5)


def test_gemini_a_safety_stop_is_reported_as_a_refusal():
    client = RecordingClient((200, {"candidates": [{"content": {}, "finishReason": "SAFETY"}]}))
    with pytest.raises(AiUnavailableError, match="declinó"):
        GeminiAdapter(model_of("gemini", "gemini-2.5-pro", "AIza-key"), client).interpret(PROMPT)


def test_anthropic_request_keeps_its_thinking_and_fallback_configuration():
    client = RecordingClient(
        (
            200,
            {
                "model": "claude-opus-5-5",
                "stop_reason": "end_turn",
                "content": [{"type": "thinking"}, {"type": "text", "text": "Interpretación."}],
                "usage": {"input_tokens": 1200, "output_tokens": 380},
            },
        )
    )
    output = AnthropicAdapter(model_of("anthropic", "claude-opus-5-5", KEY), client).interpret(PROMPT)

    call = client.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == KEY and call["headers"]["anthropic-version"]
    assert call["headers"]["anthropic-beta"] == ANTHROPIC_FALLBACK_BETA
    assert call["body"]["fallbacks"] == [{"model": ANTHROPIC_FALLBACK_MODEL}]
    assert call["body"]["output_config"] == {"effort": "medium"} and call["body"]["max_tokens"] == 8000
    for forbidden in ("temperature", "top_p", "thinking", "stream"):
        assert forbidden not in call["body"]
    assert output.text == "Interpretación." and output.model == "claude-opus-5-5"
    assert (output.input_tokens, output.output_tokens) == (1200, 380) and not output.truncated


def test_anthropic_never_offers_a_fallback_to_itself():
    payload = {
        "model": ANTHROPIC_FALLBACK_MODEL,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Ok"}],
    }
    client = RecordingClient((200, payload))
    AnthropicAdapter(model_of("anthropic", ANTHROPIC_FALLBACK_MODEL, KEY), client).interpret(PROMPT)
    assert "fallbacks" not in client.calls[0]["body"] and "anthropic-beta" not in client.calls[0]["headers"]


def test_a_rejected_fallback_option_is_retried_without_it_and_remembered():
    ok = {"model": "claude-opus-5-5", "stop_reason": "end_turn", "content": [{"type": "text", "text": "Ok"}]}
    client = RecordingClient((400, {"error": {"message": "unknown parameter: fallbacks"}}), (200, ok), (200, ok))
    adapter = AnthropicAdapter(model_of("anthropic", "claude-opus-5-5", KEY), client)
    assert adapter.interpret(PROMPT).text == "Ok"
    assert "fallbacks" in client.calls[0]["body"] and "fallbacks" not in client.calls[1]["body"]
    adapter.interpret(PROMPT)
    assert "fallbacks" not in client.calls[2]["body"]  # not tried again


def test_anthropic_refusal_truncation_and_empty_answers():
    client = RecordingClient(
        (200, {"stop_reason": "refusal", "content": []}),
        (200, {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "Corte"}]}),
        (200, {"stop_reason": "end_turn", "content": [{"type": "thinking"}]}),
    )
    adapter = AnthropicAdapter(model_of("anthropic", "claude-opus-5-5", KEY), client)
    with pytest.raises(AiUnavailableError, match="declinó"):
        adapter.interpret(PROMPT)
    assert adapter.interpret(PROMPT).truncated
    with pytest.raises(AiUnavailableError, match="no devolvió texto"):
        adapter.interpret(PROMPT)


@pytest.mark.parametrize(
    ("status", "expected", "message"),
    [
        (401, AiUnavailableError, "clave de API"),
        (403, AiUnavailableError, "clave de API"),
        (404, AiUnavailableError, "no encontró el modelo"),
        (429, RateLimitedError, "saturado"),
        (400, AiUnavailableError, "rechazó la solicitud"),
        (500, AiUnavailableError, "no está disponible"),
        (503, AiUnavailableError, "no está disponible"),
    ],
)
def test_http_failures_become_readable_errors_without_the_key(status, expected, message):
    client = RecordingClient((status, {"error": {"message": f"falló con {OPENAI_KEY}"}}))
    adapter = OpenAICompatibleAdapter(model_of("openai", "gpt-5", OPENAI_KEY), client)
    with pytest.raises(expected, match=message) as caught:
        adapter.interpret(PROMPT)
    assert OPENAI_KEY not in str(caught.value)


def test_a_model_without_a_key_is_refused_before_any_request():
    client = RecordingClient()
    with pytest.raises(AiUnavailableError, match="clave de API"):
        OpenAICompatibleAdapter(model_of("openai", "gpt-5"), client).interpret(PROMPT)
    assert client.calls == []


def test_the_connection_test_is_a_tiny_request():
    client = RecordingClient((200, openai_payload(text="OK")))
    output = OpenAICompatibleAdapter(model_of("openai", "gpt-5", OPENAI_KEY), client).test_connection()
    assert output.text == "OK" and client.calls[0]["body"]["max_tokens"] == 1500
    assert client.calls[0]["body"]["messages"][0]["content"].startswith("Responde únicamente")


# ------------------------------------------------------------------ the gateway
class PerModelClient(JsonClient):
    """Answers every URL with an OpenAI-shaped payload that echoes the requested model."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def post(self, url, headers, body, timeout=150.0):
        self.urls.append(url)
        return 200, openai_payload(model=body["model"])


def test_the_gateway_uses_the_default_model_and_the_requested_one(tmp_path):
    ai = settings(tmp_path)
    first = ai.add_model("ollama", "llama3.1", base_url="http://127.0.0.1:11434/v1")
    second = ai.add_model("openai", "gpt-5", key=OPENAI_KEY)

    gateway = ConfiguredGateway(ai, PerModelClient())
    assert gateway.default_model().id == first.id
    assert gateway.interpret(PROMPT).model == "llama3.1"
    assert gateway.interpret(PROMPT, second.id).model == "gpt-5"
    with pytest.raises(InvalidInputError, match="no está configurado"):
        gateway.interpret(PROMPT, "ai_inexistente")


def test_the_gateway_refuses_a_disabled_model_and_says_what_is_missing(tmp_path):
    ai = settings(tmp_path)
    entry = ai.add_model("ollama", "llama3.1")
    gateway = ConfiguredGateway(ai, PerModelClient())
    ai.update_model(entry.id, {"enabled": False})
    with pytest.raises(InvalidInputError, match="desactivado"):
        gateway.interpret(PROMPT, entry.id)
    with pytest.raises(AiUnavailableError, match="no está configurada"):
        gateway.interpret(PROMPT)  # the only model is off, so there is no default left
    empty = ConfiguredGateway(settings(tmp_path, name="empty.db"), RecordingClient())
    with pytest.raises(AiUnavailableError, match="no está configurada"):
        empty.interpret(PROMPT)


def test_environment_backed_providers_appear_as_read_only_models(tmp_path):
    ai = settings(tmp_path, {"ANTHROPIC_API_KEY": KEY, "OPENAI_API_KEY": OPENAI_KEY})
    providers = {model.provider for model in ai.models}
    assert providers == {"anthropic", "openai"} and all(model.read_only for model in ai.models)
    assert ai.describe()["default_server_id"] == "env_anthropic"
    assert next(model for model in ai.models if model.provider == "openai").model == "gpt-5"


def test_usage_log_counts_per_user_and_window(tmp_path):
    db = Database(tmp_path / "u.db")
    a, b = add_user(db, "ana"), add_user(db, "luis")
    usage = SqlAiUsageRepository(db)
    for _ in range(3):
        usage.add(a, "regression", "claude-opus-5-5", 1000, 400, "anthropic")
    usage.add(b, "anova", "gpt-5", 10, 5, "openai")
    assert usage.count_since(a, "2000-01-01T00:00:00") == 3 and usage.count_since(a, "2999-01-01T00:00:00") == 0
    assert (
        usage.oldest_since(a, "2000-01-01T00:00:00") is not None
        and usage.oldest_since(a, "2999-01-01T00:00:00") is None
    )
    summary = {row["username"]: row for row in usage.summary("2000-01-01T00:00:00")}
    assert (
        summary["ana"]["requests"] == 3
        and summary["ana"]["input_tokens"] == 3000
        and summary["luis"]["output_tokens"] == 5
    )


def test_a_stored_key_is_only_readable_as_a_hint():
    described = AiModel(
        id="ai_1",
        provider="ollama",
        label="Ollama",
        model="llama3.1",
        base_url="http://127.0.0.1:11434/v1",
        stored_key=LongKey,
    ).describe()
    assert described["has_key"] and described["key_hint"] == f"…{LongKey[-4:]}"
    assert "stored_key" not in described and LongKey not in json.dumps(described)
    assert described["available"] and described["provider_label"] == "Ollama (local)"


def test_a_short_key_shows_no_hint(tmp_path):
    ai = settings(tmp_path)
    ai.add_model("anthropic", "claude-opus-5-5", key="corta")
    described = json.dumps(ai.describe())
    assert "corta" not in described and "…" not in described


@pytest.mark.live
def test_real_call_when_a_key_is_available(tmp_path):
    """Opt-in: ``pytest -m live`` with ANTHROPIC_API_KEY set (it costs a few cents)."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY no está definido")
    ai = settings(tmp_path)
    ai.add_model("anthropic", "claude-opus-5-5", key=os.environ["ANTHROPIC_API_KEY"])
    output = ConfiguredGateway(ai).test_connection()
    assert output.text and output.output_tokens > 0
