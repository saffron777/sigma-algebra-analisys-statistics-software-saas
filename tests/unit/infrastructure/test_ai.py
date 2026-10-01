from types import SimpleNamespace

import anthropic
import pytest

from sigma.domain.errors import AiUnavailableError, InvalidInputError, RateLimitedError
from sigma.domain.interpretation import Prompt
from sigma.infrastructure.ai.anthropic_gateway import FALLBACK_BETA, FALLBACK_MODEL, AnthropicGateway
from sigma.infrastructure.ai.settings import DEFAULT_MODEL, AiSettings
from sigma.infrastructure.persistence.sqlite.ai_repositories import SqliteAiUsageRepository, SqliteSettingsRepository
from sigma.infrastructure.persistence.sqlite.database import Database
from tests.support import add_user

try:  # recent SDKs ship their own HTTP client package
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

KEY = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
PROMPT = Prompt(system="sistema", user="usuario")


def settings(tmp_path, environ=None, name="s.db"):
    return AiSettings(
        SqliteSettingsRepository(Database(tmp_path / name)), tmp_path, environ if environ is not None else {}
    )


# ------------------------------------------------------------------ settings
def test_key_priority_masking_and_storage(tmp_path):
    ai = settings(tmp_path)
    assert not ai.configured and ai.key_source is None and ai.describe()["key_hint"] is None
    ai.set_key(f"  {KEY}\n")
    assert ai.configured and ai.key_source == "file" and ai.api_key == KEY
    described = ai.describe()
    assert described["key_hint"] == "…1234" and KEY not in str(described)
    env = settings(tmp_path, {"ANTHROPIC_API_KEY": "sk-ant-from-the-environment-0000"})
    assert env.key_source == "env" and env.api_key == "sk-ant-from-the-environment-0000"  # the environment wins
    ai.clear_key()
    ai.clear_key()  # clearing twice is fine
    assert not ai.configured


@pytest.mark.parametrize(
    "bad", ["", "   ", "abc", "sk-ant-corta", "sk-openai-" + "a" * 30, "sk-ant-con espacio-" + "a" * 20]
)
def test_key_format_is_validated(tmp_path, bad):
    with pytest.raises(InvalidInputError):
        settings(tmp_path).set_key(bad)


def test_model_and_switch_are_persisted(tmp_path):
    ai = settings(tmp_path)
    assert ai.model == DEFAULT_MODEL == "claude-opus-5-5" and ai.enabled
    ai.set_model("claude-sonnet-5-5")
    ai.set_enabled(False)
    again = settings(tmp_path)  # a new instance reads the same database
    assert again.model == "claude-sonnet-5-5" and not again.enabled
    for bad in ("gpt-4", "claude", "claude-OPUS", "claude-opus; drop", ""):
        with pytest.raises(InvalidInputError):
            ai.set_model(bad)
    assert settings(tmp_path, {"SIGMA_AI_MODEL": "claude-haiku-4-5"}, name="other.db").model == "claude-haiku-4-5"


def test_usage_log_counts_per_user_and_window(tmp_path):
    db = Database(tmp_path / "u.db")
    a, b = add_user(db, "ana"), add_user(db, "luis")
    usage = SqliteAiUsageRepository(db)
    for _ in range(3):
        usage.add(a, "regression", "claude-opus-5-5", 1000, 400)
    usage.add(b, "anova", "claude-opus-5-5", 10, 5)
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


# ------------------------------------------------------------------ the Anthropic adapter (fake client)
class FakeMessages:
    def __init__(self, behaviours):
        self.calls = []
        self._behaviours = list(behaviours)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        behaviour = self._behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


def fake_client(*behaviours):
    messages = FakeMessages(behaviours)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def response(text="Interpretación.", stop="end_turn", model="claude-opus-5-5", extra_blocks=(), usage=(1200, 380)):
    blocks = [
        SimpleNamespace(type="thinking", thinking="razonamiento interno"),
        *extra_blocks,
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        model=model,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1], iterations=None),
    )


def gateway(tmp_path, client, **kwargs):
    ai = settings(tmp_path)
    ai.set_key(KEY)
    return AnthropicGateway(ai, client_factory=lambda key: client), ai


def http_error(cls, status, message):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def test_request_shape_for_opus_5_5(tmp_path):
    client, messages = fake_client(response())
    output = gateway(tmp_path, client)[0].interpret(PROMPT)
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["max_tokens"] == 8000
    assert call["system"] == "sistema" and call["messages"] == [{"role": "user", "content": "usuario"}]
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == [{"model": FALLBACK_MODEL}]
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "stream"):
        assert forbidden not in call  # Opus 5.5 always thinks and takes no sampling parameters
    assert output.text == "Interpretación." and output.model == "claude-opus-5-5"  # the thinking block is not shown
    assert (
        (output.input_tokens, output.output_tokens) == (1200, 380) and not output.fallback_used and not output.truncated
    )


def test_the_configured_model_is_used_and_no_fallback_to_itself(tmp_path):
    ai = settings(tmp_path)
    ai.set_key(KEY)
    ai.set_model(FALLBACK_MODEL)
    client, messages = fake_client(response(model=FALLBACK_MODEL))
    AnthropicGateway(ai, client_factory=lambda key: client).interpret(PROMPT)
    assert messages.calls[0]["model"] == FALLBACK_MODEL and "fallbacks" not in messages.calls[0]


def test_fallback_is_reported_and_truncation_flagged(tmp_path):
    marker = SimpleNamespace(type="fallback")
    client, _ = fake_client(response(stop="max_tokens", extra_blocks=[marker], model="claude-opus-4-8"))
    output = gateway(tmp_path, client)[0].interpret(PROMPT)
    assert output.fallback_used and output.truncated and output.model == "claude-opus-4-8"


def test_a_rejected_fallback_beta_is_retried_without_it_and_remembered(tmp_path):
    client, messages = fake_client(
        http_error(anthropic.BadRequestError, 400, "unknown parameter: fallbacks"), response(), response()
    )
    adapter = gateway(tmp_path, client)[0]
    assert adapter.interpret(PROMPT).text == "Interpretación."
    assert "fallbacks" in messages.calls[0] and "fallbacks" not in messages.calls[1]
    adapter.interpret(PROMPT)
    assert "fallbacks" not in messages.calls[2]  # not tried again


def test_other_bad_requests_become_a_readable_error(tmp_path):
    client, _ = fake_client(http_error(anthropic.BadRequestError, 400, "max_tokens too large"))
    with pytest.raises(AiUnavailableError, match="rechazó la solicitud"):
        gateway(tmp_path, client)[0].interpret(PROMPT)


def test_refusals_and_empty_answers(tmp_path):
    client, _ = fake_client(
        response(stop="refusal"),
        SimpleNamespace(content=[SimpleNamespace(type="thinking")], stop_reason="end_turn", model="m", usage=None),
    )
    adapter = gateway(tmp_path, client)[0]
    with pytest.raises(AiUnavailableError, match="declinó"):
        adapter.interpret(PROMPT)
    with pytest.raises(AiUnavailableError, match="no devolvió texto"):
        adapter.interpret(PROMPT)


@pytest.mark.parametrize(
    ("error", "expected", "message"),
    [
        (
            http_error(anthropic.AuthenticationError, 401, "invalid x-api-key " + KEY),
            AiUnavailableError,
            "no es válida",
        ),
        (http_error(anthropic.PermissionDeniedError, 403, "no access"), AiUnavailableError, "no tiene permiso"),
        (http_error(anthropic.RateLimitError, 429, "slow down"), RateLimitedError, "saturado"),
        (http_error(anthropic.InternalServerError, 529, "overloaded"), AiUnavailableError, "no está disponible"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), AiUnavailableError, "conectar"),
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), AiUnavailableError, "conectar"),
    ],
)
def test_errors_are_translated_and_never_leak_the_key(tmp_path, error, expected, message):
    client, _ = fake_client(error)
    with pytest.raises(expected, match=message) as caught:
        gateway(tmp_path, client)[0].interpret(PROMPT)
    assert KEY not in str(caught.value) and "ABCDEFGH" not in str(caught.value)


def test_without_a_key_the_call_is_refused_before_any_request(tmp_path):
    client, messages = fake_client()
    adapter = AnthropicGateway(settings(tmp_path), client_factory=lambda key: client)
    with pytest.raises(AiUnavailableError, match="falta la clave"):
        adapter.interpret(PROMPT)
    assert messages.calls == []


def test_connection_test_is_a_tiny_request(tmp_path):
    client, messages = fake_client(response(text="OK"))
    output = gateway(tmp_path, client)[0].test_connection()
    assert output.text == "OK" and messages.calls[0]["max_tokens"] == 1500 and "OK" in messages.calls[0]["system"]


@pytest.mark.live
def test_real_call_when_a_key_is_available(tmp_path):
    """Opt-in: ``pytest -m live`` with ANTHROPIC_API_KEY set (it costs a few cents)."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY no está definido")
    ai = AiSettings(SqliteSettingsRepository(Database(tmp_path / "live.db")), tmp_path)
    output = AnthropicGateway(ai).test_connection()
    assert output.text and output.output_tokens > 0
