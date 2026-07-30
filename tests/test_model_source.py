from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request

import pytest
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from stengents.utilities.model_source import ModelConnection, ModelSourceUnavailable, resolve_model


# --- resolution ---------------------------------------------------------

def test_resolve_model_defaults_when_nothing_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for suffix in ("NAME", "BASE_URL", "API_KEY"):
        monkeypatch.delenv(f"STENGENTS_MODEL_{suffix}", raising=False)

    connection = resolve_model("llama3.1:8b")

    assert connection == ModelConnection("llama3.1:8b", "http://127.0.0.1:11434", "local")


def test_resolve_model_reads_the_shared_env_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENGENTS_MODEL_NAME", "qwen2.5:7b-8k")
    monkeypatch.setenv("STENGENTS_MODEL_BASE_URL", "http://gym:11434")
    monkeypatch.setenv("STENGENTS_MODEL_API_KEY", "secret")

    connection = resolve_model("llama3.1:8b")

    assert connection == ModelConnection("qwen2.5:7b-8k", "http://gym:11434", "secret")


def test_explicit_name_beats_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENGENTS_MODEL_NAME", "from-env")

    assert resolve_model("default", name="from-flag").name == "from-flag"


def test_no_selection_yields_an_empty_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STENGENTS_MODEL_NAME", raising=False)

    assert resolve_model("", name=None).name == ""


def test_farm_scope_env_is_no_longer_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STENGENTS_MODEL_NAME", raising=False)
    monkeypatch.setenv("FARM_SYSTEM_MODEL_NAME", "farm-only")

    assert resolve_model("default").name == "default"


# --- credential-free record ---------------------------------------------

def test_as_record_is_credential_free() -> None:
    record = ModelConnection("qwen2.5:7b-8k", "http://gym:11434", "secret").as_record()

    assert record == {"provider": "openai-compatible", "name": "qwen2.5:7b-8k"}
    assert "base_url" not in record and "api_key" not in record and "secret" not in record.values()


def test_resolve_model_builds_a_google_ai_studio_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENGENTS_MODEL_PROVIDER", "google-ai-studio")
    monkeypatch.setenv("STENGENTS_MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    connection = resolve_model("")

    assert connection.provider == "google-ai-studio"
    model = connection.llm
    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"
    assert connection.as_record() == {
        "provider": "google-ai-studio",
        "name": "gemini-2.5-flash",
    }
    assert "gemini-secret" not in connection.as_record().values()


def test_openai_compatible_connection_keeps_its_litellm_adapter() -> None:
    connection = ModelConnection("qwen2.5:7b-8k", "http://gym:11434", "secret")

    model = connection.llm

    assert isinstance(model, LiteLlm)
    assert model.model == "openai/qwen2.5:7b-8k"


def test_gemini_preflight_checks_only_for_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENGENTS_MODEL_PROVIDER", "google-ai-studio")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    connection = resolve_model("gemini-2.5-flash")

    def opener(*_args, **_kwargs):
        raise AssertionError("Gemini preflight must not make a request")

    assert connection.preflight(opener=opener) is None


def test_resolve_model_reads_the_rate_limit_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENGENTS_RATE_LIMIT_ON_RATE_LIMIT", "fail")
    monkeypatch.setenv("STENGENTS_RATE_LIMIT_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STENGENTS_RATE_LIMIT_MAX_CUMULATIVE_WAIT_SECONDS", "15")

    connection = resolve_model("llama3.1:8b")

    assert connection.rate_limit_policy.as_record() == {
        "on_rate_limit": "fail",
        "max_attempts": 2,
        "max_cumulative_wait_seconds": 15,
        "paid_fallback": False,
    }


# --- preflight (driven through a fake opener, no network) ----------------

class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _opener(*, models=None, models_error=None, chats=None, chat_error=None):
    chat_iter = iter(chats or [])

    def opener(target, timeout=None):
        url = target.full_url if isinstance(target, Request) else target
        if url.endswith("/v1/models"):
            if models_error is not None:
                raise models_error
            return _Response({"data": models or []})
        if url.endswith("/v1/chat/completions"):
            if chat_error is not None:
                raise chat_error
            return _Response(next(chat_iter))
        raise AssertionError(f"unexpected url: {url}")

    return opener


_CONNECTION = ModelConnection("qwen2.5:7b-8k", "http://gym:11434", "local")


def test_preflight_rejects_an_unreachable_endpoint() -> None:
    opener = _opener(models_error=URLError("refused"))

    with pytest.raises(ModelSourceUnavailable, match="model_endpoint_unavailable: URLError"):
        _CONNECTION.preflight(opener=opener)


def test_preflight_rejects_a_missing_model() -> None:
    opener = _opener(models=[{"id": "some-other-model"}])

    with pytest.raises(ModelSourceUnavailable, match="configured_model_missing"):
        _CONNECTION.preflight(opener=opener)


def test_preflight_rejects_a_model_without_tool_calls() -> None:
    opener = _opener(
        models=[{"id": "qwen2.5:7b-8k"}],
        chats=[{"choices": [{"message": {"content": "hi"}}]}] * 3,
    )

    with pytest.raises(ModelSourceUnavailable, match="tool_call_incompatible: ValueError"):
        _CONNECTION.preflight(opener=opener)


def test_preflight_passes_when_the_model_calls_a_tool() -> None:
    opener = _opener(
        models=[{"id": "qwen2.5:7b-8k"}],
        chats=[{"choices": [{"message": {"tool_calls": [{"function": {"name": "list_files"}}]}}]}],
    )

    assert _CONNECTION.preflight(opener=opener) is None


# --- complete (driven through a fake completion callable, no network) ---


def _reply(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_complete_calls_the_openai_compatible_endpoint() -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _reply("hi")

    result = _CONNECTION.complete("system prompt", "user prompt", completion=completion)

    assert result == "hi"
    assert calls == [
        {
            "model": "openai/qwen2.5:7b-8k",
            "api_base": "http://gym:11434/v1",
            "api_key": "local",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
    ]


def test_complete_honors_the_response_format_parameter() -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _reply("hi")

    _CONNECTION.complete("system prompt", "user prompt", response_format="text", completion=completion)

    assert calls[0]["response_format"] == {"type": "text"}


def _gemini_connection() -> ModelConnection:
    return ModelConnection("gemini-2.5-flash", "unused", "gemini-secret", resolved_provider="google-ai-studio")


def test_complete_retries_a_transient_gemini_error_then_succeeds() -> None:
    from litellm import InternalServerError

    attempts = []
    sleeps = []

    def completion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise InternalServerError(message="high demand", llm_provider="gemini", model=kwargs["model"])
        return _reply("ok")

    result = _gemini_connection().complete(
        "system prompt", "user prompt", completion=completion, sleep=sleeps.append
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert attempts[0]["model"] == "gemini/gemini-2.5-flash"
    assert "api_base" not in attempts[0]
    assert sleeps == [2.0, 4.0]


def test_complete_gives_up_after_five_gemini_attempts() -> None:
    from litellm import ServiceUnavailableError

    attempts = []

    def completion(**kwargs):
        attempts.append(kwargs)
        raise ServiceUnavailableError(message="overloaded", llm_provider="gemini", model=kwargs["model"])

    with pytest.raises(ServiceUnavailableError):
        _gemini_connection().complete("system prompt", "user prompt", completion=completion, sleep=lambda _seconds: None)

    assert len(attempts) == 5


def test_complete_does_not_retry_a_non_transient_gemini_error() -> None:
    def completion(**kwargs):
        raise ValueError("not a transient error")

    with pytest.raises(ValueError, match="not a transient error"):
        _gemini_connection().complete("system prompt", "user prompt", completion=completion, sleep=lambda _seconds: None)


# --- complete's quota-aware retry (shares rate_limit.py's classifier) ---


def test_complete_does_not_retry_a_per_day_quota_error() -> None:
    from litellm import RateLimitError

    attempts = []
    sleeps = []

    def completion(**kwargs):
        attempts.append(kwargs)
        raise RateLimitError(
            message="429 RESOURCE_EXHAUSTED: quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            llm_provider="gemini",
            model=kwargs["model"],
        )

    with pytest.raises(RateLimitError):
        _gemini_connection().complete("system prompt", "user prompt", completion=completion, sleep=sleeps.append)

    assert len(attempts) == 1
    assert sleeps == []


def test_complete_waits_the_reported_retry_delay_for_a_per_minute_quota_error() -> None:
    from litellm import RateLimitError

    attempts = []
    sleeps = []

    def completion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 2:
            raise RateLimitError(
                message=(
                    "429 RESOURCE_EXHAUSTED: quotaId "
                    "GenerateRequestsPerMinutePerProjectPerModel-FreeTier; retryDelay 12s"
                ),
                llm_provider="gemini",
                model=kwargs["model"],
            )
        return _reply("ok")

    result = _gemini_connection().complete(
        "system prompt", "user prompt", completion=completion, sleep=sleeps.append
    )

    assert result == "ok"
    assert sleeps == [12.0]


def test_complete_gives_up_when_the_reported_retry_delay_exceeds_the_cumulative_cap() -> None:
    from litellm import RateLimitError

    attempts = []

    def completion(**kwargs):
        attempts.append(kwargs)
        raise RateLimitError(
            message="429 RESOURCE_EXHAUSTED: quotaId GenerateRequestsPerMinute; retryDelay 9999s",
            llm_provider="gemini",
            model=kwargs["model"],
        )

    with pytest.raises(RateLimitError):
        _gemini_connection().complete(
            "system prompt", "user prompt", completion=completion, sleep=lambda _seconds: None
        )

    assert len(attempts) == 1
