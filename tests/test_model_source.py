from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request

import pytest

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
    assert connection.llm.model == "gemini/gemini-2.5-flash"
    assert connection.as_record() == {
        "provider": "google-ai-studio",
        "name": "gemini-2.5-flash",
    }
    assert "gemini-secret" not in connection.as_record().values()


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
