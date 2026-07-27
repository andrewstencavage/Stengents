"""The development-time model source: one resolved connection to one model.

Every agent and the coding harness resolve one portable model connection. The
development default is the loopback OpenAI-compatible endpoint on `gym`; Google
AI Studio uses ADK's native Gemini adapter. Each caller passes its own default
model name; the stable connection, provider identity, adapter construction, and
provider-aware preflight all live here in one place.

Resolution precedence for every field: ``STENGENTS_MODEL_*`` -> the fallback.
The model name's fallback is the per-caller default the caller supplies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from google.adk.models.lite_llm import LiteLlm
from google.adk.models import Gemini

from .rate_limit import RateLimitPolicy

OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
GOOGLE_AI_STUDIO_PROVIDER = "google-ai-studio"
_PROVIDERS = frozenset({OPENAI_COMPATIBLE_PROVIDER, GOOGLE_AI_STUDIO_PROVIDER})

# A minimal urlopen-shaped opener: called with (url_or_request, timeout) and
# returning a context manager whose value is read as the HTTP response body.
Opener = Callable[..., object]


class ModelSourceUnavailable(RuntimeError):
    """The development-time model source could not be validated for a run."""


def _resolve(suffix: str, fallback: str) -> str:
    return os.environ.get(f"STENGENTS_MODEL_{suffix}", fallback)


def _resolve_rate_limit_policy() -> RateLimitPolicy:
    return RateLimitPolicy(
        on_rate_limit=os.environ.get("STENGENTS_RATE_LIMIT_ON_RATE_LIMIT", "wait"),
        max_attempts=int(os.environ.get("STENGENTS_RATE_LIMIT_MAX_ATTEMPTS", "1")),
        max_cumulative_wait_seconds=int(os.environ.get("STENGENTS_RATE_LIMIT_MAX_CUMULATIVE_WAIT_SECONDS", "30")),
        paid_fallback=os.environ.get("STENGENTS_RATE_LIMIT_PAID_FALLBACK", "false").lower() == "true",
    )


@dataclass(frozen=True)
class ModelConnection:
    """A resolved connection to one model on the development-time endpoint."""

    name: str
    base_url: str
    api_key: str
    resolved_provider: str = OPENAI_COMPATIBLE_PROVIDER
    rate_limit_policy: RateLimitPolicy = field(default_factory=RateLimitPolicy)

    @property
    def provider(self) -> str:
        return self.resolved_provider

    @property
    def llm(self) -> LiteLlm | Gemini:
        if self.provider == GOOGLE_AI_STUDIO_PROVIDER:
            return Gemini(model=self.name)
        return LiteLlm(
            model=f"openai/{self.name}",
            api_base=f"{self.base_url.rstrip('/')}/v1",
            api_key=self.api_key,
        )

    def as_record(self) -> dict[str, str]:
        """The credential-free model fragment for a run announcement or record."""
        return {"provider": self.provider, "name": self.name}

    def preflight(self, *, opener: Opener = urlopen) -> None:
        """Validate the endpoint, the selected model, and tool-call support.

        Raises ``ModelSourceUnavailable`` with a stable machine-readable reason
        if any check fails. ``opener`` is injectable so this is testable without
        a live endpoint.
        """
        if self.provider == GOOGLE_AI_STUDIO_PROVIDER:
            if not self.api_key:
                raise ModelSourceUnavailable("gemini_api_key_missing")
            return
        root = self.base_url.rstrip("/")
        try:
            with opener(f"{root}/v1/models", timeout=5) as response:
                models = json.load(response).get("data", [])
        except (URLError, TimeoutError, OSError) as error:
            raise ModelSourceUnavailable(f"model_endpoint_unavailable: {type(error).__name__}") from error
        if self.name not in {entry.get("id") for entry in models}:
            raise ModelSourceUnavailable("configured_model_missing")
        payload = json.dumps({
            "model": self.name,
            "messages": [{"role": "user", "content": "Call list_files."}],
            "tools": [{"type": "function", "function": {"name": "list_files", "description": "list files", "parameters": {"type": "object", "properties": {}}}}],
            "tool_choice": "required",
        }).encode()
        request = Request(f"{root}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
        last_error: Exception | None = None
        for _ in range(3):
            try:
                with opener(request, timeout=30) as response:
                    result = json.load(response)
                if result["choices"][0]["message"].get("tool_calls"):
                    return
                last_error = ValueError("no tool call")
            except (URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as error:
                last_error = error
        assert last_error is not None
        raise ModelSourceUnavailable(f"tool_call_incompatible: {type(last_error).__name__}") from last_error


def resolve_model(default_name: str, *, name: str | None = None) -> ModelConnection:
    """Resolve the model connection to one model on the endpoint.

    Name precedence is an explicit ``name`` (e.g. a CLI ``--model`` flag) ->
    ``STENGENTS_MODEL_NAME`` -> ``default_name``. Pass ``default_name=""`` when
    there is no sensible default (the coding agent), so a caller can detect a
    missing selection via an empty ``ModelConnection.name``.
    """

    provider = _resolve("PROVIDER", OPENAI_COMPATIBLE_PROVIDER)
    if provider not in _PROVIDERS:
        raise ValueError(f"unsupported_model_provider: {provider}")
    api_key = os.environ.get("GEMINI_API_KEY", "") if provider == GOOGLE_AI_STUDIO_PROVIDER else _resolve("API_KEY", "local")
    return ModelConnection(
        name=name or _resolve("NAME", default_name),
        base_url=_resolve("BASE_URL", "http://127.0.0.1:11434"),
        api_key=api_key,
        resolved_provider=provider,
        rate_limit_policy=_resolve_rate_limit_policy(),
    )
