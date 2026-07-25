"""The development-time model source: one resolved connection to one model.

Every agent and the coding harness talk to the same loopback model endpoint on
`gym` over an OpenAI-compatible API; only the model name legitimately differs
per caller (hello_world is happy on a small model, kiln_coach and the coding
agent want a larger tool-calling one). Each caller passes its own default model
name; the stable connection, the provider identity, the LiteLLM adapter, and the
endpoint preflight all live here in one place.

Resolution precedence for every field: ``STENGENTS_MODEL_*`` -> the fallback.
The model name's fallback is the per-caller default the caller supplies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from google.adk.models.lite_llm import LiteLlm

PROVIDER = "openai-compatible"

# A minimal urlopen-shaped opener: called with (url_or_request, timeout) and
# returning a context manager whose value is read as the HTTP response body.
Opener = Callable[..., object]


class ModelSourceUnavailable(RuntimeError):
    """The development-time model source could not be validated for a run."""


def _resolve(suffix: str, fallback: str) -> str:
    return os.environ.get(f"STENGENTS_MODEL_{suffix}", fallback)


@dataclass(frozen=True)
class ModelConnection:
    """A resolved connection to one model on the development-time endpoint."""

    name: str
    base_url: str
    api_key: str

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def llm(self) -> LiteLlm:
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

    return ModelConnection(
        name=name or _resolve("NAME", default_name),
        base_url=_resolve("BASE_URL", "http://127.0.0.1:11434"),
        api_key=_resolve("API_KEY", "local"),
    )
