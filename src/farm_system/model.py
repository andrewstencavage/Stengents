"""Shared resolution of the development-time model connection for farm agents.

Every farm agent talks to the same loopback model endpoint on `gym`; only the
model name legitimately differs per agent (hello_world is happy on a small
model, kiln_coach wants a larger tool-calling one). Each agent passes its own
default model name; the stable connection and the override precedence live here
in one place.

Precedence for every field: ``FARM_SYSTEM_MODEL_*`` (agent scope) ->
``STENGENTS_MODEL_*`` (shared harness scope) -> the fallback. The model name's
fallback is the per-agent default the caller supplies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google.adk.models.lite_llm import LiteLlm


def _resolve(suffix: str, fallback: str) -> str:
    return os.environ.get(
        f"FARM_SYSTEM_MODEL_{suffix}",
        os.environ.get(f"STENGENTS_MODEL_{suffix}", fallback),
    )


@dataclass(frozen=True)
class ModelConnection:
    """A resolved connection to one model on the development-time endpoint."""

    name: str
    base_url: str
    api_key: str

    @property
    def llm(self) -> LiteLlm:
        return LiteLlm(
            model=f"openai/{self.name}",
            api_base=f"{self.base_url.rstrip('/')}/v1",
            api_key=self.api_key,
        )


def resolve_model(default_name: str) -> ModelConnection:
    """Resolve the model connection, defaulting the name to ``default_name``."""

    return ModelConnection(
        name=_resolve("NAME", default_name),
        base_url=_resolve("BASE_URL", "http://127.0.0.1:11434"),
        api_key=_resolve("API_KEY", "local"),
    )
