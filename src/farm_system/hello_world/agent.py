"""Standard ADK hello-world agent backed by the development-time gym model."""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


MODEL_BASE_URL = os.environ.get(
    "FARM_SYSTEM_MODEL_BASE_URL",
    os.environ.get("STENGENTS_MODEL_BASE_URL", "http://127.0.0.1:11434"),
)
MODEL_NAME = os.environ.get(
    "FARM_SYSTEM_MODEL_NAME",
    os.environ.get("STENGENTS_MODEL_NAME", "llama3.1:8b"),
)
MODEL_API_KEY = os.environ.get(
    "FARM_SYSTEM_MODEL_API_KEY",
    os.environ.get("STENGENTS_MODEL_API_KEY", "local"),
)


root_agent = LlmAgent(
    name="hello_world",
    description="Responds with a friendly greeting from the farm system.",
    model=LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=f"{MODEL_BASE_URL.rstrip('/')}/v1",
        api_key=MODEL_API_KEY,
    ),
    instruction="You are the farm system's hello-world agent. Respond to the user with one friendly sentence.",
)
