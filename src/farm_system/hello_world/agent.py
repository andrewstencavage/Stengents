"""Standard ADK hello-world agent backed by the development-time gym model."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from stengents.model_source import resolve_model


MODEL = resolve_model("llama3.1:8b")


root_agent = LlmAgent(
    name="hello_world",
    description="Responds with a friendly greeting from the farm system.",
    model=MODEL.llm,
    instruction="You are the farm system's hello-world agent. Respond to the user with one friendly sentence.",
)
