"""ADK implementation of the evaluated coding agent."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from ..harness import Actions, Fixture, RunBudgetExceeded


class RunCapturePlugin:
    """Capture ADK lifecycle events in the active harness run."""

    def __init__(self, actions: Actions) -> None:
        from google.adk.plugins.base_plugin import BasePlugin

        class Plugin(BasePlugin):
            async def before_run_callback(self, *, invocation_context: object) -> None:
                actions.adk_invocation_id = str(getattr(invocation_context, "invocation_id"))

            async def before_tool_callback(self, *, tool: object, tool_args: dict[str, object], tool_context: object) -> None:
                actions.adk_lifecycle_events.append({"name": str(getattr(tool, "name", "unknown")), "phase": "before"})

            async def after_tool_callback(self, *, tool: object, tool_args: dict[str, object], tool_context: object, result: dict[str, object]) -> None:
                actions.adk_lifecycle_events.append({"name": str(getattr(tool, "name", "unknown")), "phase": "after"})

            async def on_tool_error_callback(self, *, tool: object, tool_args: dict[str, object], tool_context: object, error: Exception) -> None:
                actions.adk_lifecycle_events.append({"name": str(getattr(tool, "name", "unknown")), "phase": "error"})

        self.plugin = Plugin(name="stengents_run_capture")

    def __getattr__(self, name: str) -> object:
        return getattr(self.plugin, name)


def _agent_instruction(fixture: Fixture) -> str:
    editable_paths = ", ".join(fixture.source_surface)
    return (
        f"Repair coding fixture {fixture.identifier}. The only editable source files are: {editable_paths}. "
        "Tests are immutable and must never be written. The defect is an upper-bound index check: an index equal "
        "to the item count must raise IndexError before attempting item access. Start by calling list_files, then "
        "inspect the relevant source with read_file before editing. Use only named actions, run tests after your "
        "repair, and do not use placeholder paths."
    )


def _required_discovery_tool(actions: Actions) -> str | None:
    completed_actions = {event["name"] for event in actions.events if event["outcome"] == "ok"}
    if "list_files" not in completed_actions:
        return "list_files"
    if "read_file" not in completed_actions:
        return "read_file"
    return None


def adk_driver(*, base_url: str, model_name: str, api_key: str) -> Callable[[Actions], None]:
    """Create the coding agent over the portable LiteLLM adapter."""
    def drive(actions: Actions) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        model = LiteLlm(model=f"openai/{model_name}", api_base=f"{base_url.rstrip('/')}/v1", api_key=api_key)

        def require_discovery_tool(*, callback_context: object, llm_request: object) -> None:
            required_tool = _required_discovery_tool(actions)
            if required_tool is None:
                model._additional_args.pop("tool_choice", None)
            else:
                model._additional_args["tool_choice"] = {"type": "function", "function": {"name": required_tool}}

        agent = LlmAgent(
            name="coding_agent",
            model=model,
            instruction=_agent_instruction(actions.fixture),
            tools=[actions.list_files, actions.read_file, actions.write_source_file, actions.run_tests],
            before_model_callback=require_discovery_tool,
        )

        async def invoke() -> None:
            sessions = InMemorySessionService()
            session = await sessions.create_session(app_name="stengents", user_id="local", session_id="run")
            runner = Runner(app_name="stengents", agent=agent, session_service=sessions, plugins=[RunCapturePlugin(actions).plugin])
            async for _ in runner.run_async(user_id="local", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text="Repair the failing fixture.")])):
                pass

        remaining = max(1, actions.budget.elapsed_seconds - (time.monotonic() - actions.started))
        try:
            asyncio.run(asyncio.wait_for(invoke(), timeout=remaining))
        except TimeoutError as error:
            raise RunBudgetExceeded("run elapsed-time budget exhausted") from error

    return drive
