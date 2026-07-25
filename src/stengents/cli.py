from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .coding_agent import adk_driver
from .harness import Fixture, run_fixture


def _fixture(identifier: str) -> Fixture:
    if identifier != "normalize-index":
        raise ValueError(f"unknown fixture: {identifier}")
    return Fixture(identifier, Path(__file__).parent / "fixtures" / identifier, ("normalize_index.py",), (sys.executable, "-m", "pytest", "-q"))


def _preflight(base_url: str, model: str) -> None:
    try:
        with urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=5) as response:
            models = json.load(response).get("data", [])
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"model_endpoint_unavailable: {type(error).__name__}") from error
    if model not in {entry.get("id") for entry in models}:
        raise RuntimeError("configured_model_missing")
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "Call list_files."}], "tools": [{"type": "function", "function": {"name": "list_files", "description": "list files", "parameters": {"type": "object", "properties": {}}}}], "tool_choice": "required"}).encode()
    request = Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            if result["choices"][0]["message"].get("tool_calls"):
                return
            last_error = ValueError("no tool call")
        except (URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as error:
            last_error = error
    assert last_error is not None
    raise RuntimeError(f"tool_call_incompatible: {type(last_error).__name__}") from last_error


_USAGE = "usage: stengents run <fixture-id> [--model <name>]"


def _parse(arguments: list[str]) -> tuple[list[str], str | None]:
    positional: list[str] = []
    model_override: str | None = None
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in ("--model", "-m"):
            index += 1
            if index >= len(arguments):
                raise ValueError("--model requires a value")
            model_override = arguments[index]
        elif token.startswith("--model="):
            model_override = token[len("--model=") :]
        else:
            positional.append(token)
        index += 1
    return positional, model_override


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        positional, model_override = _parse(arguments)
    except ValueError:
        print(_USAGE, file=sys.stderr)
        return 2
    if len(positional) != 2 or positional[0] != "run":
        print(_USAGE, file=sys.stderr)
        return 2
    base_url = os.environ.get("STENGENTS_MODEL_BASE_URL")
    model = model_override or os.environ.get("STENGENTS_MODEL_NAME")
    if not base_url or not model:
        print("preflight failed: model_endpoint_unavailable; detail=STENGENTS_MODEL_BASE_URL and a model (via --model or STENGENTS_MODEL_NAME) are required", file=sys.stderr)
        return 2
    try:
        fixture = _fixture(positional[1])
        _preflight(base_url, model)
    except RuntimeError as error:
        print(f"preflight failed: {error}; endpoint={base_url}; model={model}", file=sys.stderr)
        return 2
    run_directory = Path(".stengents/runs")
    run_id = str(uuid.uuid4())
    record_path = run_directory / f"{run_id}.json"
    print(json.dumps({"run_id": run_id, "fixture_id": fixture.identifier, "model": {"provider": "openai-compatible", "name": model}, "action_limit": 25, "elapsed_time_limit_seconds": 300, "record_path": str(record_path)}))
    record_path, exit_code = run_fixture(fixture, run_directory=run_directory, model_name=model, agent_driver=adk_driver(base_url=base_url, model_name=model, api_key=os.environ.get("STENGENTS_MODEL_API_KEY", "local")), run_id=run_id)
    print(json.dumps({"record_path": str(record_path), "outcome": "passed" if exit_code == 0 else "failed"}))
    return exit_code
