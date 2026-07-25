from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from .coding_agent import adk_driver
from .harness import Fixture, run_fixture
from .model_source import ModelSourceUnavailable, resolve_model


def _fixture(identifier: str) -> Fixture:
    if identifier != "normalize-index":
        raise ValueError(f"unknown fixture: {identifier}")
    return Fixture(identifier, Path(__file__).parent / "fixtures" / identifier, ("normalize_index.py",), (sys.executable, "-m", "pytest", "-q"))


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
    connection = resolve_model("", name=model_override)
    if not connection.name:
        print("preflight failed: configured_model_missing; detail=a model is required via --model or STENGENTS_MODEL_NAME", file=sys.stderr)
        return 2
    try:
        fixture = _fixture(positional[1])
        connection.preflight()
    except (ValueError, ModelSourceUnavailable) as error:
        print(f"preflight failed: {error}; endpoint={connection.base_url}; model={connection.name}", file=sys.stderr)
        return 2
    run_directory = Path(".stengents/runs")
    run_id = str(uuid.uuid4())
    record_path = run_directory / f"{run_id}.json"
    print(json.dumps({"run_id": run_id, "fixture_id": fixture.identifier, "model": connection.as_record(), "action_limit": 25, "elapsed_time_limit_seconds": 300, "record_path": str(record_path)}))
    record_path, exit_code = run_fixture(fixture, run_directory=run_directory, model=connection.as_record(), agent_driver=adk_driver(connection), run_id=run_id)
    print(json.dumps({"record_path": str(record_path), "outcome": "passed" if exit_code == 0 else "failed"}))
    return exit_code
