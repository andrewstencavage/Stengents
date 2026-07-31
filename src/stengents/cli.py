from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .coding_agent import adk_driver
from .harness import Fixture, run_fixture
from .utilities.model_source import ModelSourceUnavailable, resolve_model
from .run_record import RunOutcome


def _fixture(identifier: str) -> Fixture:
    if identifier != "normalize-index":
        raise ValueError(f"unknown fixture: {identifier}")
    return Fixture(identifier, Path(__file__).parent / "fixtures" / identifier, ("normalize_index.py",), (sys.executable, "-m", "pytest", "-q"))


_USAGE = (
    "usage: stengents run <fixture-id> [--model <name>]\n"
    "       stengents review-benchmark [--model <name>] [--write-baseline]\n"
    "       stengents serve-coach [--model <name>] [--port <n>]"
)


def _parse(arguments: list[str]) -> tuple[list[str], str | None, bool, int | None]:
    positional: list[str] = []
    model_override: str | None = None
    write_baseline = False
    port_override: int | None = None
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
        elif token == "--write-baseline":
            write_baseline = True
        elif token == "--port":
            index += 1
            if index >= len(arguments):
                raise ValueError("--port requires a value")
            port_override = int(arguments[index])
        elif token.startswith("--port="):
            port_override = int(token[len("--port=") :])
        else:
            positional.append(token)
        index += 1
    return positional, model_override, write_baseline, port_override


def _run_command(fixture_id: str, model_override: str | None) -> int:
    connection = resolve_model("", name=model_override)
    if not connection.name:
        print("preflight failed: configured_model_missing; detail=a model is required via --model or STENGENTS_MODEL_NAME", file=sys.stderr)
        return 2
    try:
        fixture = _fixture(fixture_id)
        connection.preflight()
    except (ValueError, ModelSourceUnavailable) as error:
        print(f"preflight failed: {error}; endpoint={connection.base_url}; model={connection.name}", file=sys.stderr)
        return 2
    run_directory = Path(".stengents/runs")
    run_id = str(uuid.uuid4())
    record_path = run_directory / f"{run_id}.json"
    print(json.dumps({"run_id": run_id, "fixture_id": fixture.identifier, "model": connection.as_record(), "action_limit": 25, "elapsed_time_limit_seconds": 300, "record_path": str(record_path)}))
    record_path, exit_code = run_fixture(
        fixture,
        run_directory=run_directory,
        model=connection.as_record(),
        agent_driver=adk_driver(connection),
        run_id=run_id,
        rate_limit_policy=connection.rate_limit_policy,
    )
    print(json.dumps({"record_path": str(record_path), "outcome": RunOutcome.from_exit_code(exit_code).value}))
    return exit_code


def _review_benchmark_command(model_override: str | None, write_baseline_flag: bool = False) -> int:
    # Deferred imports: the review capability pulls in pydantic/litellm, which the
    # `run` path does not need.
    from .workout_review import CAPABILITY_VERSION
    from .workout_review.benchmark_runner import build_artifact, corpus_hash, gate_benchmark, run_benchmark, write_artifact, write_baseline
    from .workout_review.evaluator import load_corpus
    from .workout_review.review import DEFAULT_MODEL_NAME

    connection = resolve_model(DEFAULT_MODEL_NAME, name=model_override)
    if not connection.name:
        print("preflight failed: configured_model_missing; detail=a model is required via --model or STENGENTS_MODEL_NAME", file=sys.stderr)
        return 2
    try:
        connection.preflight()
    except ModelSourceUnavailable as error:
        print(f"preflight failed: {error}; endpoint={connection.base_url}; model={connection.name}", file=sys.stderr)
        return 2
    cases = load_corpus()
    run_id = str(uuid.uuid4())
    print(json.dumps({"run_id": run_id, "corpus_hash": corpus_hash(), "case_count": len(cases), "model": connection.as_record(), "capability_version": CAPABILITY_VERSION}))
    results, aggregate, reviews = run_benchmark(cases, model=connection)
    gate = gate_benchmark(cases, results, aggregate, model=connection)
    artifact = build_artifact(results=results, aggregate=aggregate, reviews=reviews, model_record=connection.as_record(), run_id=run_id, gate=gate)
    record_path = write_artifact(artifact)
    print(json.dumps({"record_path": str(record_path), "aggregate": artifact["aggregate"], "gate": {"passed": gate.passed}}))
    if write_baseline_flag:
        baseline_path = write_baseline(artifact)
        print(json.dumps({"baseline_path": str(baseline_path)}))
    return 0


def _serve_coach_command(model_override: str | None, port_override: int | None) -> int:
    # Deferred imports: this path pulls in pydantic/litellm, which `run` doesn't need.
    from .workout_review import kiln_mcp_client
    from .workout_review.review import DEFAULT_MODEL_NAME, review_workout
    from .workout_review.server import serve

    connection = resolve_model(DEFAULT_MODEL_NAME, name=model_override)
    if not connection.name:
        print("preflight failed: configured_model_missing; detail=a model is required via --model or STENGENTS_MODEL_NAME", file=sys.stderr)
        return 2
    try:
        connection.preflight()
    except ModelSourceUnavailable as error:
        print(f"preflight failed: {error}; endpoint={connection.base_url}; model={connection.name}", file=sys.stderr)
        return 2

    # ADR-0005: the coach server reads Kiln over MCP, not kiln_client's HTTP
    # API — kiln_coach's chat LlmAgent keeps that HTTP path untouched.
    def review(workout_id: str):
        return review_workout(
            workout_id,
            fetch=kiln_mcp_client.fetch_workout,
            fetch_history=kiln_mcp_client.fetch_sessions,
            fetch_plans=kiln_mcp_client.fetch_plans,
            model=connection,
        )

    # Defaults to every interface, not just loopback: Kiln reaches this from a
    # separate host (or a separate Docker network namespace), the same LAN-trust
    # boundary Kiln's own KILN_HOST=0.0.0.0 default already assumes. Override
    # with STENGENTS_COACH_HOST for a loopback-only run.
    host = os.environ.get("STENGENTS_COACH_HOST", "0.0.0.0")
    server = serve(host=host, port=port_override or 8787, review_workout=review)
    host, port = server.server_address
    print(json.dumps({"host": host, "port": port, "model": connection.as_record()}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        positional, model_override, write_baseline_flag, port_override = _parse(arguments)
    except ValueError:
        print(_USAGE, file=sys.stderr)
        return 2
    if not positional:
        print(_USAGE, file=sys.stderr)
        return 2
    command = positional[0]
    if command == "run" and len(positional) == 2:
        return _run_command(positional[1], model_override)
    if command == "review-benchmark" and len(positional) == 1:
        return _review_benchmark_command(model_override, write_baseline_flag)
    if command == "serve-coach" and len(positional) == 1:
        return _serve_coach_command(model_override, port_override)
    print(_USAGE, file=sys.stderr)
    return 2
