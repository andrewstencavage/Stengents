from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

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
    "       stengents serve-coach [--model <name>] [--port <n>]\n"
    "       stengents strava-sync [--once] [--interval <seconds>] [--dry-run] [--max-attempts <n>]\n"
    "       stengents strava-download [--once] [--interval <seconds>] [--dry-run] [--limit <n>]"
)


def _parse(
    arguments: list[str],
) -> tuple[list[str], str | None, bool, int | None, bool, int | None, bool, int | None, int | None]:
    positional: list[str] = []
    model_override: str | None = None
    write_baseline = False
    port_override: int | None = None
    once = False
    interval: int | None = None
    dry_run = False
    max_attempts: int | None = None
    limit: int | None = None
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
        elif token == "--once":
            once = True
        elif token == "--interval":
            index += 1
            if index >= len(arguments):
                raise ValueError("--interval requires a value")
            interval = int(arguments[index])
        elif token.startswith("--interval="):
            interval = int(token[len("--interval=") :])
        elif token == "--dry-run":
            dry_run = True
        elif token == "--max-attempts":
            index += 1
            if index >= len(arguments):
                raise ValueError("--max-attempts requires a value")
            max_attempts = int(arguments[index])
        elif token.startswith("--max-attempts="):
            max_attempts = int(token[len("--max-attempts=") :])
        elif token == "--limit":
            index += 1
            if index >= len(arguments):
                raise ValueError("--limit requires a value")
            limit = int(arguments[index])
        elif token.startswith("--limit="):
            limit = int(token[len("--limit=") :])
        else:
            positional.append(token)
        index += 1
    return positional, model_override, write_baseline, port_override, once, interval, dry_run, max_attempts, limit


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
    from .auto_replan.kiln_adapter import run_auto_replan
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

    # Issue #66: Auto-replan runs alongside the review inside the same
    # post-Session hook (`server.py`'s `do_GET`), best-effort — a failed or
    # errored run is caught and logged there, never raised back through here.
    # A successful run's own `ReplanDecision.reason` (and each template
    # update's `reason`) is printed here as Auto-replan's whole "light-touch
    # audit trail" (ADR-0002, kiln): Kiln's write schemas have no field to
    # carry that reasoning into the Plan/Template themselves, so a server log
    # line is the only place a captain (or a developer) can see *why* a given
    # write happened, short of reading Kiln's Plan history diff by eye.
    def auto_replan(workout_id: str) -> None:
        decision = run_auto_replan(workout_id)
        print(
            json.dumps(
                {
                    "auto_replan": workout_id,
                    "selected_template": decision.selected_template,
                    "activate": decision.activate,
                    "template_updates": [u.reason for u in decision.template_updates],
                    "reason": decision.reason,
                }
            )
        )

    # Defaults to every interface, not just loopback: Kiln reaches this from a
    # separate host (or a separate Docker network namespace), the same LAN-trust
    # boundary Kiln's own KILN_HOST=0.0.0.0 default already assumes. Override
    # with STENGENTS_COACH_HOST for a loopback-only run.
    host = os.environ.get("STENGENTS_COACH_HOST", "0.0.0.0")
    server = serve(host=host, port=port_override or 8787, review_workout=review, auto_replan=auto_replan)
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


def _strava_sync_command(
    *,
    once: bool,
    interval: int | None,
    dry_run: bool,
    max_attempts: int | None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    # Deferred import: this path is plain Python + Playwright, no model connection at
    # all — unlike every other command here, it must not require a preflight, so nothing
    # model/pydantic/litellm-shaped is imported until this command actually runs.
    from farm_system.strava_uploader.sync import DEFAULT_MAX_ATTEMPTS, run_sync

    poll_interval = interval if interval is not None else 300
    attempts_cap = max_attempts if max_attempts is not None else DEFAULT_MAX_ATTEMPTS
    # Unset (the default) targets real strava.com; point this at a local
    # mock_strava.serve() instance to validate the flow with no real credentials
    # (see farm_system/strava_uploader/README.md).
    playwright_base_url = os.environ.get("STRAVA_UPLOAD_BASE_URL") or None

    while True:
        try:
            summary = run_sync(playwright_base_url=playwright_base_url, dry_run=dry_run, max_attempts=attempts_cap)
        except Exception as error:  # noqa: BLE001 - a bad pass (Kiln unreachable, ...) must not kill a periodic loop
            summary = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(summary))
        if once:
            return 1 if "error" in summary else 0
        try:
            sleep(poll_interval)
        except KeyboardInterrupt:
            return 0


def _strava_download_command(
    *,
    once: bool,
    interval: int | None,
    dry_run: bool,
    limit: int | None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    # Deferred import: this path is plain Python + Playwright, no model connection at
    # all — same reasoning as `_strava_sync_command` below, and for the same reason,
    # must not require a preflight.
    from farm_system.strava_downloader.sync import DEFAULT_LIMIT, run_sync

    poll_interval = interval if interval is not None else 300
    scrape_limit = limit if limit is not None else DEFAULT_LIMIT
    # Unset (the default) targets real strava.com; point this at a local
    # mock_strava.serve() instance to validate the flow with no real credentials
    # (see farm_system/strava_downloader/README.md).
    playwright_base_url = os.environ.get("STRAVA_DOWNLOAD_BASE_URL") or None

    while True:
        try:
            summary = run_sync(playwright_base_url=playwright_base_url, dry_run=dry_run, limit=scrape_limit)
        except Exception as error:  # noqa: BLE001 - a bad pass (Kiln or Strava unreachable, ...) must not kill a periodic loop
            summary = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(summary))
        if once:
            return 1 if "error" in summary or summary.get("scrape_error") else 0
        try:
            sleep(poll_interval)
        except KeyboardInterrupt:
            return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        positional, model_override, write_baseline_flag, port_override, once, interval, dry_run, max_attempts, limit = _parse(arguments)
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
    if command == "strava-sync" and len(positional) == 1:
        return _strava_sync_command(once=once, interval=interval, dry_run=dry_run, max_attempts=max_attempts)
    if command == "strava-download" and len(positional) == 1:
        return _strava_download_command(once=once, interval=interval, dry_run=dry_run, limit=limit)
    print(_USAGE, file=sys.stderr)
    return 2
