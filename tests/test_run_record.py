from __future__ import annotations

import json
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from stengents.coding_agent.agent import RunCapturePlugin, _agent_instruction, _required_discovery_tool
from stengents.harness import (
    Actions,
    Fixture,
    RunBudget,
    run_fixture,
)
from stengents.run_record import RunOutcome, build_run_record, derive_outcome


def test_run_capture_plugin_callbacks_are_awaitable(tmp_path: Path) -> None:
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), 0)
    plugin = RunCapturePlugin(actions)

    asyncio.run(plugin.before_run_callback(invocation_context=SimpleNamespace(invocation_id="invocation-1")))

    assert actions.adk_invocation_id == "invocation-1"


def test_run_fixture_writes_a_passing_record_after_the_agent_repairs_the_source(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    source = fixture_root / "normalize_index.py"
    fixture_root.mkdir()
    source.write_text(
        "def normalize_index(items, index):\n"
        "    if index < 0 or index > len(items):\n"
        "        raise IndexError(index)\n"
        "    return items[index]\n"
    )
    (fixture_root / "test_normalize_index.py").write_text(
        "import pytest\n\n"
        "from normalize_index import normalize_index\n\n"
        "def test_rejects_index_at_the_upper_bound():\n"
        "    with pytest.raises(IndexError):\n"
        "        normalize_index(['a'], 1)\n"
    )

    fixture = Fixture(
        identifier="normalize-index",
        root=fixture_root,
        source_surface=("normalize_index.py",),
        verifier=(sys.executable, "-m", "pytest", "-q"),
    )

    def repair(actions) -> None:
        actions.list_files()
        actions.read_file("normalize_index.py")
        actions.write_source_file(
            "normalize_index.py",
            "def normalize_index(items, index):\n"
            "    if index < 0 or index >= len(items):\n"
            "        raise IndexError(index)\n"
            "    return items[index]\n",
        )
        actions.run_tests()

    record_path, exit_code = run_fixture(
        fixture,
        run_directory=tmp_path / "runs",
        model={"provider": "openai-compatible", "name": "test-model"},
        agent_driver=repair,
        budget=RunBudget(action_limit=25, elapsed_seconds=300),
    )

    record = json.loads(record_path.read_text())
    assert exit_code == 0
    assert record["schema_version"] == 3
    assert record["outcome"] == "passed"
    assert record["fixture"]["id"] == "normalize-index"
    assert record["model"] == {"provider": "openai-compatible", "name": "test-model"}
    assert [event["name"] for event in record["tool_events"]] == [
        "list_files",
        "read_file",
        "write_source_file",
        "run_tests",
    ]
    events = {event["name"]: event for event in record["tool_events"]}
    assert events["read_file"]["args"] == {"path": "normalize_index.py"}
    assert events["read_file"]["result_summary"].startswith("def normalize_index(items, index):")
    assert events["run_tests"]["result_summary"] == {"exit_code": 0, "passed": True}
    assert record["adk"] == {"invocation_id": None, "agent": "coding_agent"}
    assert record["verification"] == {
        "command": [sys.executable, "-m", "pytest", "-q"],
        "exit_code": 0,
        "passed": True,
    }
    assert record["artifacts"][0]["path"] == "normalize_index.py"


def test_write_rejects_paths_outside_the_source_surface(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert actions.write_source_file("other.py", "value = 2\n") == (
        "rejected: other.py is not in the fixture source surface; only source.py may be changed"
    )


def test_read_returns_a_rejection_for_a_path_outside_the_fixture(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert actions.read_file("../escape.py") == (
        "rejected: ../escape.py is outside the fixture; call list_files to see the readable paths"
    )
    assert actions.events[-1]["outcome"] == "ok"


def test_read_returns_a_rejection_for_a_missing_fixture_file(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert actions.read_file("missing.py") == (
        "rejected: missing.py does not exist in the fixture; call list_files to see the readable paths"
    )


def test_a_model_reading_a_bad_path_is_not_a_harness_failure(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "normalize_index.py").write_text("pass\n")
    fixture = Fixture("normalize-index", fixture_root, ("normalize_index.py",), (sys.executable, "-c", "raise SystemExit(1)"))

    def agent_reads_outside_the_fixture(actions) -> None:
        actions.read_file("/etc/passwd")

    record_path, exit_code = run_fixture(
        fixture, run_directory=tmp_path / "runs", model={"provider": "openai-compatible", "name": "test-model"}, agent_driver=agent_reads_outside_the_fixture
    )

    verification = json.loads(record_path.read_text())["verification"]
    assert "harness_failed" not in verification
    assert verification["passed"] is False
    assert exit_code == 1


def test_write_accepts_a_normalized_variant_of_an_allowlisted_path(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert actions.write_source_file("./source.py", "value = 2\n") == "written"


def test_tool_event_records_bounded_args_and_result(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    actions.write_source_file("source.py", "y = " + "x" * 250)

    event = actions.events[-1]
    assert event["name"] == "write_source_file"
    assert event["args"]["path"] == "source.py"
    # the oversized content is truncated to the reducer's bound, never stored whole
    assert event["args"]["content"].endswith("…(+54)") and len(event["args"]["content"]) == 206
    assert event["outcome"] == "ok"
    assert event["result_summary"] == "written"


def test_agent_instruction_names_the_fixture_and_editable_source_surface(tmp_path: Path) -> None:
    fixture = Fixture("normalize-index", tmp_path, ("normalize_index.py",), (sys.executable, "-c", ""))

    instruction = _agent_instruction(fixture)

    assert "normalize-index" in instruction
    assert "normalize_index.py" in instruction
    assert "list_files" in instruction
    assert "read_file" in instruction
    assert "Tests are immutable" in instruction
    assert "upper-bound index check" in instruction


def test_required_discovery_tool_forces_listing_then_reading(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert _required_discovery_tool(actions) == "list_files"
    actions.list_files()
    assert _required_discovery_tool(actions) == "read_file"
    actions.read_file("source.py")
    assert _required_discovery_tool(actions) is None


def test_run_fixture_distinguishes_a_harness_error_from_a_fixture_failure(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "normalize_index.py").write_text("pass\n")
    fixture = Fixture("normalize-index", fixture_root, ("normalize_index.py",), (sys.executable, "-c", "raise SystemExit(1)"))

    def broken_agent(actions) -> None:
        raise ConnectionError("model unavailable")

    record_path, exit_code = run_fixture(
        fixture, run_directory=tmp_path / "runs", model={"provider": "openai-compatible", "name": "test-model"}, agent_driver=broken_agent
    )

    assert exit_code == 2
    assert json.loads(record_path.read_text())["verification"]["harness_failed"] is True


# --- The Run record as a deep module: pure shape + outcome derivation --------


def _verification(**overrides: object) -> dict[str, object]:
    verification: dict[str, object] = {"command": ["true"], "exit_code": None, "passed": False}
    verification.update(overrides)
    return verification


def test_outcome_exit_codes_follow_the_adr_0002_taxonomy() -> None:
    assert RunOutcome.PASSED.exit_code == 0
    assert RunOutcome.FAILED.exit_code == 1
    assert RunOutcome.HARNESS_FAILED.exit_code == 2
    for outcome in RunOutcome:
        assert RunOutcome.from_exit_code(outcome.exit_code) is outcome


def test_derive_outcome_treats_a_passing_verification_as_passed() -> None:
    assert derive_outcome(_verification(exit_code=0, passed=True)) is RunOutcome.PASSED


def test_derive_outcome_treats_an_exhausted_budget_as_a_model_failure_not_a_harness_failure() -> None:
    verification = _verification(error="run action budget exhausted", budget_exhausted=True)
    assert derive_outcome(verification) is RunOutcome.FAILED


def test_derive_outcome_treats_a_rejected_verifier_as_failed() -> None:
    assert derive_outcome(_verification(exit_code=1, passed=False)) is RunOutcome.FAILED


def test_derive_outcome_treats_a_flagged_harness_fault_as_harness_failed() -> None:
    verification = _verification(error="model unavailable", harness_failed=True)
    assert derive_outcome(verification) is RunOutcome.HARNESS_FAILED


def test_build_run_record_stores_the_derived_outcome_and_full_shape() -> None:
    record, outcome = build_run_record(
        run_id="run-1",
        started_at="2026-07-25T00:00:00Z",
        duration_ms=12,
        fixture={"id": "normalize-index", "revision": "abc"},
        adk={"invocation_id": "inv-1", "agent": "coding_agent"},
        model={"provider": "openai-compatible", "name": "test-model"},
        tool_events=[{"name": "run_tests", "outcome": "ok"}],
        artifacts=[{"path": "normalize_index.py", "sha256": "def"}],
        verification=_verification(exit_code=0, passed=True),
    )

    assert outcome is RunOutcome.PASSED
    assert record["schema_version"] == 3
    assert record["outcome"] == "passed"
    assert record["run_id"] == "run-1"
    assert record["harness"] == {"id": "stengents", "revision": "working-tree"}
    assert record["fixture"] == {"id": "normalize-index", "revision": "abc"}
    assert record["verification"]["passed"] is True


def test_build_run_record_reports_harness_failed_when_flagged() -> None:
    _record, outcome = build_run_record(
        run_id="run-2",
        started_at="2026-07-25T00:00:00Z",
        duration_ms=1,
        fixture={"id": "normalize-index", "revision": "abc"},
        adk={"invocation_id": None, "agent": "coding_agent"},
        model={"provider": "openai-compatible", "name": "test-model"},
        tool_events=[],
        artifacts=[],
        verification=_verification(error="model unavailable", harness_failed=True),
    )

    assert outcome is RunOutcome.HARNESS_FAILED
    assert outcome.exit_code == 2
