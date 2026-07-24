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
        model_name="test-model",
        agent_driver=repair,
        budget=RunBudget(action_limit=25, elapsed_seconds=300),
    )

    record = json.loads(record_path.read_text())
    assert exit_code == 0
    assert record["schema_version"] == 1
    assert record["fixture"]["id"] == "normalize-index"
    assert record["model"] == {"provider": "openai-compatible", "name": "test-model"}
    assert [event["name"] for event in record["tool_events"]] == [
        "list_files",
        "read_file",
        "write_source_file",
        "run_tests",
    ]
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


def test_write_accepts_a_normalized_variant_of_an_allowlisted_path(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    fixture = Fixture("fixture", tmp_path, ("source.py",), (sys.executable, "-c", ""))
    actions = Actions(tmp_path, fixture, RunBudget(), time.monotonic())

    assert actions.write_source_file("./source.py", "value = 2\n") == "written"


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
        fixture, run_directory=tmp_path / "runs", model_name="test-model", agent_driver=broken_agent
    )

    assert exit_code == 2
    assert json.loads(record_path.read_text())["verification"]["harness_failed"] is True
