from __future__ import annotations

import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from stengents.harness import Actions, Fixture, RunBudget, RunCapturePlugin, run_fixture


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
