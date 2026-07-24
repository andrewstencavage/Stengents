from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class RunBudgetExceeded(RuntimeError):
    """A deterministic failure caused by exceeding the fixed run budget."""


@dataclass(frozen=True)
class Fixture:
    identifier: str
    root: Path
    source_surface: tuple[str, ...]
    verifier: tuple[str, ...]


@dataclass(frozen=True)
class RunBudget:
    action_limit: int = 25
    elapsed_seconds: int = 300


class Actions:
    """The only filesystem and verification actions exposed to a coding agent."""

    def __init__(self, root: Path, fixture: Fixture, budget: RunBudget, started: float) -> None:
        self.root, self.fixture, self.budget, self.started = root, fixture, budget, started
        self.events: list[dict[str, object]] = []
        self.adk_invocation_id: str | None = None
        self.adk_lifecycle_events: list[dict[str, str]] = []

    def _path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise ValueError("path must remain inside the fixture")
        return candidate

    def _fixture_relative_path(self, path: str) -> str:
        return self._path(path).relative_to(self.root.resolve()).as_posix()

    def _act(self, name: str, operation: Callable[[], T]) -> T:
        if len(self.events) >= self.budget.action_limit:
            raise RunBudgetExceeded("run action budget exhausted")
        if time.monotonic() - self.started > self.budget.elapsed_seconds:
            raise RunBudgetExceeded("run elapsed-time budget exhausted")
        offset = round((time.monotonic() - self.started) * 1000)
        action_started = time.monotonic()
        try:
            result = operation()
        except Exception:
            self.events.append({"name": name, "started_offset_ms": offset, "duration_ms": round((time.monotonic() - action_started) * 1000), "outcome": "error"})
            raise
        self.events.append({"name": name, "started_offset_ms": offset, "duration_ms": round((time.monotonic() - action_started) * 1000), "outcome": "ok"})
        return result

    def list_files(self) -> list[str]:
        return self._act("list_files", lambda: sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()))

    def read_file(self, path: str) -> str:
        return self._act("read_file", lambda: self._path(self._fixture_relative_path(path)).read_text())

    def write_source_file(self, path: str, content: str) -> str:
        """Write only an allowlisted source file; rejected paths are returned to the agent."""
        def write() -> str:
            try:
                relative_path = self._fixture_relative_path(path)
            except ValueError as error:
                return f"rejected: {error}"
            if relative_path not in self.fixture.source_surface:
                allowed_paths = ", ".join(self.fixture.source_surface)
                return f"rejected: {path} is not in the fixture source surface; only {allowed_paths} may be changed"
            self._path(relative_path).write_text(content)
            return "written"
        return self._act("write_source_file", write)

    def run_tests(self) -> dict[str, object]:
        def verify() -> dict[str, object]:
            completed = subprocess.run(self.fixture.verifier, cwd=self.root, capture_output=True, text=True, timeout=self.budget.elapsed_seconds, check=False)
            return {"exit_code": completed.returncode, "passed": completed.returncode == 0}
        return self._act("run_tests", verify)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fixture(
    fixture: Fixture,
    *,
    run_directory: Path,
    model_name: str,
    agent_driver: Callable[[Actions], None],
    budget: RunBudget = RunBudget(),
    run_id: str | None = None,
) -> tuple[Path, int]:
    """Run one fixture in an ephemeral copy and atomically persist its record."""
    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    run_directory.mkdir(parents=True, exist_ok=True)
    verification: dict[str, object] = {"command": list(fixture.verifier), "exit_code": None, "passed": False}
    with tempfile.TemporaryDirectory(prefix=f"stengents-{fixture.identifier}-") as workspace:
        root = Path(workspace) / fixture.identifier
        shutil.copytree(fixture.root, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        actions = Actions(root, fixture, budget, started)
        try:
            agent_driver(actions)
            completed = subprocess.run(fixture.verifier, cwd=root, capture_output=True, text=True, timeout=budget.elapsed_seconds, check=False)
            verification.update(exit_code=completed.returncode, passed=completed.returncode == 0)
        except RunBudgetExceeded as error:
            verification["error"] = str(error)
            verification["budget_exhausted"] = True
        except Exception as error:
            verification["error"] = str(error)
            verification["harness_failed"] = True
        artifacts = [{"path": path, "sha256": _digest(root / path)} for path in fixture.source_surface]
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "harness": {"id": "stengents", "revision": "working-tree"},
            "fixture": {"id": fixture.identifier, "revision": _digest(fixture.root / fixture.source_surface[0])},
            "adk": {"invocation_id": actions.adk_invocation_id, "agent": "coding_agent", "tool_lifecycle_events": actions.adk_lifecycle_events},
            "model": {"provider": "openai-compatible", "name": model_name},
            "tool_events": actions.events,
            "artifacts": artifacts,
            "verification": verification,
        }
    record_path = run_directory / f"{run_id}.json"
    temporary_path = record_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(record_path)
    if verification["passed"]:
        return record_path, 0
    return record_path, 2 if verification.get("harness_failed") else 1
