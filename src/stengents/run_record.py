"""The Run record: the persisted trace and deterministic outcome of one fixture run.

A **Run record** is the local JSON trace of one ``stengents run`` execution. This
module owns its shape and the ADR-0002 outcome taxonomy as pure, unit-tested
values, mirroring the way ``kiln_coach`` factors ``build_turn_record`` out of its
logging shell. ``run_fixture`` in ``harness.py`` is left to orchestrate the
ephemeral workspace, the agent driver, and persistence; it assembles the
verification result and hands it here.

See ``docs/adr/0002-run-outcome-taxonomy.md`` for why the boundary between
``failed`` and ``harness_failed`` is a deliberate rule rather than an accident of
where an exception lands.
"""

from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = 3


class RunOutcome(Enum):
    """The process-level result of a run, and the exit code it maps to.

    The three-way taxonomy of ADR-0002: ``PASSED`` (the verifier accepted the
    result), ``FAILED`` (a fair attempt the verifier rejected — every way the
    *model* can fall short, including an exhausted budget), and
    ``HARNESS_FAILED`` (the harness could not render a fair verdict at all).
    """

    PASSED = "passed"
    FAILED = "failed"
    HARNESS_FAILED = "harness_failed"

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES[self]

    @classmethod
    def from_exit_code(cls, exit_code: int) -> RunOutcome:
        return _OUTCOMES_BY_EXIT_CODE[exit_code]


_EXIT_CODES = {
    RunOutcome.PASSED: 0,
    RunOutcome.FAILED: 1,
    RunOutcome.HARNESS_FAILED: 2,
}

_OUTCOMES_BY_EXIT_CODE = {code: outcome for outcome, code in _EXIT_CODES.items()}


def derive_outcome(verification: dict[str, object]) -> RunOutcome:
    """Map a completed verification result onto the run's outcome (pure).

    A passed verification is ``PASSED``; an unfinished verdict the harness flagged
    ``harness_failed`` is ``HARNESS_FAILED``; everything else — a wrong patch, a
    rejected out-of-fixture path, an exhausted budget — is a ``FAILED`` verdict
    about the model, never about the harness.
    """
    if verification.get("passed"):
        return RunOutcome.PASSED
    if verification.get("harness_failed"):
        return RunOutcome.HARNESS_FAILED
    return RunOutcome.FAILED


def build_run_record(
    *,
    run_id: str,
    started_at: str,
    duration_ms: int,
    fixture: dict[str, str],
    adk: dict[str, object],
    model: dict[str, str],
    tool_events: list[dict[str, object]],
    artifacts: list[dict[str, str]],
    verification: dict[str, object],
) -> tuple[dict[str, object], RunOutcome]:
    """Assemble one Run record and derive its outcome (pure).

    Returns the record dict — with the derived ``outcome`` stored on it, so the
    taxonomy is the record's interface rather than something each reader
    reconstructs from ``verification`` flags — and the ``RunOutcome`` whose
    ``exit_code`` the caller returns as the process status.
    """
    outcome = derive_outcome(verification)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "duration_ms": duration_ms,
        "harness": {"id": "stengents", "revision": "working-tree"},
        "fixture": fixture,
        "adk": adk,
        "model": model,
        "tool_events": tool_events,
        "artifacts": artifacts,
        "outcome": outcome.value,
        "verification": verification,
    }
    return record, outcome
