"""The deterministic evaluator for Auto-replan decisions (issue #64).

Scores a ``ReplanDecision`` against one benchmark case's ``expectations.json``
with **no model judgment** — every verdict is a structured comparison against
the case's expected fields. This mirrors ``stengents.workout_review.evaluator``'s
deterministic-evaluator philosophy exactly: structured assertions on schema
validity, the selected template, the draft's shape/content, the activate
flag, and any proposed template update — never prose grading, since
``decide_auto_replan`` has no prose to grade (it calls no model).

The checks:

* **schema-valid** — the decision is a valid ``ReplanDecision`` (re-asserted
  here, not trusted).
* **template correct** — ``selected_template`` matches the case's expectation
  (including the ``None``/no-match case).
* **activate correct** — the ``activate`` flag matches.
* **week start correct** — when the case specifies one, ``draft.weekStart``
  matches exactly.
* **weekdays correct** — every ``{weekday, name, type}`` the case lists is
  matched by the draft's corresponding day (covers both "untouched" days and
  the reordered/rest days a case is specifically testing).
* **template update correct** — whether a template update was proposed at all
  matches the case's expectation, and (when the case specifies exact content)
  the proposed prescribed sets for the named exercise match verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .contract import ReplanDecision
from .decision import decide_auto_replan

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    """One loaded benchmark case: raw Kiln-shaped Strategy/Templates/Sessions
    plus its structured expectations."""

    case_id: str
    strategy: dict | None
    templates: tuple[dict, ...]
    recent_sessions: tuple[dict, ...]
    expected_template: str | None
    expected_activate: bool
    expected_week_start: str | None
    expected_weekdays: tuple[dict, ...]
    expects_template_update: bool | None
    expected_template_update: dict | None


def load_case(case_dir: Path) -> Case:
    """Load a case directory (``input.json`` + ``expectations.json``)."""
    payload = json.loads((case_dir / "input.json").read_text())
    expectations = json.loads((case_dir / "expectations.json").read_text())
    return Case(
        case_id=case_dir.name,
        strategy=payload.get("strategy"),
        templates=tuple(payload.get("templates", [])),
        recent_sessions=tuple(payload.get("sessions", [])),
        expected_template=expectations.get("expected_template"),
        expected_activate=expectations["expected_activate"],
        expected_week_start=expectations.get("expected_week_start"),
        expected_weekdays=tuple(expectations.get("expected_weekdays", [])),
        expects_template_update=expectations.get("expects_template_update"),
        expected_template_update=expectations.get("expected_template_update"),
    )


def load_corpus(benchmark_dir: Path = BENCHMARK_DIR) -> list[Case]:
    """Load every case directory under ``benchmark_dir``, sorted by id."""
    return [load_case(p) for p in sorted(benchmark_dir.iterdir()) if p.is_dir()]


def run_case(case: Case) -> ReplanDecision:
    """Run the real ``decide_auto_replan`` against one case's frozen pool —
    the corpus exercises the whole pipeline (selection, draft construction,
    reorder, template sync), not just a hand-assembled decision."""
    return decide_auto_replan(case.strategy, list(case.templates), list(case.recent_sessions))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseResult:
    """The deterministic score of one decision against one case."""

    case_id: str
    schema_valid: bool
    schema_error: str | None
    template_correct: bool
    activate_correct: bool
    week_start_correct: bool
    weekdays_correct: bool
    mismatched_weekdays: tuple[str, ...]
    template_update_correct: bool
    template_update_detail: str | None = field(default=None)

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "schema_valid": self.schema_valid,
            "template_correct": self.template_correct,
            "activate_correct": self.activate_correct,
            "week_start_correct": self.week_start_correct,
            "weekdays_correct": self.weekdays_correct,
            "template_update_correct": self.template_update_correct,
        }

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _coerce_decision(decision: ReplanDecision | dict) -> tuple[ReplanDecision | None, str | None]:
    try:
        if isinstance(decision, ReplanDecision):
            return ReplanDecision.model_validate(decision.model_dump()), None
        return ReplanDecision.model_validate(decision), None
    except ValidationError as error:
        return None, str(error)


def _weekdays_correct(decision: ReplanDecision, expected_weekdays: tuple[dict, ...]) -> tuple[bool, tuple[str, ...]]:
    if not expected_weekdays:
        return True, ()
    if decision.draft is None:
        return False, tuple(e["weekday"] for e in expected_weekdays)
    by_weekday = {w.weekday: w for w in decision.draft.workouts}
    mismatched = []
    for expected in expected_weekdays:
        actual = by_weekday.get(expected["weekday"])
        if actual is None or actual.name != expected["name"] or (
            "type" in expected and actual.type != expected["type"]
        ):
            mismatched.append(expected["weekday"])
    return not mismatched, tuple(mismatched)


def _template_update_correct(decision: ReplanDecision, case: Case) -> tuple[bool, str | None]:
    if case.expects_template_update is not None:
        has_update = bool(decision.template_updates)
        if has_update != case.expects_template_update:
            return False, f"expected template update presence {case.expects_template_update}, got {has_update}"

    if case.expected_template_update is None:
        return True, None

    expected = case.expected_template_update
    for update in decision.template_updates:
        if update.template.name != expected.get("template_name"):
            continue
        workout = next((w for w in update.template.workouts if w.weekday == expected["weekday"]), None)
        if workout is None:
            continue
        activity = next((a for a in workout.activities if a.exerciseId == expected["exercise_id"]), None)
        if activity is None:
            continue
        actual_sets = [(s.reps, s.value, s.unit) for s in activity.prescribedSets]
        expected_sets = [(s["reps"], s.get("value"), s.get("unit")) for s in expected["prescribed_sets"]]
        if actual_sets == expected_sets:
            return True, None
        return False, f"prescribed sets {actual_sets} != expected {expected_sets}"
    return False, "no matching template update found"


def evaluate_decision(decision: ReplanDecision | dict, case: Case) -> CaseResult:
    """Score one ``ReplanDecision`` against one loaded ``Case`` (pure, no model)."""
    valid, schema_error = _coerce_decision(decision)
    if valid is None:
        return CaseResult(
            case_id=case.case_id,
            schema_valid=False,
            schema_error=schema_error,
            template_correct=False,
            activate_correct=False,
            week_start_correct=False,
            weekdays_correct=False,
            mismatched_weekdays=tuple(e["weekday"] for e in case.expected_weekdays),
            template_update_correct=False,
        )

    template_correct = valid.selected_template == case.expected_template
    activate_correct = valid.activate == case.expected_activate
    week_start_correct = case.expected_week_start is None or (
        valid.draft is not None and valid.draft.weekStart == case.expected_week_start
    )
    weekdays_correct, mismatched = _weekdays_correct(valid, case.expected_weekdays)
    template_update_correct, detail = _template_update_correct(valid, case)

    return CaseResult(
        case_id=case.case_id,
        schema_valid=True,
        schema_error=None,
        template_correct=template_correct,
        activate_correct=activate_correct,
        week_start_correct=week_start_correct,
        weekdays_correct=weekdays_correct,
        mismatched_weekdays=mismatched,
        template_update_correct=template_update_correct,
        template_update_detail=detail,
    )
