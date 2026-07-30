"""Deterministic progression and Plan-streak computation (ADR 0004).

Two signals, computed in code rather than left to the model: a per-exercise
**Top-set delta** (this Session's top set vs. the single most recent prior's)
and a **Plan streak** (consecutive fully-elapsed weeks hitting every scheduled
Workout). Both build real ``Observation``s directly from source data — no
model call, no candidate-index indirection — so there is no arithmetic left
for the model to get wrong. See ``CONTEXT.md``'s "Top-set delta" and "Plan
streak" entries for the exact rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .contract import FactEvidence, Observation, SetEvidence
from .grounding import as_text

# --- top-set delta --------------------------------------------------------


def top_set(rows: list[SetEvidence]) -> SetEvidence | None:
    """The set with the highest ``load`` (ties broken by more ``reps``).

    A bodyweight movement's sets all carry ``load=None``; ``None`` sorts below
    every real load so the comparison falls through to ``reps`` alone.
    """
    if not rows:
        return None
    return max(rows, key=lambda row: (row.load if row.load is not None else -1.0, row.reps))


def _fmt_load(load: float | None, load_type: str) -> str:
    return "bodyweight" if load is None else f"{load:g}{load_type}"


def _fmt_top(top: SetEvidence, num_sets: int) -> str:
    return f"{num_sets}×{top.reps}@{_fmt_load(top.load, top.loadType)}"


def top_set_delta(name: str, subject_rows: list[SetEvidence], prior_rows: list[SetEvidence]) -> Observation | None:
    """A ``progression`` Observation comparing this Session's top set for
    ``name`` against the most recent prior's, or ``None`` if either side has
    no sets to compare (nothing to say).

    Priority waterfall: report the first of top-set load, set count, then
    top-set reps that differs; an exact match on all three is itself reported
    as "held steady," not silence. Evidence is every set from both sides, so a
    reader can verify the count as well as the top set.
    """
    subject_top, prior_top = top_set(subject_rows), top_set(prior_rows)
    if subject_top is None or prior_top is None:
        return None

    subject_n, prior_n = len(subject_rows), len(prior_rows)
    if subject_top.load != prior_top.load:
        direction = "up" if (subject_top.load or 0) > (prior_top.load or 0) else "down"
        claim = (
            f"{name}: top set {_fmt_top(subject_top, subject_n)}, {direction} from "
            f"{_fmt_top(prior_top, prior_n)} last time."
        )
    elif subject_n != prior_n:
        direction = "up" if subject_n > prior_n else "down"
        claim = (
            f"{name}: {subject_n} sets at {_fmt_load(subject_top.load, subject_top.loadType)} this time, "
            f"{direction} from {prior_n} sets last time."
        )
    elif subject_top.reps != prior_top.reps:
        direction = "up" if subject_top.reps > prior_top.reps else "down"
        claim = (
            f"{name}: top set {subject_top.reps} reps at {_fmt_load(subject_top.load, subject_top.loadType)}, "
            f"{direction} from {prior_top.reps} reps last time."
        )
    else:
        claim = f"{name}: held steady at {_fmt_top(subject_top, subject_n)}, same as last time."

    return Observation(
        kind="fact",
        confidence="firm",
        category="progression",
        claim=claim,
        evidence=[*subject_rows, *prior_rows],
    )


# --- plan streak -----------------------------------------------------------


@dataclass(frozen=True)
class StreakResult:
    """A Plan streak: how many consecutive fully-elapsed weeks were hit, and
    the one milestone Session per hit week (oldest first) an Observation about
    it can cite as evidence."""

    weeks: int
    milestones: tuple[dict, ...]


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def plan_streak(plans: list[dict], sessions: list[dict], as_of: str) -> StreakResult:
    """Consecutive fully-elapsed calendar weeks, ending with the most recent
    one to finish before ``as_of``'s week, where every non-rest scheduled
    Workout in that week's Plan had >=1 finished Session.

    Matches Kiln's own ``doneWorkoutNames`` semantics exactly: no day-level
    deadline within the week, and abandoned Sessions never count (only
    ``status == "finished"`` Sessions are considered). A week with no active
    or historical Plan (drafts don't count) breaks the streak; the current,
    still-in-progress week is excluded from the count either way.
    """
    by_week: dict[str, dict] = {}
    for plan in plans:
        if plan.get("status") == "draft":
            continue
        week_start = plan.get("weekStart")
        if week_start:
            by_week.setdefault(week_start, plan)

    week = _monday(_parse_date(as_of)) - timedelta(days=7)
    weeks = 0
    milestones: list[dict] = []
    while True:
        plan = by_week.get(week.isoformat())
        if plan is None:
            break
        required = {
            day["name"]
            for day in plan.get("days", [])
            if day.get("type") != "rest" and day.get("activities") and day.get("name")
        }
        week_sessions = [
            s for s in sessions if s.get("planId") == plan.get("id") and s.get("status") == "finished"
        ]
        done = {s.get("workoutName") for s in week_sessions}
        if not required.issubset(done):
            break
        milestones.append(max(week_sessions, key=lambda s: s.get("date") or ""))
        weeks += 1
        week -= timedelta(days=7)

    milestones.reverse()  # oldest hit week first
    return StreakResult(weeks=weeks, milestones=tuple(milestones))


def streak_observation(result: StreakResult) -> Observation | None:
    """An ``adherence`` Observation for a nonzero Plan streak, citing one real
    Session's ``date`` per hit week — or ``None`` when there is no streak."""
    if result.weeks <= 0:
        return None
    plural = "s" if result.weeks != 1 else ""
    claim = f"{result.weeks} week{plural} straight hitting every scheduled Workout in your Plan."
    evidence = [
        FactEvidence(workout_id=session["id"], exercise=None, field="date", value=as_text(session.get("date")))
        for session in result.milestones
    ]
    return Observation(kind="fact", confidence="firm", category="adherence", claim=claim, evidence=evidence)
