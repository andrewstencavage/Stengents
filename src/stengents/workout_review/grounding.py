"""Shared verbatim grounding: does an ``Evidence`` row resolve to real Kiln data?

Generation (:mod:`review`) and evaluation (:mod:`evaluator`) must agree on
*exactly* when an ``Evidence`` row grounds in a pool of Sessions. They used to
carry separate copies of this rule and drifted apart: the generator offered a
synthesised ``sets_completed`` candidate (from ``len(performedSets)``) that the
evaluator would never resolve (it only accepts a literal ``sets_completed``
field), so the model dutifully cited a candidate that could not ground and it
scored as an invented fact. This module is the single source of truth so the two
sides cannot diverge again — the generator offers and cites only what grounds,
and the evaluator scores against the identical check.
"""

from __future__ import annotations

from collections.abc import Iterable

from farm_system.kiln_coach.kiln_client import performed_sets

from .contract import Evidence, FactEvidence, SetEvidence


def as_text(value: object) -> str:
    """Render a source datum as the verbatim text a ``FactEvidence.value`` carries."""
    if isinstance(value, bool):  # JSON-lowercase, never Python's ``True``/``False``
        return "true" if value else "false"
    return str(value)


def sessions_by_id(pool: Iterable[dict]) -> dict[str, dict]:
    return {session["id"]: session for session in pool if "id" in session}


def set_rows_by_session(pool: Iterable[dict]) -> dict[str, dict[tuple[str | None, int], dict]]:
    """Every performed set in the pool, keyed ``session id -> (exercise, index)``."""
    index: dict[str, dict[tuple[str | None, int], dict]] = {}
    for session in pool:
        if "id" not in session:
            continue
        rows: dict[tuple[str | None, int], dict] = {}
        for activity in session.get("activities") or []:
            for row in performed_sets(activity):
                rows[(activity.get("name"), row["set_index"])] = row
        index[session["id"]] = rows
    return index


def _resolve_set_evidence(row: SetEvidence, set_index: dict) -> bool:
    """A SetEvidence resolves iff a matching performed set exists verbatim."""
    source = set_index.get(row.workout_id, {}).get((row.exercise, row.set_index))
    if source is None:
        return False
    return (
        source["reps"] == row.reps
        and source["load"] == row.load
        and source["loadType"] == row.loadType
    )


def _activity_by_name(session: dict, name: str | None) -> dict | None:
    for activity in session.get("activities") or []:
        if activity.get("name") == name:
            return activity
    return None


def _resolve_fact_evidence(row: FactEvidence, sessions: dict[str, dict]) -> bool:
    """A FactEvidence resolves iff its field carries the cited value verbatim."""
    session = sessions.get(row.workout_id)
    if session is None:
        return False

    if row.exercise is None:
        # Session-level fields. ``workout`` is Kiln's ``workoutName``.
        source: object
        if row.field == "workout":
            source = session.get("workoutName")
        elif row.field in {"feel", "minutes", "date", "type"}:
            source = session.get(row.field)
        else:  # an activity-level field cited with no exercise cannot resolve
            return False
        if source is None:
            return False
        if row.field == "date":  # accept the full ISO stamp or its date prefix
            return row.value == as_text(source) or row.value == as_text(source)[:10]
        return row.value == as_text(source)

    activity = _activity_by_name(session, row.exercise)
    if activity is None:
        return False

    if row.field == "planned_sets":
        prescribed = (activity.get("plannedActivity") or {}).get("prescribedSets")
        if prescribed is None:
            return False
        return row.value == as_text(len(prescribed))
    if row.field in {"note", "feel", "skipped", "spec", "sets_completed"}:
        source = activity.get(row.field)
        if source is None:
            return False
        return row.value == as_text(source)
    return False  # a session-only field (minutes/date/...) cited on an activity


def resolves(row: Evidence, set_index: dict, sessions: dict[str, dict]) -> bool:
    """True iff ``row`` grounds verbatim in the given pool indexes."""
    if isinstance(row, SetEvidence):
        return _resolve_set_evidence(row, set_index)
    return _resolve_fact_evidence(row, sessions)


class Grounding:
    """Precomputes a pool's indexes and answers whether an ``Evidence`` row grounds.

    A convenience over the free functions for callers that check many rows against
    one fixed pool (generation guarding its candidates, the evaluator scoring a
    review): build once, ask :meth:`resolves` repeatedly.
    """

    def __init__(self, pool: Iterable[dict]) -> None:
        pool = list(pool)
        self._sessions = sessions_by_id(pool)
        self._set_index = set_rows_by_session(pool)

    def resolves(self, row: Evidence) -> bool:
        return resolves(row, self._set_index, self._sessions)
