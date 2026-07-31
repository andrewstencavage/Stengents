"""The pure Auto-replan decision function (issue #64; parent issue #63).

``decide_auto_replan(strategy, templates, recent_sessions) -> ReplanDecision``
is the seam issue #63's developer user story #15 calls out: "Auto-replan's
decision logic (which template, what draft, whether to activate) implemented
as a single pure function separate from the MCP write calls, so the decision
logic is testable via fixtures without a live Kiln instance." No I/O, no model
call, no randomness — every input is a plain ``dict`` (or ``list[dict]``)
shaped like raw Kiln JSON, exactly the way :mod:`stengents.workout_review`
takes raw Session dicts. A thin, not-yet-built MCP adapter (out of scope for
#64) is meant to fetch these three inputs and execute the writes this decision
proposes.

Three sub-problems, in order:

1. **Selection** (:func:`select_template`) — which named Plan Template
   applies, decided by evaluating the Strategy's **replan rules** (a small,
   deterministic directive convention this module defines and documents
   below) against facts computed from ``recent_sessions``. No model call: a
   Strategy is freeform Markdown authored by the captain in conversation
   (kiln's ``author-strategy`` skill), but the *replan rules* block is a
   structured sub-section of it, by convention, precisely so this step stays
   pure and fixture-testable rather than needing an LLM to read prose.
2. **Draft construction** (:func:`build_draft`) — the selected template's
   content, laid out across the calendar week containing the just-finished
   Session's date, with :func:`reorder_week` detecting and compensating for a
   Session performed on a different weekday than the template schedules it
   (issue #63 captain user story #6).
3. **Template freshness** (:func:`sync_template_update`) — comparing what was
   actually performed against the template's current prescribed sets for that
   exercise, proposing an upsert-ready content update when they differ
   (issue #63 captain user story #4: "kept up to date automatically").

See :mod:`.contract` for the two documented interface assumptions this module
relies on (Plan Template content shape; ``exerciseId`` treated as the
exercise's display name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .contract import (
    WEEKDAY_ORDER,
    DraftActivity,
    DraftWorkout,
    PlanDraft,
    PlanTemplateContent,
    PrescribedSet,
    ReplanDecision,
    TemplateUpdate,
    Weekday,
)

# --------------------------------------------------------------------------- #
# Strategy replan-rules directive convention
# --------------------------------------------------------------------------- #
#
# A Strategy's ``markdown`` may contain one fenced code block tagged
# ``replan-rules``, holding ``key: value`` lines (one per line, first colon is
# the split point). Recognized keys:
#
#   default: <template-name>            — the fallback when nothing else matches
#   deload_every_weeks: <int>           — deload cadence, in distinct weeks
#   deload_template: <template-name>    — which template a deload week selects
#   keyword: <text>                     — a case-insensitive substring to look
#                                          for in the just-finished Session's
#                                          notes/spec text
#   keyword_template: <template-name>   — which template a keyword hit selects
#
# Priority, first match wins: keyword > deload cadence > default. A rule whose
# named template isn't present in ``templates`` is treated as not matching
# (falls through to the next rule) rather than raising — a captain can name a
# template in Strategy prose before it's actually been created.
#
# This convention is this module's own invention (documented here, not fixed
# by any Kiln schema): Kiln's ``strategySchema`` is just ``{version, created,
# markdown}`` — freeform text — and nothing in kiln's schemas.js or the
# author-strategy skill mandates a machine-parseable structure yet. Selecting
# a template from arbitrary prose would need a model in the loop, which the
# ticket's pure-function signature (no ``model``/``complete`` parameter, unlike
# ``review_workout``) rules out. Reconciling this convention with the
# author-strategy skill's actual prompting is flagged as an open question for
# human follow-up.
_RULES_BLOCK = re.compile(r"```replan-rules\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class StrategyRules:
    """Parsed replan-rules directives from one Strategy's markdown."""

    default_template: str | None = None
    deload_every_weeks: int | None = None
    deload_template: str | None = None
    keyword: str | None = None
    keyword_template: str | None = None


def parse_strategy_rules(strategy: dict | None) -> StrategyRules:
    """Parse the ``replan-rules`` fenced block out of a Strategy's markdown.

    ``strategy`` is raw Kiln Strategy JSON (``{version, created, markdown}``)
    or ``None`` when no Strategy has been authored yet. Unrecognized keys and
    unparseable values are ignored rather than raising — a hand-edited or
    partially-written rules block degrades to whatever it does specify.
    """
    if not strategy:
        return StrategyRules()
    markdown = strategy.get("markdown") or ""
    match = _RULES_BLOCK.search(markdown)
    if match is None:
        return StrategyRules()

    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip().lower()] = value.strip()

    deload_every_weeks: int | None = None
    if "deload_every_weeks" in values:
        try:
            deload_every_weeks = int(values["deload_every_weeks"])
        except ValueError:
            deload_every_weeks = None

    return StrategyRules(
        default_template=values.get("default") or None,
        deload_every_weeks=deload_every_weeks,
        deload_template=values.get("deload_template") or None,
        keyword=values.get("keyword") or None,
        keyword_template=values.get("keyword_template") or None,
    )


# --------------------------------------------------------------------------- #
# Facts computed from recent Sessions
# --------------------------------------------------------------------------- #


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def subject_session(recent_sessions: list[dict]) -> dict | None:
    """The Session that just finished: the newest-``date`` finished Session.

    Auto-replan runs on the post-Session hook, so among the ``recent_sessions``
    handed in (which include it, per the ticket's own contract), it is
    unambiguously the most recent finished one.
    """
    finished = [s for s in recent_sessions if s.get("status", "finished") == "finished" and s.get("date")]
    if not finished:
        return None
    return max(finished, key=lambda s: s["date"])


def distinct_week_count(recent_sessions: list[dict]) -> int:
    """The number of distinct ISO calendar weeks represented among finished
    Sessions in ``recent_sessions`` — the deterministic "week index" a deload
    cadence rule is checked against. Fully controlled by ``recent_sessions``,
    which callers scope to whatever history window they consider "recent"."""
    weeks: set[tuple[int, int]] = set()
    for session in recent_sessions:
        if session.get("status", "finished") != "finished" or not session.get("date"):
            continue
        day = _parse_date(session["date"])
        weeks.add(day.isocalendar()[:2])
    return len(weeks)


def _session_text(session: dict) -> str:
    """All free text on a Session worth a keyword search: activity notes/specs
    and the session-level workout name."""
    parts = [session.get("workoutName") or ""]
    for activity in session.get("activities") or []:
        parts.append(activity.get("note") or "")
        parts.append(activity.get("spec") or "")
    return " ".join(parts).lower()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_template(
    strategy: dict | None, templates: list[dict], recent_sessions: list[dict]
) -> tuple[dict | None, str]:
    """Which Plan Template applies, and a human-readable reason why.

    Evaluates :class:`StrategyRules` against facts computed from
    ``recent_sessions`` (see module docstring for the priority order:
    keyword > deload cadence > default). Returns ``(None, reason)`` when no
    rule names a template present in ``templates`` — including when
    ``templates`` is empty or no Strategy/rules exist at all.
    """
    by_name = {t.get("name"): t for t in templates if t.get("name")}
    rules = parse_strategy_rules(strategy)
    subject = subject_session(recent_sessions)

    if rules.keyword and rules.keyword_template and rules.keyword_template in by_name:
        if subject is not None and rules.keyword.lower() in _session_text(subject):
            return by_name[rules.keyword_template], (
                f"Strategy keyword {rules.keyword!r} matched the just-finished Session; "
                f"selected template {rules.keyword_template!r}."
            )

    if rules.deload_every_weeks and rules.deload_template and rules.deload_template in by_name:
        weeks = distinct_week_count(recent_sessions)
        if weeks > 0 and weeks % rules.deload_every_weeks == 0:
            return by_name[rules.deload_template], (
                f"Strategy deload cadence (every {rules.deload_every_weeks} weeks) is due at "
                f"week {weeks}; selected template {rules.deload_template!r}."
            )

    if rules.default_template and rules.default_template in by_name:
        return by_name[rules.default_template], (
            f"No conditional Strategy rule matched; selected the default template "
            f"{rules.default_template!r}."
        )

    return None, "No Strategy replan rule named a template present in the given Plan Templates."


# --------------------------------------------------------------------------- #
# Draft construction
# --------------------------------------------------------------------------- #


def _to_prescribed_sets(raw: list[dict]) -> list[PrescribedSet]:
    return [
        PrescribedSet(id=s["id"], reps=s["reps"], value=s.get("value"), unit=s.get("unit"))
        for s in raw
    ]


def _to_draft_activity(raw: dict) -> DraftActivity:
    return DraftActivity(
        id=raw["id"],
        exerciseId=raw["exerciseId"],
        restSeconds=raw.get("restSeconds", 0),
        prescribedSets=_to_prescribed_sets(raw.get("prescribedSets") or []),
    )


def _to_draft_workout(raw: dict) -> DraftWorkout:
    return DraftWorkout(
        weekday=raw["weekday"],
        name=raw["name"],
        type=raw["type"],
        estimatedMinutes=raw.get("estimatedMinutes"),
        activities=[_to_draft_activity(a) for a in raw.get("activities") or []],
    )


def _rest_workout(weekday: Weekday) -> DraftWorkout:
    return DraftWorkout(weekday=weekday, name="Rest", type="rest", activities=[])


def template_workouts_by_weekday(template: dict) -> dict[Weekday, DraftWorkout]:
    """The template's workouts keyed by weekday, defaulting any weekday the
    template doesn't define to an auto-generated rest day."""
    by_weekday = {w.get("weekday"): _to_draft_workout(w) for w in template.get("workouts") or []}
    return {day: by_weekday.get(day) or _rest_workout(day) for day in WEEKDAY_ORDER}


def reorder_week(
    workouts_by_weekday: dict[Weekday, DraftWorkout], scheduled: Weekday, actual: Weekday
) -> tuple[dict[Weekday, DraftWorkout], str | None]:
    """Compensate for a Session performed on a different weekday than the
    template schedules it (issue #63 captain user story #6), returning the adjusted
    mapping plus a human-readable note (``None`` when ``scheduled == actual``,
    nothing to compensate for).

    **Performed late** (``actual`` after ``scheduled``): the displaced workout
    moves to the day it was actually performed; every day from there through
    Sunday cascades forward by one to make room, "reordering the remaining
    week" (the ticket's own phrase) rather than just marking one day rest and
    leaving the rest of the week untouched; the Sunday-most original workout
    that no longer fits drops off the week. ``scheduled`` itself becomes a
    rest day — nothing happened there; it happened later.

    **Performed early** (``actual`` before ``scheduled``): a narrower, single
    swap — ``scheduled`` and ``actual`` simply exchange content — rather than
    the fuller cascade, since nothing in the ticket's required scenarios
    exercises this direction and a swap never drops a day's content.

    A gap of more than one day where intervening days aren't independently
    accounted for in ``recent_sessions`` is a known, documented limitation:
    only ``scheduled`` and the cascaded/swapped range are touched; days
    strictly between that aren't ``scheduled`` or ``actual`` are left as the
    template originally specified.
    """
    if scheduled == actual:
        return dict(workouts_by_weekday), None

    result = dict(workouts_by_weekday)
    sched_i, actual_i = WEEKDAY_ORDER.index(scheduled), WEEKDAY_ORDER.index(actual)
    displaced = workouts_by_weekday[scheduled]

    if actual_i > sched_i:
        window = WEEKDAY_ORDER[actual_i:]
        originals = [workouts_by_weekday[day] for day in window]
        shifted = [displaced] + originals[:-1]
        for day, workout in zip(window, shifted):
            result[day] = workout.model_copy(update={"weekday": day})
        result[scheduled] = _rest_workout(scheduled)
        note = (
            f"{displaced.name!r} was scheduled for {scheduled} but performed on {actual}; "
            f"{scheduled} is now rest and the remaining week shifted forward a day."
        )
    else:
        swapped = workouts_by_weekday[actual]
        result[scheduled] = swapped.model_copy(update={"weekday": scheduled})
        result[actual] = displaced.model_copy(update={"weekday": actual})
        note = (
            f"{displaced.name!r} was scheduled for {scheduled} but performed early on {actual}; "
            f"swapped with {actual}'s {swapped.name!r}."
        )
    return result, note


def _week_focus(template: dict) -> str:
    """A template's ``weekFocus``, falling back to its ``name`` and then a
    fixed default — ``weekFocus`` is required (min length 1) on both
    ``PlanDraft`` and ``PlanTemplateContent``, but a hand-authored template
    fixture or a sparse live template may not set it."""
    return template.get("weekFocus") or template.get("name") or "Training week"


def build_draft(
    template: dict, workouts_by_weekday: dict[Weekday, DraftWorkout], week_start: date
) -> PlanDraft:
    """Assemble the final draft Plan from an already-reordered weekday mapping."""
    return PlanDraft(
        weekStart=week_start.isoformat(),
        weekFocus=_week_focus(template),
        workouts=[workouts_by_weekday[day] for day in WEEKDAY_ORDER],
    )


# --------------------------------------------------------------------------- #
# Template freshness (issue #63 captain user story #4)
# --------------------------------------------------------------------------- #
#
# **Third documented risk, flagged for human review** (in addition to the two
# in :mod:`.contract`): the Session shape this reads (``activities[].
# performedSets[].measurement.{value,unit}``) is Kiln's HTTP ``/api/sessions``
# shape — the same one :mod:`stengents.workout_review`'s fixtures use, and
# what the ticket asks this module's fixtures to mirror ("the same way
# Workout Review's input.json cases are self-contained bundles of raw Kiln
# Session JSON"). But ADR-0005 has the coach server becoming a pure MCP
# client, and Kiln's actual MCP ``sessionActivitySchema`` (kiln issue #170,
# ``mcp/schemas.js``) is ``{name, spec, skipped, sets_completed, note,
# feel}`` — no ``performedSets``/``measurement`` at all. Against a genuinely
# MCP-sourced Session, ``_performed_prescribed_sets`` never has data to work
# from and this freshness sync never fires. Reconciling that gap (extending
# MCP's Session shape, or sourcing performed-set detail some other way) is
# out of scope for #64's pure-function seam and left to the not-yet-built MCP
# adapter.


def _performed_prescribed_sets(activity_id: str, performed: list[dict]) -> list[PrescribedSet]:
    sets: list[PrescribedSet] = []
    for index, raw in enumerate(performed):
        measurement = raw.get("measurement") or {}
        sets.append(
            PrescribedSet(
                id=f"{activity_id}-performed-{index}",
                reps=raw.get("reps") or 0,
                value=measurement.get("value"),
                unit=measurement.get("unit"),
            )
        )
    return sets


def _sets_differ(current: list[PrescribedSet], performed: list[PrescribedSet]) -> bool:
    key = lambda sets: [(s.reps, s.value, s.unit) for s in sets]  # noqa: E731
    return key(current) != key(performed)


def sync_template_update(
    template: dict, actual_workout: DraftWorkout, subject: dict
) -> TemplateUpdate | None:
    """Propose a revised template content upsert when what was actually
    performed on ``actual_workout`` differs from the template's current
    prescribed sets for that exercise (matching by ``exerciseId`` == the
    performed activity's display ``name`` — see :mod:`.contract`'s documented
    assumption). ``None`` when nothing differs or nothing matches.
    """
    performed_by_name = {a.get("name"): a for a in subject.get("activities") or [] if a.get("name")}
    updated_activities: list[DraftActivity] = []
    changed = False
    for activity in actual_workout.activities:
        performed = performed_by_name.get(activity.exerciseId)
        performed_sets = (performed or {}).get("performedSets") or []
        if performed is None or not performed_sets:
            updated_activities.append(activity)
            continue
        fresh_sets = _performed_prescribed_sets(activity.id, performed_sets)
        if _sets_differ(activity.prescribedSets, fresh_sets):
            updated_activities.append(activity.model_copy(update={"prescribedSets": fresh_sets}))
            changed = True
        else:
            updated_activities.append(activity)

    if not changed:
        return None

    updated_workout = actual_workout.model_copy(update={"activities": updated_activities})
    by_weekday = template_workouts_by_weekday(template)
    by_weekday[actual_workout.weekday] = updated_workout
    content = PlanTemplateContent(
        name=template["name"],
        category=template.get("category") or "general",
        weekFocus=_week_focus(template),
        workouts=[by_weekday[day] for day in WEEKDAY_ORDER],
    )
    return TemplateUpdate(
        template=content,
        reason=(
            f"{actual_workout.name!r}'s performed sets on {actual_workout.weekday} differ from "
            f"the template's current prescribed sets; syncing so the template stays fresh."
        ),
    )


# --------------------------------------------------------------------------- #
# The decision function
# --------------------------------------------------------------------------- #


def decide_auto_replan(
    strategy: dict | None, templates: list[dict], recent_sessions: list[dict]
) -> ReplanDecision:
    """The pure Auto-replan decision (issue #64): which Plan Template applies,
    a fully-populated draft Plan, whether to activate it, and any Plan
    Template content updates to persist.

    Never raises: an input that yields no selectable template returns a
    schema-valid ``ReplanDecision`` with ``selected_template=None``,
    ``draft=None``, ``activate=False``, and a ``reason`` explaining the gap —
    mirroring ``review_workout``'s "always return valid content" contract.
    """
    template, select_reason = select_template(strategy, templates, recent_sessions)
    if template is None:
        return ReplanDecision(
            selected_template=None, draft=None, activate=False, template_updates=[], reason=select_reason
        )

    subject = subject_session(recent_sessions)
    if subject is None:
        return ReplanDecision(
            selected_template=None,
            draft=None,
            activate=False,
            template_updates=[],
            reason="No finished Session to build a draft against.",
        )

    subject_date = _parse_date(subject["date"])
    week_start = _monday(subject_date)
    actual_weekday = WEEKDAY_ORDER[subject_date.weekday()]

    by_weekday = template_workouts_by_weekday(template)
    scheduled_weekday = next(
        (day for day, workout in by_weekday.items() if workout.name == subject.get("workoutName")), None
    )

    reorder_note: str | None = None
    if scheduled_weekday is not None and scheduled_weekday != actual_weekday:
        by_weekday, reorder_note = reorder_week(by_weekday, scheduled_weekday, actual_weekday)

    draft = build_draft(template, by_weekday, week_start)

    template_updates: list[TemplateUpdate] = []
    actual_workout = by_weekday.get(actual_weekday)
    if actual_workout is not None:
        update = sync_template_update(template, actual_workout, subject)
        if update is not None:
            template_updates.append(update)

    reason = select_reason if reorder_note is None else f"{select_reason} {reorder_note}"
    return ReplanDecision(
        selected_template=template["name"],
        draft=draft,
        activate=True,
        template_updates=template_updates,
        reason=reason,
    )
