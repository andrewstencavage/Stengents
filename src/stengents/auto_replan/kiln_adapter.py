"""The Auto-replan MCP adapter (issue #66; parent issue #63).

The "thin adapter" issue #63's developer user story called for: fetches
Auto-replan's three inputs (Strategy, Plan Templates, recent Sessions) over
Kiln's MCP planning boundary, calls the pure :func:`~.decision.decide_auto_replan`,
and executes whatever writes the resulting :class:`~.contract.ReplanDecision`
proposes — a ``create_plan_template`` upsert for each template update, a
``create_plan_draft``, and (only when the decision says so) an explicit
``select_current_plan``. ``create_plan_draft``'s own "draft only, never
auto-activates" contract is untouched by this module: activation is always
this adapter's own separate write call, never implicit in the draft call.

:func:`run_auto_replan` is the single public entry point, wired into the
coach server's existing post-Session hook (``cli.py``'s ``_serve_coach_command``,
alongside ``review_workout`` — see ``workout_review/server.py``). It does not
swallow its own errors: the caller (the post-Session hook) is this feature's
fault-tolerance boundary, exactly mirroring how that hook already turns a
failed ``review_workout`` into a logged 502 rather than a crash, instead of
duplicating that policy here.

Three integration gaps, verified live against a real Kiln instance (issue
#170's ``mcp/schemas.js``/``mcp/server.js`` and ``local/store.js``, branch
``issue-170-plan-template-mcp-tools``) rather than assumed, and resolved here
rather than by changing #64's pure ``decide_auto_replan``:

1. **Template shape** (:func:`translate_template`). ``decide_auto_replan``
   needs a Plan Template's *structured* write shape (``workoutInputSchema``:
   ``{id, exerciseId, restSeconds, prescribedSets}`` per Activity — the same
   shape ``create_plan_template`` accepts). ``list_plan_templates``/
   ``get_plan_template`` return a different, lossier *display* shape instead
   (``daysToWorkouts`` in Kiln's ``mcp/server.js``: ``{name, spec,
   restSeconds}`` per Activity) — confirmed by reading ``mcp/server.js``
   directly, not assumed. This is a real, and only partially reversible,
   information loss: an Activity's real ``id``/``exerciseId`` are gone from
   the display shape entirely (recovered here via the live exercise catalog,
   see gap 3 below), and a ``prescribedSets`` entry's ``value``/``unit``
   (load) are gone with no fallback anywhere in the display shape — only
   ``reps`` survives, via ``daysToWorkouts``'s ``"<reps> reps, <reps>
   reps, ..."`` fallback text for any Activity with no free-text ``spec``
   (always true for an Activity ``create_plan_template`` itself wrote, since
   its input schema has no ``spec`` field at all). See
   :func:`_prescribed_sets_from_spec` for the practical effect of that loss:
   because a synced template's ``value``/``unit`` are stripped again the
   moment they're read back (the loss isn't a one-time transcription error,
   it's structural to this read shape), a template with any prescribed load
   at all will tend to look "stale" and pick up a sync on *every* subsequent
   Session for that exercise, not just once — a real, recurring write, not a
   one-off correction. Not a correctness bug (the synced content is still
   accurate — no blocking guardrail exists per Kiln's ADR-0002 either way),
   but write amplification worth knowing about, and only fixable by Kiln
   exposing the structured shape on its read side too.

2. **Session shape**: verified live (``local/store.js``'s ``saveSession`` /
   ``rowSession`` / ``toMcpSession``) that this is *not* the mismatch #64's
   docstrings worried about, for the common case. A browser-Trained Session's
   ``activities`` JSON blob — including each Activity's ``performedSets`` —
   is stored and forwarded verbatim by both the HTTP and MCP read paths alike
   (``toMcpSession`` just passes ``session.activities`` through unfiltered);
   ``mcp/schemas.js``'s sparse ``sessionActivitySchema`` (``{name, spec,
   skipped, sets_completed, note, feel}``, no ``performedSets``) only
   constrains the MCP **write** tool ``log_session``'s input, not what
   ``list_sessions``/``get_planning_context`` actually return from the store.
   So ``decide_auto_replan``'s fixtures (mirroring the old HTTP shape) match
   real MCP-read Sessions for the common browser-Trained case with no
   translation needed. A genuinely sparse Session *is* possible — one logged
   through the MCP ``log_session`` tool itself (an agent-driven path, not the
   Train console) — and does carry no ``performedSets``; ``sync_template_update``
   already degrades gracefully for that case (skips the freshness sync for
   any Activity with no performed sets to compare, never raises), so nothing
   further was added here for it.

3. **``exerciseId``-as-name assumption**: confirmed wrong against Kiln's real
   exercise catalog (``get_planning_context``'s ``exercises``, e.g. id
   ``"barbell-back-squat"`` for display name ``"Barbell Back Squat"`` —
   Kiln resolves Activities to Exercises strictly by id, never by name, per
   Kiln's own ``CONTEXT.md``). #64's pure function was built and tested only
   against fixtures that (necessarily, having no exercise catalog to draw
   real ids from) set ``exerciseId`` equal to the display name, and matches
   a template Activity to a performed Session Activity by comparing
   ``DraftActivity.exerciseId`` against the performed Activity's ``name``
   field — which only works when those two really are the same string. Fixed
   here, not in ``decide_auto_replan``: both :func:`translate_template` and
   :func:`_translate_session_for_decision` resolve a real Activity/Session
   name to its catalog id *before* handing anything to
   ``decide_auto_replan``, so its existing name-keyed matching logic works
   unmodified — it is genuinely comparing two ids, it just doesn't know that.
   An unresolvable name (no catalog match) is dropped from a translated
   template Activity, or left untouched on a translated Session Activity —
   either way degrading to "this Activity doesn't participate in matching"
   rather than inventing a fake id Kiln's real ``create_plan_template``/
   ``create_plan_draft`` would reject.
"""

from __future__ import annotations

import re

from stengents.workout_review import kiln_mcp_client
from stengents.workout_review.kiln_mcp_client import Connect, DEFAULT_TIMEOUT

from .contract import ReplanDecision, to_kiln_json
from .decision import decide_auto_replan

# The usual source of `recent_sessions` is `get_planning_context`'s own
# fixed `sessions` window (`limit: 50`, mcp/server.js) — not this constant.
# This is only the (deliberately more generous) search window for the
# `fetch_workout` fallback below, used solely on the rarer path where the
# just-finished Session isn't already in that fixed-50 window (see
# `run_auto_replan`).
SESSION_LIMIT = 100


# --------------------------------------------------------------------------- #
# Exercise catalog: the id <-> name lookup gap 1 and gap 3 both need
# --------------------------------------------------------------------------- #


def _exercise_name_to_id(exercises: list[dict]) -> dict[str, str]:
    """A lowercased ``name``/``alias`` -> real Kiln ``exerciseId`` lookup,
    built fresh from the live catalog on every call (``get_planning_context``'s
    ``exercises``) — Kiln resolves Activities to Exercises strictly by id
    (module docstring, gap 3), and this is the only source of that mapping
    available to Stengents; nothing here is cached across Sessions since the
    catalog itself is live, mutable Kiln state."""
    lookup: dict[str, str] = {}
    for exercise in exercises:
        exercise_id = exercise.get("id")
        if not isinstance(exercise_id, str) or not exercise_id:
            continue
        name = exercise.get("name")
        if isinstance(name, str) and name.strip():
            lookup.setdefault(name.strip().lower(), exercise_id)
        for alias in exercise.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                lookup.setdefault(alias.strip().lower(), exercise_id)
    return lookup


def _resolve_exercise_id(name: object, name_to_id: dict[str, str]) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    return name_to_id.get(name.strip().lower())


# --------------------------------------------------------------------------- #
# Gap 1: Plan Template display shape -> decide_auto_replan's structured shape
# --------------------------------------------------------------------------- #

_REPS_PATTERN = re.compile(r"(\d+)\s*reps?", re.IGNORECASE)


def _prescribed_sets_from_spec(activity_id: str, spec: str) -> list[dict]:
    """Best-effort ``prescribedSets`` reconstruction from a display-shape
    Activity's ``spec`` text (see module docstring, gap 1).

    Recovers the set count and each set's ``reps`` from Kiln's
    ``"<reps> reps, <reps> reps, ..."`` fallback rendering; ``value``/``unit``
    (load) are not present anywhere in the display shape and are always
    ``None`` on the result — a documented, one-directional information loss,
    not a bug to work around here. A ``spec`` that doesn't match that pattern
    at all (hand-authored some other way, never round-tripped through
    Auto-replan) degrades to a single ``reps=0`` set rather than raising, so
    the pipeline keeps running; that template's draft/sync will look wrong
    until Auto-replan resyncs it from a real performed Session.
    """
    matches = _REPS_PATTERN.findall(spec or "")
    if not matches:
        return [{"id": f"{activity_id}-set-0", "reps": 0}]
    return [{"id": f"{activity_id}-set-{index}", "reps": int(reps)} for index, reps in enumerate(matches)]


def translate_template(display_template: dict, name_to_id: dict[str, str]) -> dict | None:
    """One Kiln read-shape Plan Template (``list_plan_templates``/
    ``get_plan_template``) -> the structured raw dict shape ``decide_auto_replan``
    reads as input (see module docstring, gap 1). ``None`` for a template with
    no ``name`` at all — not a real Kiln template, nothing to translate.

    Each workout's Activities are rebuilt with a synthetic-but-stable
    ``id`` (``"<weekday>-<index>"``, unique across the whole template since
    weekdays don't repeat) and a real ``exerciseId`` resolved from the
    catalog (gap 3); an Activity whose display name has no catalog match is
    dropped from its workout rather than carrying a fabricated id forward.
    """
    name = display_template.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    workouts: list[dict] = []
    for workout in display_template.get("workouts") or []:
        weekday = workout.get("weekday")
        activities: list[dict] = []
        for index, activity in enumerate(workout.get("activities") or []):
            exercise_id = _resolve_exercise_id(activity.get("name"), name_to_id)
            if exercise_id is None:
                continue
            activity_id = f"{weekday}-{index}"
            activities.append(
                {
                    "id": activity_id,
                    "exerciseId": exercise_id,
                    "restSeconds": activity.get("restSeconds") or 0,
                    "prescribedSets": _prescribed_sets_from_spec(activity_id, activity.get("spec") or ""),
                }
            )
        workouts.append(
            {
                "weekday": weekday,
                "name": workout.get("name"),
                "type": workout.get("type"),
                "estimatedMinutes": workout.get("estimatedMinutes"),
                "activities": activities,
            }
        )

    return {
        "name": name,
        "category": display_template.get("category") or "general",
        "weekFocus": display_template.get("weekFocus"),
        "workouts": workouts,
    }


# --------------------------------------------------------------------------- #
# Gap 3: resolve a Session's performed Activity names to real exerciseIds too
# --------------------------------------------------------------------------- #


def _translate_session_for_decision(session: dict, name_to_id: dict[str, str]) -> dict:
    """A copy of ``session`` with each Activity's ``name`` resolved to its
    real Kiln ``exerciseId`` when the catalog has one (gap 3) — the only way
    ``sync_template_update``'s ``performed_by_name`` lookup (keyed by
    ``DraftActivity.exerciseId``, itself now a real id after
    :func:`translate_template`) can ever match a real performed Activity,
    which Kiln only ever records by display name, never by id. Only ``name``
    changes; every other field (``performedSets``, ``note``, ``skipped``,
    ``spec``, ...) passes through untouched, so this is safe to build purely
    for the decision call without disturbing anything else that reads the
    same Session (Workout Review's own fetch is entirely separate).
    """
    activities = []
    for activity in session.get("activities") or []:
        exercise_id = _resolve_exercise_id(activity.get("name"), name_to_id)
        activities.append({**activity, "name": exercise_id} if exercise_id is not None else activity)
    return {**session, "activities": activities}


# --------------------------------------------------------------------------- #
# The adapter's one public entry point
# --------------------------------------------------------------------------- #


def run_auto_replan(
    workout_id: str,
    *,
    connect: Connect = kiln_mcp_client.connect,
    timeout: float = DEFAULT_TIMEOUT,
    session_limit: int = SESSION_LIMIT,
) -> ReplanDecision:
    """Auto-replan's whole pipeline for one just-finished Session: fetch ->
    translate -> decide -> write.

    Fetches the Strategy, every Plan Template, and recent Sessions (plus the
    exercise catalog the two gap-1/gap-3 translations both need) via MCP;
    ensures ``workout_id`` itself is among the Sessions handed to
    ``decide_auto_replan`` (``get_planning_context``'s fixed-``limit`` recent
    Sessions should already include it, since Auto-replan runs right after
    that Session was saved, but this is not assumed — a miss falls back to
    an explicit ``fetch_workout``); runs the pure decision; and executes
    whatever writes it proposes — a ``create_plan_template`` upsert per
    proposed template update, then (only when there's a ``draft`` at all) one
    ``create_plan_draft``, followed by ``select_current_plan`` only when the
    decision says ``activate``.

    Always returns the ``ReplanDecision`` actually acted on (including a
    "nothing to do" decision — ``selected_template=None``, no writes at all,
    per ``decide_auto_replan``'s own "never raises" contract). Does **not**
    swallow a read or write failure itself: propagates whatever
    ``kiln_mcp_client`` raises. The caller — the coach server's post-Session
    hook — is where this feature's fault tolerance lives (see
    ``workout_review/server.py``), exactly mirroring how that hook already
    turns a failed ``review_workout`` into a logged 502 rather than a crash,
    rather than this module duplicating that policy.
    """
    context = kiln_mcp_client.get_planning_context(connect=connect, timeout=timeout)
    display_templates = kiln_mcp_client.list_plan_templates(connect=connect, timeout=timeout)

    sessions = list(context.get("sessions") or [])
    if not any(session.get("id") == workout_id for session in sessions):
        subject = kiln_mcp_client.fetch_workout(workout_id, limit=session_limit, connect=connect, timeout=timeout)
        if subject is not None:
            sessions = [subject, *sessions]

    name_to_id = _exercise_name_to_id(context.get("exercises") or [])
    templates = [
        translated
        for translated in (translate_template(template, name_to_id) for template in display_templates)
        if translated is not None
    ]
    decision_sessions = [_translate_session_for_decision(session, name_to_id) for session in sessions]

    decision = decide_auto_replan(context.get("strategy"), templates, decision_sessions)

    for update in decision.template_updates:
        kiln_mcp_client.create_plan_template(to_kiln_json(update.template), connect=connect, timeout=timeout)

    if decision.draft is not None:
        plan = kiln_mcp_client.create_plan_draft(to_kiln_json(decision.draft), connect=connect, timeout=timeout)
        if decision.activate:
            plan_id = plan.get("planId") or plan.get("id")
            if plan_id:
                kiln_mcp_client.select_current_plan(plan_id, connect=connect, timeout=timeout)

    return decision
