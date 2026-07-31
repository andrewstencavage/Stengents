"""Tests for the Auto-replan MCP adapter (issue #66): input fetch -> the three
translation gaps (template shape, session shape, exerciseId-as-name) ->
``decide_auto_replan`` -> write dispatch.

Fakes ``kiln_mcp_client``'s ``connect`` seam the same way ``test_kiln_mcp_client.py``
does, so nothing here ever spawns a real Kiln process — a separate live E2E
run against the ``kiln-issue-170`` worktree covers the real wire format.
"""

from __future__ import annotations

import pytest

from stengents.auto_replan.kiln_adapter import (
    _exercise_name_to_id,
    _prescribed_sets_from_spec,
    _translate_session_for_decision,
    run_auto_replan,
    translate_template,
)
from test_kiln_mcp_client import _FakeConnection, _fake_connect

EXERCISES = [
    {"id": "barbell-back-squat", "name": "Barbell Back Squat", "aliases": ["Back Squat"]},
    {"id": "barbell-bench-press", "name": "Barbell Bench Press", "aliases": []},
]


# --------------------------------------------------------------------------- #
# Gap 1: template display shape -> decide_auto_replan's structured shape
# --------------------------------------------------------------------------- #


def test_prescribed_sets_from_spec_recovers_rep_counts_from_kilns_fallback_text() -> None:
    sets = _prescribed_sets_from_spec("a1", "5 reps, 5 reps, 5 reps")

    assert [s["reps"] for s in sets] == [5, 5, 5]
    assert all(s.get("value") is None and s.get("unit") is None for s in sets)
    assert [s["id"] for s in sets] == ["a1-set-0", "a1-set-1", "a1-set-2"]


def test_prescribed_sets_from_spec_degrades_to_one_zero_rep_set_when_unparseable() -> None:
    assert _prescribed_sets_from_spec("a1", "heavy triples") == [{"id": "a1-set-0", "reps": 0}]


def test_translate_template_resolves_display_names_to_real_exercise_ids() -> None:
    name_to_id = _exercise_name_to_id(EXERCISES)
    display_template = {
        "name": "strength",
        "category": "strength",
        "weekFocus": "Base strength",
        "created": "2026-07-01T00:00:00.000Z",
        "updated": None,
        "workouts": [
            {
                "weekday": "Monday",
                "name": "Squat Day",
                "type": "strength",
                "estimatedMinutes": None,
                "activities": [{"name": "Back Squat", "spec": "5 reps, 5 reps", "restSeconds": 90}],
            },
        ],
    }

    translated = translate_template(display_template, name_to_id)

    activity = translated["workouts"][0]["activities"][0]
    assert activity["exerciseId"] == "barbell-back-squat"  # not the display name "Back Squat"
    assert activity["restSeconds"] == 90
    assert [s["reps"] for s in activity["prescribedSets"]] == [5, 5]
    assert activity["id"]  # non-empty, stable, unique across the template


def test_translate_template_drops_an_activity_with_no_catalog_match() -> None:
    name_to_id = _exercise_name_to_id(EXERCISES)
    display_template = {
        "name": "strength",
        "category": "strength",
        "weekFocus": "Base strength",
        "workouts": [
            {
                "weekday": "Monday",
                "name": "Squat Day",
                "type": "strength",
                "activities": [{"name": "Unknown Exercise", "spec": "5 reps", "restSeconds": 90}],
            },
        ],
    }

    translated = translate_template(display_template, name_to_id)

    assert translated["workouts"][0]["activities"] == []


def test_translate_template_returns_none_for_a_nameless_entry() -> None:
    assert translate_template({"category": "strength", "workouts": []}, {}) is None


# --------------------------------------------------------------------------- #
# Gap 3: performed Session Activity names -> real exerciseIds too
# --------------------------------------------------------------------------- #


def test_translate_session_for_decision_resolves_activity_names_to_ids() -> None:
    name_to_id = _exercise_name_to_id(EXERCISES)
    session = {
        "id": "s1",
        "date": "2026-07-27T10:00:00.000Z",
        "activities": [{"name": "Back Squat", "performedSets": [{"reps": 5}]}],
    }

    translated = _translate_session_for_decision(session, name_to_id)

    assert translated["activities"][0]["name"] == "barbell-back-squat"
    assert translated["activities"][0]["performedSets"] == [{"reps": 5}]  # untouched
    assert session["activities"][0]["name"] == "Back Squat"  # original session never mutated


def test_translate_session_for_decision_leaves_an_unresolvable_name_untouched() -> None:
    session = {"id": "s1", "activities": [{"name": "Mystery Move"}]}

    translated = _translate_session_for_decision(session, {})

    assert translated["activities"][0]["name"] == "Mystery Move"


# --------------------------------------------------------------------------- #
# run_auto_replan: fetch -> translate -> decide -> write dispatch
# --------------------------------------------------------------------------- #

STRENGTH_DISPLAY_TEMPLATE = {
    "name": "strength",
    "category": "strength",
    "weekFocus": "Base strength",
    "created": "2026-07-01T00:00:00.000Z",
    "updated": None,
    "workouts": [
        {
            "weekday": "Monday",
            "name": "Squat Day",
            "type": "strength",
            "estimatedMinutes": None,
            "activities": [{"name": "Barbell Back Squat", "spec": "5 reps, 5 reps, 5 reps", "restSeconds": 90}],
        },
        {"weekday": "Tuesday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
        {"weekday": "Wednesday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
        {"weekday": "Thursday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
        {"weekday": "Friday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
        {"weekday": "Saturday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
        {"weekday": "Sunday", "name": "Rest", "type": "rest", "estimatedMinutes": None, "activities": []},
    ],
}

STRATEGY = {
    "version": 1,
    "created": "2026-07-01T00:00:00.000Z",
    "markdown": "# Strategy\n\n```replan-rules\ndefault: strength\n```\n",
}


def _finished_session(session_id: str, *, date: str, workout_name: str, exercise: str = "Barbell Back Squat") -> dict:
    return {
        "id": session_id,
        "status": "finished",
        "date": date,
        "workoutName": workout_name,
        "type": "strength",
        "minutes": 35,
        "feel": "right",
        "activities": [
            {
                "name": exercise,
                "performedSets": [{"reps": 5, "measurement": {"value": 5, "unit": "plate"}}],
            }
        ],
    }


# Reuses test_kiln_mcp_client's fake connection rather than a second copy of
# the same "record calls, answer from a canned table" shape.
_FakeMcp = _FakeConnection


def _base_responses(**overrides):
    responses = {
        "get_planning_context": {
            "strategy": STRATEGY,
            "plans": [],
            "sessions": [_finished_session("w1", date="2026-07-27T10:05:00.000Z", workout_name="Squat Day")],
            "exercises": EXERCISES,
        },
        "list_plan_templates": {"templates": [STRENGTH_DISPLAY_TEMPLATE]},
        "create_plan_template": {"template": {"name": "strength"}},
        "create_plan_draft": {"plan": {"planId": "plan-1"}},
        "select_current_plan": {"plan": {"planId": "plan-1"}},
    }
    responses.update(overrides)
    return responses


def test_run_auto_replan_creates_and_activates_a_draft_end_to_end() -> None:
    fake = _FakeMcp(_base_responses())

    decision = run_auto_replan("w1", connect=_fake_connect(fake))

    assert decision.selected_template == "strength"
    assert decision.activate is True
    called_tools = [name for name, _ in fake.calls]
    assert "create_plan_draft" in called_tools
    assert "select_current_plan" in called_tools
    # select_current_plan is called with the id create_plan_draft actually returned.
    select_call = next(args for name, args in fake.calls if name == "select_current_plan")
    assert select_call == {"planId": "plan-1"}
    # The draft handed to Kiln carries a real exerciseId, never the display name.
    draft_call = next(args for name, args in fake.calls if name == "create_plan_draft")
    assert draft_call["workouts"][0]["activities"][0]["exerciseId"] == "barbell-back-squat"


def test_run_auto_replan_syncs_the_template_when_gap_1s_lost_load_makes_it_look_stale() -> None:
    # The performed Session recorded 5 reps @ 5 plates (a real, structured
    # PrescribedSet), but the template's own reconstructed prescribedSets
    # (gap 1: value/unit are unrecoverable from Kiln's display-shape
    # `list_plan_templates` read) carry no load at all -- so decide_auto_replan
    # correctly (if a little eagerly) treats that as "the template is stale"
    # and proposes a sync. Documents the accepted, harmless false-positive
    # from gap 1's information loss rather than hiding it.
    fake = _FakeMcp(_base_responses())

    decision = run_auto_replan("w1", connect=_fake_connect(fake))

    assert decision.template_updates  # a sync was genuinely proposed
    called_tools = [name for name, _ in fake.calls]
    assert "create_plan_template" in called_tools
    template_call = next(args for name, args in fake.calls if name == "create_plan_template")
    assert template_call["name"] == "strength"


def test_run_auto_replan_does_not_activate_when_the_decision_says_not_to() -> None:
    # No Strategy at all -> select_template finds nothing -> no draft, no writes.
    fake = _FakeMcp(_base_responses(get_planning_context={
        "strategy": None,
        "plans": [],
        "sessions": [_finished_session("w1", date="2026-07-27T10:05:00.000Z", workout_name="Squat Day")],
        "exercises": EXERCISES,
    }))

    decision = run_auto_replan("w1", connect=_fake_connect(fake))

    assert decision.selected_template is None
    assert decision.activate is False
    called_tools = [name for name, _ in fake.calls]
    assert "create_plan_draft" not in called_tools
    assert "select_current_plan" not in called_tools
    assert "create_plan_template" not in called_tools


def test_run_auto_replan_falls_back_to_fetch_workout_when_the_subject_is_not_in_recent_sessions() -> None:
    # get_planning_context's sessions window doesn't include the just-finished
    # Session (a plausible replication-lag edge case) -- list_sessions (via
    # fetch_workout) is consulted as a fallback.
    responses = _base_responses()
    responses["get_planning_context"] = {
        "strategy": STRATEGY,
        "plans": [],
        "sessions": [],
        "exercises": EXERCISES,
    }
    responses["list_sessions"] = {
        "sessions": [
            {**_finished_session("w1", date="2026-07-27T10:05:00.000Z", workout_name="Squat Day"), "sessionId": "w1"}
        ]
    }
    del responses["list_sessions"]["sessions"][0]["id"]
    fake = _FakeMcp(responses)

    decision = run_auto_replan("w1", connect=_fake_connect(fake))

    assert decision.selected_template == "strength"
    assert "list_sessions" in [name for name, _ in fake.calls]


def test_run_auto_replan_propagates_a_read_failure_rather_than_swallowing_it() -> None:
    class _FailingMcp(_FakeMcp):
        def call_tool(self, name, arguments=None, **_):
            raise RuntimeError("kiln unreachable")

    with pytest.raises(RuntimeError, match="kiln unreachable"):
        run_auto_replan("w1", connect=_fake_connect(_FailingMcp({})))
