"""Tests for the pure Auto-replan decision function (issue #64; parent issue #63).

Exercises ``decide_auto_replan`` directly against hand-built raw-Kiln-shaped
inputs — the same seam the fixture corpus (``tests/test_auto_replan_evaluator.py``
and ``src/stengents/auto_replan/benchmark/``) exercises through frozen JSON.
"""

from __future__ import annotations

from stengents.auto_replan.contract import PlanDraft, to_kiln_json
from stengents.auto_replan.decision import (
    decide_auto_replan,
    distinct_week_count,
    parse_strategy_rules,
    select_template,
)

STRENGTH_TEMPLATE = {
    "schemaVersion": 1,
    "name": "strength",
    "category": "strength",
    "weekFocus": "Base strength",
    "workouts": [
        {
            "weekday": "Monday",
            "name": "Squat Day",
            "type": "strength",
            "activities": [
                {
                    "id": "a1",
                    "exerciseId": "Back Squat",
                    "restSeconds": 90,
                    "prescribedSets": [
                        {"id": "s1", "reps": 5, "value": 5, "unit": "plate"},
                        {"id": "s2", "reps": 5, "value": 5, "unit": "plate"},
                        {"id": "s3", "reps": 5, "value": 5, "unit": "plate"},
                    ],
                }
            ],
        },
        {"weekday": "Tuesday", "name": "Rest", "type": "rest", "activities": []},
        {
            "weekday": "Wednesday",
            "name": "Push Day",
            "type": "strength",
            "activities": [
                {
                    "id": "a2",
                    "exerciseId": "Bench Press",
                    "restSeconds": 90,
                    "prescribedSets": [
                        {"id": "s1", "reps": 8, "value": 3, "unit": "plate"},
                        {"id": "s2", "reps": 8, "value": 3, "unit": "plate"},
                    ],
                }
            ],
        },
        {"weekday": "Thursday", "name": "Pull Day", "type": "strength", "activities": [
            {
                "id": "a3",
                "exerciseId": "Barbell Row",
                "restSeconds": 90,
                "prescribedSets": [{"id": "s1", "reps": 8, "value": 3, "unit": "plate"}],
            }
        ]},
        {
            "weekday": "Friday",
            "name": "Leg Day",
            "type": "strength",
            "activities": [
                {
                    "id": "a4",
                    "exerciseId": "Front Squat",
                    "restSeconds": 90,
                    "prescribedSets": [{"id": "s1", "reps": 8, "value": 2, "unit": "plate"}],
                }
            ],
        },
        {"weekday": "Saturday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Sunday", "name": "Rest", "type": "rest", "activities": []},
    ],
}

DELOAD_TEMPLATE = {
    "schemaVersion": 1,
    "name": "deload",
    "category": "deload",
    "weekFocus": "Deload week",
    "workouts": [
        {
            "weekday": "Monday",
            "name": "Light Squat Day",
            "type": "strength",
            "activities": [
                {
                    "id": "a1",
                    "exerciseId": "Back Squat",
                    "restSeconds": 120,
                    "prescribedSets": [{"id": "s1", "reps": 5, "value": 2, "unit": "plate"}],
                }
            ],
        },
        {"weekday": "Tuesday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Wednesday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Thursday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Friday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Saturday", "name": "Rest", "type": "rest", "activities": []},
        {"weekday": "Sunday", "name": "Rest", "type": "rest", "activities": []},
    ],
}

DEFAULT_STRATEGY = {
    "version": 1,
    "created": "2026-07-01T00:00:00.000Z",
    "markdown": "# Strategy\n\nGeneral strength focus.\n\n```replan-rules\ndefault: strength\n```\n",
}

DEFAULT_DELOAD_STRATEGY = {
    "version": 2,
    "created": "2026-07-01T00:00:00.000Z",
    "markdown": (
        "# Strategy\n\nDeload every 4th week.\n\n"
        "```replan-rules\ndefault: strength\ndeload_every_weeks: 4\ndeload_template: deload\n```\n"
    ),
}


def _session(
    session_id: str,
    date: str,
    workout_name: str,
    activities: list[dict],
    status: str = "finished",
) -> dict:
    return {
        "id": session_id,
        "status": status,
        "date": date,
        "workoutName": workout_name,
        "type": "strength",
        "minutes": 40,
        "feel": "right",
        "activities": activities,
    }


def _activity(name: str, performed_sets: list[tuple[int, float | None, str]], note: str | None = None) -> dict:
    activity = {
        "name": name,
        "plannedActivity": {"prescribedSets": [0] * len(performed_sets)},
        "performedSets": [
            {"reps": reps, "measurement": {"value": value, "unit": unit}} for reps, value, unit in performed_sets
        ],
    }
    if note is not None:
        activity["note"] = note
    return activity


# --------------------------------------------------------------------------- #
# Strategy replan-rules parsing
# --------------------------------------------------------------------------- #
def test_parse_strategy_rules_reads_default() -> None:
    rules = parse_strategy_rules(DEFAULT_STRATEGY)
    assert rules.default_template == "strength"
    assert rules.deload_every_weeks is None


def test_parse_strategy_rules_reads_deload_cadence() -> None:
    rules = parse_strategy_rules(DEFAULT_DELOAD_STRATEGY)
    assert rules.deload_every_weeks == 4
    assert rules.deload_template == "deload"


def test_parse_strategy_rules_with_no_strategy_is_empty() -> None:
    rules = parse_strategy_rules(None)
    assert rules.default_template is None
    assert rules.deload_every_weeks is None


def test_parse_strategy_rules_with_no_rules_block_is_empty() -> None:
    strategy = {"version": 1, "created": "x", "markdown": "# Strategy\n\nJust prose, no rules block."}
    rules = parse_strategy_rules(strategy)
    assert rules.default_template is None


# --------------------------------------------------------------------------- #
# distinct_week_count
# --------------------------------------------------------------------------- #
def test_distinct_week_count_counts_unique_iso_weeks() -> None:
    sessions = [
        _session("s1", "2026-06-29T10:00:00.000Z", "A", []),  # week 1
        _session("s2", "2026-07-06T10:00:00.000Z", "A", []),  # week 2
        _session("s3", "2026-07-06T18:00:00.000Z", "A", []),  # same week as s2
        _session("s4", "2026-07-13T10:00:00.000Z", "A", []),  # week 3
        _session("s5", "2026-07-20T10:00:00.000Z", "A", []),  # week 4
    ]
    assert distinct_week_count(sessions) == 4


def test_distinct_week_count_ignores_abandoned_sessions() -> None:
    sessions = [
        _session("s1", "2026-06-29T10:00:00.000Z", "A", []),
        _session("s2", "2026-07-06T10:00:00.000Z", "A", [], status="abandoned"),
    ]
    assert distinct_week_count(sessions) == 1


# --------------------------------------------------------------------------- #
# select_template
# --------------------------------------------------------------------------- #
def test_select_template_falls_back_to_default() -> None:
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])])]
    template, reason = select_template(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert template is not None
    assert template["name"] == "strength"
    assert "default" in reason.lower()


def test_select_template_deload_due_selects_deload() -> None:
    sessions = [
        _session("s1", "2026-06-29T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s2", "2026-07-06T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s3", "2026-07-13T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s4", "2026-07-20T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
    ]
    template, reason = select_template(DEFAULT_DELOAD_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert template is not None
    assert template["name"] == "deload"
    assert "deload" in reason.lower()


def test_select_template_deload_not_due_selects_default() -> None:
    sessions = [
        _session("s1", "2026-06-29T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s2", "2026-07-06T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
    ]
    template, _ = select_template(DEFAULT_DELOAD_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert template is not None
    assert template["name"] == "strength"


def test_select_template_no_strategy_no_default_selects_nothing() -> None:
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])])]
    template, reason = select_template(None, [STRENGTH_TEMPLATE], sessions)
    assert template is None
    assert reason


def test_select_template_named_template_missing_falls_through() -> None:
    strategy = {
        "version": 1,
        "created": "x",
        "markdown": "```replan-rules\ndefault: nonexistent\n```",
    }
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [])]
    template, reason = select_template(strategy, [STRENGTH_TEMPLATE], sessions)
    assert template is None


# --------------------------------------------------------------------------- #
# decide_auto_replan: normal week (no reorder, no template update)
# --------------------------------------------------------------------------- #
def test_normal_week_selects_template_and_builds_matching_draft() -> None:
    # Monday session performed exactly as scheduled — no reorder needed.
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate"), (5, 5, "plate"), (5, 5, "plate")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)

    assert decision.selected_template == "strength"
    assert decision.activate is True
    assert decision.template_updates == []
    assert isinstance(decision.draft, PlanDraft)
    assert decision.draft.weekStart == "2026-07-27"  # the Monday of that week
    assert decision.draft.weekFocus == "Base strength"
    assert len(decision.draft.workouts) == 7
    monday = next(w for w in decision.draft.workouts if w.weekday == "Monday")
    assert monday.name == "Squat Day"
    wednesday = next(w for w in decision.draft.workouts if w.weekday == "Wednesday")
    assert wednesday.name == "Push Day"  # untouched


def test_subject_workout_not_in_template_skips_reorder() -> None:
    # An ad hoc Session whose name matches nothing in the selected template:
    # no scheduled weekday to compare against, so no reorder is attempted.
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Impromptu Hike", [_activity("Hiking", [(0, None, "sec")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert decision.selected_template == "strength"
    monday = next(w for w in decision.draft.workouts if w.weekday == "Monday")
    assert monday.name == "Squat Day"  # template's own Monday, untouched


def test_template_missing_a_weekday_defaults_to_rest() -> None:
    partial_template = {
        "schemaVersion": 1,
        "name": "partial",
        "category": "strength",
        "weekFocus": "Partial week",
        "workouts": [
            {
                "weekday": "Monday",
                "name": "Squat Day",
                "type": "strength",
                "activities": [
                    {
                        "id": "a1",
                        "exerciseId": "Back Squat",
                        "restSeconds": 90,
                        "prescribedSets": [{"id": "s1", "reps": 5, "value": 5, "unit": "plate"}],
                    }
                ],
            }
        ],
    }
    strategy = {"version": 1, "created": "x", "markdown": "```replan-rules\ndefault: partial\n```"}
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])])]
    decision = decide_auto_replan(strategy, [partial_template], sessions)
    assert len(decision.draft.workouts) == 7
    sunday = next(w for w in decision.draft.workouts if w.weekday == "Sunday")
    assert sunday.type == "rest"


def test_decision_never_raises_on_empty_inputs() -> None:
    decision = decide_auto_replan(None, [], [])
    assert decision.selected_template is None
    assert decision.draft is None
    assert decision.activate is False
    assert decision.template_updates == []
    assert decision.reason


# --------------------------------------------------------------------------- #
# decide_auto_replan: Strategy-directed deload
# --------------------------------------------------------------------------- #
def test_deload_week_selects_deload_template_and_its_content() -> None:
    sessions = [
        _session("s1", "2026-06-29T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s2", "2026-07-06T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s3", "2026-07-13T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session(
            "s4",
            "2026-07-20T10:00:00.000Z",
            "Light Squat Day",
            [_activity("Back Squat", [(5, 2, "plate")])],
        ),
    ]
    decision = decide_auto_replan(DEFAULT_DELOAD_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert decision.selected_template == "deload"
    assert decision.activate is True
    monday = next(w for w in decision.draft.workouts if w.weekday == "Monday")
    assert monday.name == "Light Squat Day"


# --------------------------------------------------------------------------- #
# decide_auto_replan: out-of-order Session reorders the remaining week
# --------------------------------------------------------------------------- #
def test_out_of_order_session_reorders_remaining_week() -> None:
    # Thursday's "Pull Day" was actually performed on Friday.
    sessions = [
        _session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])]),
        _session("s2", "2026-07-31T10:00:00.000Z", "Pull Day", [_activity("Barbell Row", [(8, 3, "plate")])]),
    ]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    workouts = {w.weekday: w for w in decision.draft.workouts}

    assert workouts["Thursday"].name == "Rest"
    assert workouts["Thursday"].type == "rest"
    assert workouts["Friday"].name == "Pull Day"  # the displaced workout, now where it happened
    assert workouts["Saturday"].name == "Leg Day"  # Friday's original, cascaded forward
    assert workouts["Sunday"].name == "Rest"  # Saturday's original was already rest
    # Untouched days stay untouched.
    assert workouts["Monday"].name == "Squat Day"
    assert workouts["Wednesday"].name == "Push Day"


def test_session_performed_early_swaps_the_two_days() -> None:
    # "Pull Day" is scheduled for Thursday but performed a day early, on Wednesday.
    sessions = [_session("s1", "2026-07-29T10:00:00.000Z", "Pull Day", [_activity("Barbell Row", [(8, 3, "plate")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    workouts = {w.weekday: w for w in decision.draft.workouts}

    assert workouts["Wednesday"].name == "Pull Day"  # performed early, now here
    assert workouts["Thursday"].name == "Push Day"  # swapped, nothing lost
    # Everything else untouched.
    assert workouts["Monday"].name == "Squat Day"
    assert workouts["Friday"].name == "Leg Day"


# --------------------------------------------------------------------------- #
# Template freshness sync
# --------------------------------------------------------------------------- #
def test_template_update_proposed_when_performed_sets_differ() -> None:
    sessions = [
        _session(
            "s1",
            "2026-07-27T10:00:00.000Z",
            "Squat Day",
            [_activity("Back Squat", [(5, 6, "plate"), (5, 6, "plate"), (5, 6, "plate")])],  # heavier than template
        )
    ]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert len(decision.template_updates) == 1
    update = decision.template_updates[0]
    assert update.template.name == "strength"
    monday = next(w for w in update.template.workouts if w.weekday == "Monday")
    squat = next(a for a in monday.activities if a.exerciseId == "Back Squat")
    assert all(s.value == 6 for s in squat.prescribedSets)


def test_no_template_update_when_performed_sets_match() -> None:
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate"), (5, 5, "plate"), (5, 5, "plate")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    assert decision.template_updates == []


# --------------------------------------------------------------------------- #
# Kiln-compatible serialization (planDraftInputSchema shape)
# --------------------------------------------------------------------------- #
def test_to_kiln_json_omits_unset_optional_fields() -> None:
    # Kiln's zod schema declares estimatedMinutes/value/unit with .optional()
    # but no paired .nullable() — an explicit JSON null is rejected, so an
    # unset optional field must be omitted from the wire shape entirely, not
    # serialized as null the way a bare Pydantic model_dump() would.
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    payload = to_kiln_json(decision.draft)
    for workout in payload["workouts"]:
        assert "estimatedMinutes" not in workout
    assert payload["schemaVersion"] == 1


def test_bare_model_dump_would_have_failed_kiln_validation() -> None:
    # Documents the gotcha to_kiln_json exists to avoid: the default dump
    # renders the unset optional field as null, the shape zod rejects.
    sessions = [_session("s1", "2026-07-27T10:00:00.000Z", "Squat Day", [_activity("Back Squat", [(5, 5, "plate")])])]
    decision = decide_auto_replan(DEFAULT_STRATEGY, [STRENGTH_TEMPLATE, DELOAD_TEMPLATE], sessions)
    naive = decision.draft.model_dump(mode="json")
    assert naive["workouts"][0]["estimatedMinutes"] is None
