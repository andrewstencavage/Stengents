"""Tests for the Auto-replan benchmark corpus and its deterministic evaluator
(issue #64): loads every ``benchmark/<case-id>/`` directory, runs the real
``decide_auto_replan`` against each case's frozen ``input.json``, and scores
the result against ``expectations.json`` with no model judgment — mirroring
``tests/test_workout_review_evaluator.py``'s structure and philosophy.
"""

from __future__ import annotations

from stengents.auto_replan.evaluator import (
    BENCHMARK_DIR,
    Case,
    evaluate_decision,
    load_case,
    load_corpus,
    run_case,
)


def _case(case_id: str) -> Case:
    return load_case(BENCHMARK_DIR / case_id)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def test_load_corpus_returns_all_cases() -> None:
    cases = load_corpus()
    assert len(cases) >= 6
    ids = {c.case_id for c in cases}
    assert {
        "c01-normal-week",
        "c02-strategy-directed-deload",
        "c03-out-of-order-reorder",
    } <= ids


# --------------------------------------------------------------------------- #
# Every case passes against the real decision function
# --------------------------------------------------------------------------- #
def test_every_case_passes_the_real_decision_function() -> None:
    for case in load_corpus():
        decision = run_case(case)
        result = evaluate_decision(decision, case)
        assert result.passed, f"{case.case_id}: {result.checks} mismatched={result.mismatched_weekdays} detail={result.template_update_detail}"


# --------------------------------------------------------------------------- #
# Per-scenario spot checks (acceptance-criteria scenarios named in issue #64)
# --------------------------------------------------------------------------- #
def test_normal_week_case_selects_strength_and_activates() -> None:
    case = _case("c01-normal-week")
    decision = run_case(case)
    assert decision.selected_template == "strength"
    assert decision.activate is True
    assert decision.template_updates == []


def test_deload_case_selects_deload_template() -> None:
    case = _case("c02-strategy-directed-deload")
    decision = run_case(case)
    assert decision.selected_template == "deload"
    monday = next(w for w in decision.draft.workouts if w.weekday == "Monday")
    assert monday.name == "Light Squat Day"


def test_out_of_order_case_reorders_remaining_week() -> None:
    case = _case("c03-out-of-order-reorder")
    decision = run_case(case)
    by_weekday = {w.weekday: w for w in decision.draft.workouts}
    assert by_weekday["Thursday"].type == "rest"
    assert by_weekday["Friday"].name == "Pull Day"
    assert by_weekday["Saturday"].name == "Leg Day"


def test_no_strategy_case_does_not_activate() -> None:
    case = _case("c05-no-strategy-no-activate")
    decision = run_case(case)
    assert decision.selected_template is None
    assert decision.draft is None
    assert decision.activate is False


def test_content_sync_case_proposes_a_template_update() -> None:
    case = _case("c06-template-content-sync")
    decision = run_case(case)
    assert len(decision.template_updates) == 1


# --------------------------------------------------------------------------- #
# The evaluator itself catches a wrong decision (not just a hand-rolled pass)
# --------------------------------------------------------------------------- #
def test_evaluator_flags_wrong_template_selection() -> None:
    case = _case("c01-normal-week")
    decision = run_case(case)
    wrong = decision.model_copy(update={"selected_template": "deload"})
    result = evaluate_decision(wrong, case)
    assert not result.passed
    assert not result.template_correct


def test_evaluator_flags_wrong_activate_flag() -> None:
    case = _case("c01-normal-week")
    decision = run_case(case)
    wrong = decision.model_copy(update={"activate": False})
    result = evaluate_decision(wrong, case)
    assert not result.passed
    assert not result.activate_correct


def test_evaluator_flags_missing_reorder() -> None:
    case = _case("c03-out-of-order-reorder")
    # A decision that never reordered — Thursday still shows Pull Day, not rest.
    decision = run_case(_case("c01-normal-week"))  # wrong shape entirely, reused for its draft-less mismatch
    result = evaluate_decision(decision, case)
    assert not result.passed
    assert "Thursday" in result.mismatched_weekdays or "Friday" in result.mismatched_weekdays


def test_evaluator_flags_schema_invalid_decision() -> None:
    case = _case("c01-normal-week")
    bad = {"selected_template": "strength", "draft": {"not": "valid"}, "activate": True, "reason": "x"}
    result = evaluate_decision(bad, case)
    assert not result.schema_valid
    assert result.schema_error is not None
    assert not result.passed


def test_evaluator_flags_wrong_template_update_content() -> None:
    case = _case("c06-template-content-sync")
    decision = run_case(case)
    update = decision.template_updates[0]
    tampered_workout = next(w for w in update.template.workouts if w.weekday == "Monday")
    tampered_activity = tampered_workout.activities[0]
    tampered_sets = [s.model_copy(update={"value": 99}) for s in tampered_activity.prescribedSets]
    tampered_activity = tampered_activity.model_copy(update={"prescribedSets": tampered_sets})
    tampered_workout = tampered_workout.model_copy(update={"activities": [tampered_activity]})
    workouts = [tampered_workout if w.weekday == "Monday" else w for w in update.template.workouts]
    tampered_template = update.template.model_copy(update={"workouts": workouts})
    tampered_update = update.model_copy(update={"template": tampered_template})
    wrong = decision.model_copy(update={"template_updates": [tampered_update]})
    result = evaluate_decision(wrong, case)
    assert not result.passed
    assert not result.template_update_correct
