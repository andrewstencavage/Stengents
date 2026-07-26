"""Tests for the passing threshold (#33) — the pure floor logic and the new
required-limitation-recall aggregate that feeds it."""

from __future__ import annotations

from stengents.workout_review.evaluator import AggregateMetrics, CaseResult, aggregate_results
from stengents.workout_review.threshold import (
    QUALITY,
    SAFETY,
    Threshold,
    evaluate_floors,
)


def _aggregate(
    *,
    schema=1.0,
    detail=0.66,
    limitation=0.333,
    evidence=1.0,
    unsupported=0.0,
    cases_passing=5,
    case_count=12,
) -> AggregateMetrics:
    """An AggregateMetrics defaulting to representative 0.2.0 numbers (detail at
    its ~0.66 central, not the 0.708 high-tail single run)."""
    return AggregateMetrics(
        case_count=case_count,
        schema_valid_rate=schema,
        required_detail_recall=detail,
        required_limitation_recall=limitation,
        evidence_validity_rate=evidence,
        unsupported_claim_rate=unsupported,
        cases_passing=cases_passing,
    )


def _by_metric(floors) -> dict[str, object]:
    return {floor.metric: floor for floor in floors}


# --------------------------------------------------------------------------- #
# The 0.2.0 acceptance case (#37): the recorded baseline must clear every floor.
# --------------------------------------------------------------------------- #
def test_representative_0_2_0_run_clears_every_floor() -> None:
    floors = evaluate_floors(_aggregate())
    assert all(floor.passed for floor in floors)


# --------------------------------------------------------------------------- #
# Floor comparators and tiers
# --------------------------------------------------------------------------- #
def test_quality_floors_fail_below_bound() -> None:
    floors = _by_metric(evaluate_floors(_aggregate(detail=0.59, limitation=0.32)))
    assert not floors["required_detail_recall"].passed
    assert not floors["required_limitation_recall"].passed
    assert floors["required_detail_recall"].tier == QUALITY


def test_quality_floors_pass_exactly_at_bound() -> None:
    floors = _by_metric(evaluate_floors(_aggregate(detail=0.60, limitation=0.33)))
    assert floors["required_detail_recall"].passed
    assert floors["required_limitation_recall"].passed


def test_low_tail_0_2_0_run_still_clears_the_detail_floor() -> None:
    # The observed low run (0.632) must pass — the whole point of anchoring the
    # floor below the spread rather than to the 0.708 fluke.
    floors = _by_metric(evaluate_floors(_aggregate(detail=0.632)))
    assert floors["required_detail_recall"].passed


def test_safety_floors_are_zero_tolerance() -> None:
    floors = _by_metric(evaluate_floors(_aggregate(evidence=0.999, unsupported=0.001)))
    assert not floors["evidence_validity_rate"].passed
    assert not floors["unsupported_claim_rate"].passed
    assert floors["evidence_validity_rate"].tier == SAFETY
    assert floors["unsupported_claim_rate"].tier == SAFETY


def test_schema_floor_is_structural_and_must_be_perfect() -> None:
    floors = _by_metric(evaluate_floors(_aggregate(schema=0.99)))
    floor = floors["schema_valid_rate"]
    assert floor.tier == "structural"
    assert not floor.passed


def test_threshold_defaults_match_the_33_decision() -> None:
    threshold = Threshold()
    assert threshold.evidence_validity_min == 1.0
    assert threshold.unsupported_claim_max == 0.0
    assert threshold.schema_valid_min == 1.0
    assert threshold.required_detail_recall_min == 0.60
    assert threshold.required_limitation_recall_min == 0.33


def test_floor_to_dict_carries_verdict() -> None:
    floor = _by_metric(evaluate_floors(_aggregate(detail=0.5)))["required_detail_recall"]
    data = floor.to_dict()
    assert data["metric"] == "required_detail_recall"
    assert data["raw_passed"] is False
    assert data["passed"] is False
    assert data["bound"] == 0.60


# --------------------------------------------------------------------------- #
# required-limitation recall — micro-averaged over cases that require one (#33)
# --------------------------------------------------------------------------- #
def _result(case_id: str, *, required: int, matched: int) -> CaseResult:
    """A minimal CaseResult carrying only the limitation counts under test."""
    return CaseResult(
        case_id=case_id,
        subject_workout_id="w",
        schema_valid=True,
        schema_error=None,
        cited_evidence_total=0,
        cited_evidence_valid=0,
        invalid_evidence=(),
        required_detail_total=0,
        required_detail_matched=0,
        missing_required_evidence=(),
        required_limitations_total=required,
        required_limitations_matched=matched,
        missing_limitations=(),
        forbidden_present=(),
        observation_total=0,
        unsupported_observations=0,
    )


def test_limitation_recall_ignores_cases_without_a_required_limitation() -> None:
    # Two cases require a limitation (1 matched), plus a no-requirement case that
    # must NOT dilute the rate up toward 1.0.
    results = [
        _result("c1", required=1, matched=1),
        _result("c2", required=1, matched=0),
        _result("c3", required=0, matched=0),
    ]
    aggregate = aggregate_results(results)
    # Micro over the 2 requiring cases: 1/2, not 2/3 (which a mean-with-free-1.0 gives).
    assert aggregate.required_limitation_recall == 0.5


def test_limitation_recall_is_one_when_corpus_requires_none() -> None:
    results = [_result("c1", required=0, matched=0), _result("c2", required=0, matched=0)]
    assert aggregate_results(results).required_limitation_recall == 1.0
