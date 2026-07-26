"""Tests for the deterministic factual evaluator (ticket #24).

Hand-crafted ``WorkoutReview``s — one clean passing review and several
deliberately-bad ones — are scored against real corpus cases. Each of the six
deterministic checks and each of the five #8 metrics is exercised so a regression
in any single verdict fails a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stengents.workout_review.contract import (
    FactEvidence,
    Limitation,
    Observation,
    SetEvidence,
    WorkoutReview,
)
from stengents.workout_review.evaluator import (
    BENCHMARK_DIR,
    Case,
    aggregate_results,
    evaluate_review,
    load_case,
    load_corpus,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ev(row: dict) -> SetEvidence | FactEvidence:
    return SetEvidence.model_validate(row) if row["kind"] == "set" else FactEvidence.model_validate(row)


def _case(case_id: str) -> Case:
    return load_case(BENCHMARK_DIR / case_id)


def _gold_review(case: Case) -> WorkoutReview:
    """A clean review that satisfies every deterministic check for ``case``.

    Cites all required evidence under a single (non-progression) observation and
    names every required limitation's exercise in a ``Limitation.detail``.
    """
    evidence = [_ev(row) for row in case.required_evidence]
    observations = (
        [Observation(kind="fact", confidence="firm", category="performance", claim="c", evidence=evidence)]
        if evidence
        else []
    )
    limitations = [
        Limitation(kind=row["kind"], detail=f"{row['exercise']} was not fully performed")
        for row in case.required_limitations
    ]
    return WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="factual summary",
        observations=observations,
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def test_load_corpus_returns_all_cases() -> None:
    cases = load_corpus()
    assert 10 <= len(cases) <= 20
    assert all(case.subject_workout_id in {s["id"] for s in case.pool} for case in cases)


# --------------------------------------------------------------------------- #
# The clean passing review
# --------------------------------------------------------------------------- #
def test_clean_review_passes_every_check_for_all_cases() -> None:
    for case in load_corpus():
        result = evaluate_review(_gold_review(case), case)
        assert result.passed, f"{case.case_id}: {result.checks}"
        assert result.schema_valid
        assert result.evidence_validity_rate == 1.0
        assert result.required_detail_recall == 1.0
        assert result.unsupported_claim_rate == 0.0


def test_clean_progression_review_with_prior_evidence_is_supported() -> None:
    # c02 subject bench is 4 plate; the prior Session is 3 plate — a supported
    # progression claim because it cites an earlier Session.
    case = _case("c02-improvement-vs-history")
    subject, prior = case.required_evidence[0], case.required_evidence[1]
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="up from last time",
        observations=[
            Observation(
                kind="inference",
                confidence="firm",
                category="progression",
                claim="heavier than the prior session",
                evidence=[_ev(subject), _ev(prior)],
            )
        ],
    )
    result = evaluate_review(review, case)
    assert result.passed
    assert result.no_unsupported_claims
    assert result.unsupported_observations == 0


# --------------------------------------------------------------------------- #
# Invented evidence (grounding)
# --------------------------------------------------------------------------- #
def test_invented_evidence_fails_grounding_and_recall() -> None:
    case = _case("c02-improvement-vs-history")
    good = _ev(case.required_evidence[1])  # the real prior-session set
    invented = SetEvidence(
        kind="set",
        workout_id=case.subject_workout_id,
        exercise="Bench Press",
        set_index=0,
        reps=99,  # no such set exists
        load=4,
        loadType="plate",
    )
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(
                kind="fact",
                confidence="firm",
                category="performance",
                claim="c",
                evidence=[good, invented],
            )
        ],
    )
    result = evaluate_review(review, case)
    assert not result.passed
    assert not result.all_evidence_valid
    assert result.cited_evidence_total == 2
    assert result.cited_evidence_valid == 1
    assert result.evidence_validity_rate == 0.5
    assert len(result.invalid_evidence) == 1
    # The invented set means the observation is unsupported.
    assert result.unsupported_observations == 1
    assert result.unsupported_claim_rate == 1.0
    # It also drops recall: the subject's real set0 was never cited.
    assert result.required_detail_recall < 1.0


def test_invalid_fact_value_does_not_resolve() -> None:
    case = _case("c05-incomplete-workout")
    wrong = FactEvidence(
        kind="fact",
        workout_id=case.subject_workout_id,
        exercise="Overhead Press",
        field="planned_sets",
        value="7",  # real value is "4"
    )
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(kind="fact", confidence="firm", category="adherence", claim="c", evidence=[wrong])
        ],
    )
    result = evaluate_review(review, case)
    assert result.cited_evidence_valid == 0
    assert not result.all_evidence_valid


# --------------------------------------------------------------------------- #
# Forbidden observations
# --------------------------------------------------------------------------- #
def test_forbidden_progression_observation_is_flagged() -> None:
    # c04 forbids a progression claim about Hack Squat (first-ever instance).
    case = _case("c04-exercise-substitution")
    hack_set = _ev(case.required_evidence[0])  # the Hack Squat set
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(
                kind="inference",
                confidence="tentative",
                category="progression",
                claim="stronger on hack squat",
                evidence=[hack_set],
            )
        ],
    )
    result = evaluate_review(review, case)
    assert not result.passed
    assert not result.no_forbidden_observations
    assert ("progression", "Hack Squat") in result.forbidden_present
    # First-ever movement: the progression claim is also unsupported.
    assert result.unsupported_observations == 1


# --------------------------------------------------------------------------- #
# Missing required limitation
# --------------------------------------------------------------------------- #
def test_missing_required_limitation_is_flagged() -> None:
    # c07 requires an insufficient_history limitation naming Goblet Squat.
    case = _case("c07-insufficient-history")
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(
                kind="fact",
                confidence="firm",
                category="performance",
                claim="c",
                evidence=[_ev(case.required_evidence[0])],
            )
        ],
        limitations=[],  # the required limitation is absent
    )
    result = evaluate_review(review, case)
    assert not result.passed
    assert not result.all_required_limitations_present
    assert result.required_limitations_matched == 0
    assert ("insufficient_history", "Goblet Squat") in result.missing_limitations


def test_limitation_of_right_kind_not_naming_exercise_does_not_match() -> None:
    case = _case("c07-insufficient-history")
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(
                kind="fact",
                confidence="firm",
                category="performance",
                claim="c",
                evidence=[_ev(case.required_evidence[0])],
            )
        ],
        # Right kind, but does not name Goblet Squat.
        limitations=[Limitation(kind="insufficient_history", detail="not enough prior data")],
    )
    result = evaluate_review(review, case)
    assert not result.all_required_limitations_present


# --------------------------------------------------------------------------- #
# Progression claim with no comparison evidence
# --------------------------------------------------------------------------- #
def test_progression_claim_without_prior_evidence_is_unsupported() -> None:
    # A progression claim resting only on the subject's own set — grounded, but
    # unsupported because it cites no earlier Session (#19).
    case = _case("c02-improvement-vs-history")
    subject_set = _ev(case.required_evidence[0])  # resolves to the subject
    assert subject_set.workout_id == case.subject_workout_id
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(
                kind="inference",
                confidence="firm",
                category="progression",
                claim="I improved",
                evidence=[subject_set],
            )
        ],
    )
    result = evaluate_review(review, case)
    assert result.all_evidence_valid  # grounding is fine
    assert result.evidence_validity_rate == 1.0
    assert not result.no_unsupported_claims  # but the claim is unsupported
    assert result.unsupported_observations == 1
    assert result.unsupported_claim_rate == 1.0
    assert not result.passed


# --------------------------------------------------------------------------- #
# Schema validity
# --------------------------------------------------------------------------- #
def test_schema_invalid_review_fails_and_zeroes_downstream() -> None:
    case = _case("c02-improvement-vs-history")
    # Four observations violates the ≤3 hard cap — invalid as a dict payload.
    bad = {
        "workout_id": case.subject_workout_id,
        "summary": "s",
        "observations": [
            {
                "kind": "fact",
                "confidence": "firm",
                "category": "performance",
                "claim": "c",
                "evidence": [case.required_evidence[0]],
            }
        ]
        * 4,
        "limitations": [],
    }
    result = evaluate_review(bad, case)
    assert not result.schema_valid
    assert result.schema_error is not None
    assert not result.passed
    assert result.cited_evidence_total == 0


def test_evidence_free_observation_is_schema_invalid() -> None:
    case = _case("c02-improvement-vs-history")
    bad = {
        "workout_id": case.subject_workout_id,
        "summary": "s",
        "observations": [
            {"kind": "fact", "confidence": "firm", "category": "performance", "claim": "c", "evidence": []}
        ],
        "limitations": [],
    }
    result = evaluate_review(bad, case)
    assert not result.schema_valid


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def test_aggregate_over_gold_reviews_is_perfect() -> None:
    results = [evaluate_review(_gold_review(case), case) for case in load_corpus()]
    aggregate = aggregate_results(results)
    assert aggregate.case_count == len(results)
    assert aggregate.schema_valid_rate == 1.0
    assert aggregate.required_detail_recall == 1.0
    assert aggregate.evidence_validity_rate == 1.0
    assert aggregate.unsupported_claim_rate == 0.0
    assert aggregate.cases_passing == aggregate.case_count
    assert aggregate.cases_passing_rate == 1.0


def test_aggregate_reflects_a_single_bad_case() -> None:
    corpus = load_corpus()
    results = [evaluate_review(_gold_review(case), case) for case in corpus]
    # Replace one case's review with a schema-invalid one.
    bad_case = next(c for c in corpus if c.case_id == "c02-improvement-vs-history")
    results = [r for r in results if r.case_id != bad_case.case_id]
    results.append(evaluate_review({"workout_id": "x", "summary": "s", "observations": [{}] * 4}, bad_case))
    aggregate = aggregate_results(results)
    assert aggregate.case_count == len(corpus)
    assert aggregate.schema_valid_rate == (len(corpus) - 1) / len(corpus)
    assert aggregate.cases_passing == len(corpus) - 1


def test_aggregate_of_empty_results_is_neutral() -> None:
    aggregate = aggregate_results([])
    assert aggregate.case_count == 0
    assert aggregate.schema_valid_rate == 1.0
    assert aggregate.cases_passing == 0
    assert aggregate.cases_passing_rate == 1.0


def test_evidence_validity_rate_is_row_share_across_cases() -> None:
    # One case with a half-invalid observation; validity rate is the row share.
    case = _case("c02-improvement-vs-history")
    good = _ev(case.required_evidence[0])
    invented = SetEvidence(
        kind="set", workout_id=case.subject_workout_id, exercise="Bench Press",
        set_index=9, reps=1, load=1, loadType="plate",
    )
    review = WorkoutReview(
        workout_id=case.subject_workout_id,
        summary="s",
        observations=[
            Observation(kind="fact", confidence="firm", category="performance", claim="c", evidence=[good, invented])
        ],
    )
    aggregate = aggregate_results([evaluate_review(review, case)])
    assert aggregate.evidence_validity_rate == 0.5
