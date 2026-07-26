import pytest
from pydantic import ValidationError

from stengents.workout_review import (
    Limitation,
    Observation,
    SetEvidence,
    WorkoutReview,
    review_workout,
)


def test_review_workout_returns_a_schema_valid_empty_review_for_a_known_id() -> None:
    session = {"id": "known-1", "workoutName": "Lower B", "date": "2026-07-24T10:00:00.000Z"}

    review = review_workout("known-1", fetch=lambda _id: session, model=object())

    assert isinstance(review, WorkoutReview)
    assert review.workout_id == "known-1"
    assert review.observations == []
    assert review.limitations == []


def test_review_workout_acknowledges_an_unknown_id_instead_of_raising() -> None:
    review = review_workout("nope", fetch=lambda _id: None, model=object())

    assert review.workout_id == "nope"
    assert [limitation.kind for limitation in review.limitations] == ["missing_data"]
    assert review.observations == []


def test_review_hard_caps_observations_at_three() -> None:
    good = Observation(
        kind="fact",
        confidence="firm",
        category="performance",
        claim="did a set",
        evidence=[SetEvidence(workout_id="w", exercise="Squat", set_index=0, reps=5, load=3.0, loadType="plate")],
    )

    with pytest.raises(ValidationError):
        WorkoutReview(workout_id="w", summary="", observations=[good] * 4)


def test_every_observation_requires_at_least_one_evidence_row() -> None:
    with pytest.raises(ValidationError):
        Observation(kind="inference", confidence="tentative", category="progression", claim="up", evidence=[])


def test_evidence_union_discriminates_set_from_fact() -> None:
    obs = Observation(
        kind="fact",
        confidence="firm",
        category="data_quality",
        claim="timed hold logged",
        evidence=[SetEvidence(workout_id="w", exercise="Plank", set_index=0, reps=0, load=45.0, loadType="sec")],
    )

    assert isinstance(obs.evidence[0], SetEvidence)
    assert obs.evidence[0].reps == 0


def test_limitation_rejects_an_unlisted_kind() -> None:
    with pytest.raises(ValidationError):
        Limitation(kind="made_up", detail="x")
