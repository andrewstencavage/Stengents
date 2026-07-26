import json

import pytest
from pydantic import ValidationError

from stengents.workout_review import (
    FactEvidence,
    Limitation,
    Observation,
    SetEvidence,
    WorkoutReview,
    review_workout,
)
from stengents.workout_review.grounding import Grounding
from stengents.workout_review.review import (
    _candidate_evidence,
    _resolve_evidence,
    select_comparison_history,
)


# --- test data ----------------------------------------------------------


def _activity(name: str, reps_loads: list[tuple[int, float]], unit: str = "plate", **extra) -> dict:
    return {
        "name": name,
        "performedSets": [
            {"reps": reps, "measurement": {"value": load, "unit": unit}} for reps, load in reps_loads
        ],
        **extra,
    }


def _session(session_id: str, date: str, activities: list[dict], **extra) -> dict:
    return {
        "id": session_id,
        "status": "finished",
        "workoutName": "Lower B",
        "type": "strength",
        "date": date,
        "minutes": 40,
        "feel": "good",
        "activities": activities,
        **extra,
    }


def _fake_complete(payload: dict):
    """A model seam that ignores the prompt and returns a canned structured reply."""

    def complete(_model, _prompt):
        return json.dumps(payload)

    return complete


# --- contract invariants (unchanged) ------------------------------------


def test_review_workout_acknowledges_an_unknown_id_instead_of_raising() -> None:
    review = review_workout(
        "nope",
        fetch=lambda _id: None,
        fetch_history=lambda: [],
        model=object(),
        complete=_fake_complete({}),
    )

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


# --- comparison history selection (#19) ---------------------------------


def test_history_selects_same_named_priors_newest_first_capped_at_ten() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    pool = [subject] + [
        _session(f"p{i}", f"2026-07-{10 + i:02d}T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
        for i in range(12)
    ]

    history = select_comparison_history(subject, pool)

    priors = history["Cable Squat"]
    assert len(priors) == 10  # capped at HISTORY_CAP
    dates = [prior["session"]["date"] for prior in priors]
    assert dates == sorted(dates, reverse=True)  # newest first
    assert priors[0]["session"]["id"] == "p11"  # the most recent prior


def test_history_ignores_later_sessions_and_differently_named_activities() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    later = _session("later", "2026-07-25T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    other_name = _session("other", "2026-07-20T10:00:00.000Z", [_activity("Bench Press", [(8, 2.0)])])
    real_prior = _session("prior", "2026-07-21T10:00:00.000Z", [_activity("Cable Squat", [(8, 3.0)])])

    history = select_comparison_history(subject, [subject, later, other_name, real_prior])

    assert [prior["session"]["id"] for prior in history["Cable Squat"]] == ["prior"]


def test_history_excludes_priors_without_performed_sets() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    freeform = _session(
        "freeform", "2026-07-21T10:00:00.000Z", [{"name": "Cable Squat", "spec": "warmup", "sets_completed": 1}]
    )

    history = select_comparison_history(subject, [subject, freeform])

    assert history["Cable Squat"] == []


# --- generation ---------------------------------------------------------


def test_no_priors_yields_an_insufficient_history_limitation() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],  # itself only; no strictly-earlier priors
        model=object(),
        complete=_fake_complete({"summary": "s", "observations": [], "limitations": []}),
    )

    kinds = [limitation.kind for limitation in review.limitations]
    assert "insufficient_history" in kinds
    assert any("Cable Squat" in limitation.detail for limitation in review.limitations)


def test_every_emitted_evidence_row_resolves_to_the_input_data() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0), (8, 3.0)])])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Cable Squat", [(8, 3.0)])])
    pool = [subject, prior]

    selected = select_comparison_history(subject, pool)
    candidates = _candidate_evidence(subject, selected)
    valid = {evidence.model_dump_json() for evidence in candidates}

    # The model cites the first two candidate ids (1-based).
    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: pool,
        model=object(),
        complete=_fake_complete(
            {
                "summary": "Cable Squat performed.",
                "observations": [
                    {
                        "kind": "fact",
                        "confidence": "firm",
                        "category": "performance",
                        "claim": "Two working sets logged.",
                        "evidence_ids": [1, 2],
                    }
                ],
                "limitations": [],
            }
        ),
    )

    assert len(review.observations) == 1
    cited = review.observations[0].evidence
    assert cited  # at least one row
    for row in cited:
        assert row.model_dump_json() in valid  # resolves verbatim to real input data


def test_invented_or_unknown_evidence_ids_are_dropped() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        model=object(),
        complete=_fake_complete(
            {
                "summary": "s",
                "observations": [
                    {
                        "kind": "fact",
                        "confidence": "firm",
                        "category": "performance",
                        "claim": "cites a nonexistent row",
                        "evidence_ids": [9999],  # out of range -> observation dropped
                    }
                ],
                "limitations": [],
            }
        ),
    )

    assert review.observations == []  # evidence-required: no valid row -> no observation


def test_generation_caps_observations_at_three() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    one = {
        "kind": "fact",
        "confidence": "firm",
        "category": "performance",
        "claim": "a set was logged",
        "evidence_ids": [1],
    }

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        model=object(),
        complete=_fake_complete({"summary": "s", "observations": [one] * 5, "limitations": []}),
    )

    assert len(review.observations) <= 3


def test_progression_claim_without_history_is_dropped() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],  # no priors -> zero history for Cable Squat
        model=object(),
        complete=_fake_complete(
            {
                "summary": "s",
                "observations": [
                    {
                        "kind": "inference",
                        "confidence": "tentative",
                        "category": "progression",
                        "claim": "load increased vs last time",
                        "evidence_ids": [1],
                    }
                ],
                "limitations": [],
            }
        ),
    )

    assert review.observations == []  # #19: no progression claim without history
    assert any(limitation.kind == "insufficient_history" for limitation in review.limitations)


def test_a_progression_claim_is_kept_when_history_exists() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.5)])])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    pool = [subject, prior]
    candidates = _candidate_evidence(subject, select_comparison_history(subject, pool))
    subject_set = next(
        i for i, e in enumerate(candidates, start=1) if isinstance(e, SetEvidence) and e.workout_id == "subj"
    )
    prior_set = next(
        i for i, e in enumerate(candidates, start=1) if isinstance(e, SetEvidence) and e.workout_id == "prior"
    )

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: pool,
        model=object(),
        complete=_fake_complete(
            {
                "summary": "s",
                "observations": [
                    {
                        "kind": "inference",
                        "confidence": "firm",
                        "category": "progression",
                        "claim": "load went up from 3 to 3.5 plate",
                        "evidence_ids": [subject_set, prior_set],
                    }
                ],
                "limitations": [],
            }
        ),
    )

    assert [o.category for o in review.observations] == ["progression"]


def test_a_model_failure_degrades_to_a_factual_claim_free_review() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    def boom(_model, _prompt):
        raise RuntimeError("endpoint unreachable")

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        model=object(),
        complete=boom,
    )

    assert isinstance(review, WorkoutReview)
    assert review.workout_id == "subj"
    assert review.observations == []
    assert review.summary  # a deterministic factual summary, not empty


def test_candidate_evidence_is_grounded_in_subject_and_history() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="strong")])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Cable Squat", [(8, 3.0)])])
    candidates = _candidate_evidence(subject, select_comparison_history(subject, [subject, prior]))

    set_ids = {e.workout_id for e in candidates if isinstance(e, SetEvidence)}
    assert set_ids == {"subj", "prior"}  # both subject and history sets are citable
    assert any(isinstance(e, FactEvidence) and e.field == "note" and e.value == "strong" for e in candidates)
    assert any(isinstance(e, FactEvidence) and e.exercise is None and e.field == "feel" for e in candidates)


def test_structured_activity_offers_no_derived_sets_completed_candidate() -> None:
    # A structured activity's count is derived from performedSets, which the
    # evaluator never resolves against a synthesised ``sets_completed`` value; the
    # sets themselves are the grounded citation. So no such candidate is offered.
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0), (8, 3.0), (7, 3.0)])])
    candidates = _candidate_evidence(subject, select_comparison_history(subject, [subject]))

    assert not any(isinstance(e, FactEvidence) and e.field == "sets_completed" for e in candidates)
    # ...and every candidate that IS offered grounds verbatim in the real data.
    grounding = Grounding([subject])
    assert all(grounding.resolves(e) for e in candidates)
    subject_sets = [e for e in candidates if isinstance(e, SetEvidence) and e.workout_id == "subj"]
    assert len(subject_sets) == 3  # the count is citable one-per-set, not as a fact


def test_freeform_activity_still_offers_its_literal_sets_completed() -> None:
    # A freeform activity (no performedSets) carries ``sets_completed`` as a real
    # field, which grounds — that candidate must still be offered.
    freeform = {"name": "Warmup Row", "spec": "5 min easy", "sets_completed": 1}
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [freeform])
    candidates = _candidate_evidence(subject, select_comparison_history(subject, [subject]))

    matches = [e for e in candidates if isinstance(e, FactEvidence) and e.field == "sets_completed"]
    assert [e.value for e in matches] == ["1"]
    assert Grounding([subject]).resolves(matches[0])


def test_resolvability_guard_drops_a_cited_row_that_does_not_ground() -> None:
    # The self-detectable rejection: even if an ungroundable row were offered as a
    # candidate, decode must never surface it — only rows that resolve survive.
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    grounding = Grounding([subject])
    candidates = _candidate_evidence(subject, select_comparison_history(subject, [subject]))

    bogus = SetEvidence(workout_id="subj", exercise="Cable Squat", set_index=99, reps=1, load=999.0, loadType="plate")
    candidates = [*candidates, bogus]
    bogus_id = len(candidates)  # ids are 1-based

    resolved = _resolve_evidence([1, bogus_id], candidates, grounding)

    assert bogus not in resolved  # the ungroundable row is dropped
    assert resolved == [candidates[0]]  # the grounded row (candidate 1) survives
