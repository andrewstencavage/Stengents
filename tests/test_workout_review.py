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
from stengents.workout_review.progression import (
    StreakResult,
    plan_streak,
    streak_observation,
    top_set,
    top_set_delta,
)
from stengents.workout_review.review import (
    Partition,
    _candidate_evidence,
    _malformed_set_notes,
    _needs_extraction,
    _resolve_evidence,
    build_extraction_prompt,
    partition_session,
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


def _set(workout_id: str, exercise: str, index: int, reps: int, load: float | None, unit: str = "plate") -> SetEvidence:
    return SetEvidence(workout_id=workout_id, exercise=exercise, set_index=index, reps=reps, load=load, loadType=unit)


def _plan(plan_id: str, week_start: str, status: str, workouts: list[str]) -> dict:
    return {
        "id": plan_id,
        "status": status,
        "weekStart": week_start,
        "days": [{"name": w, "type": "strength", "activities": [{"name": "Ex"}]} for w in workouts],
    }


def _plan_session(session_id: str, plan_id: str, workout_name: str, date: str, status: str = "finished") -> dict:
    return {"id": session_id, "planId": plan_id, "workoutName": workout_name, "status": status, "date": date}


def _fake_complete(payload: dict):
    """A model seam that ignores the prompt and returns a canned structured reply."""

    def complete(_model, _prompt):
        return json.dumps(payload)

    return complete


def _route_complete(extraction: dict):
    """A model seam for extraction calls only — every model call in the new
    pipeline (ADR 0004) is an extraction call; there is no synthesis call to
    route to. ``extraction`` maps a partition key -> that partition's canned
    findings payload; a partition not listed gets no findings.
    """

    def complete(_model, prompt):
        for line in prompt.splitlines():
            if line.startswith("PARTITION: "):
                key = line[len("PARTITION: ") :]
                return json.dumps(extraction.get(key, {"observations": [], "limitations": []}))
        return json.dumps({"observations": [], "limitations": []})

    return complete


def _boom(_model, _prompt):
    raise AssertionError("no model call should have been made")


# --- contract invariants --------------------------------------------------


def test_review_workout_acknowledges_an_unknown_id_instead_of_raising() -> None:
    review = review_workout(
        "nope",
        fetch=lambda _id: None,
        fetch_history=lambda: [],
        fetch_plans=lambda: [],
        model=object(),
        complete=_fake_complete({}),
    )

    assert review.workout_id == "nope"
    assert [limitation.kind for limitation in review.limitations] == ["missing_data"]
    assert review.observations == []


def test_review_no_longer_caps_observations_at_three() -> None:
    # ADR 0004: the #17 contract's 0-3 cap is removed.
    good = Observation(
        kind="fact",
        confidence="firm",
        category="performance",
        claim="did a set",
        evidence=[SetEvidence(workout_id="w", exercise="Squat", set_index=0, reps=5, load=3.0, loadType="plate")],
    )

    review = WorkoutReview(workout_id="w", observations=[good] * 5)
    assert len(review.observations) == 5


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


# --- comparison history selection (#19, unchanged) -----------------------


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


# --- top-set delta (ADR 0004) ---------------------------------------------


def test_top_set_picks_highest_load_ties_broken_by_reps() -> None:
    rows = [_set("w", "Bench", 0, 8, 4.0), _set("w", "Bench", 1, 6, 5.0), _set("w", "Bench", 2, 10, 5.0)]
    assert top_set(rows) == rows[2]  # 5.0 load, more reps than rows[1]


def test_top_set_falls_back_to_reps_for_bodyweight_sets() -> None:
    rows = [_set("w", "Pushup", 0, 12, None), _set("w", "Pushup", 1, 15, None)]
    assert top_set(rows) == rows[1]


def test_top_set_delta_reports_load_change_first() -> None:
    subject = [_set("subj", "Bench", 0, 8, 5.0)]
    prior = [_set("prior", "Bench", 0, 8, 4.0)]
    obs = top_set_delta("Bench", subject, prior)
    assert obs.category == "progression"
    assert obs.kind == "fact"
    assert "up from" in obs.claim
    assert set(obs.evidence) == {*subject, *prior}


def test_top_set_delta_reports_a_decrease() -> None:
    subject = [_set("subj", "Bench", 0, 8, 3.0)]
    prior = [_set("prior", "Bench", 0, 8, 4.0)]
    obs = top_set_delta("Bench", subject, prior)
    assert "down from" in obs.claim


def test_top_set_delta_falls_through_to_set_count_when_load_ties() -> None:
    subject = [_set("subj", "Row", 0, 10, 4.0), _set("subj", "Row", 1, 10, 4.0), _set("subj", "Row", 2, 10, 4.0)]
    prior = [_set("prior", "Row", 0, 10, 4.0), _set("prior", "Row", 1, 10, 4.0)]
    obs = top_set_delta("Row", subject, prior)
    assert "3 sets" in obs.claim and "2 sets" in obs.claim


def test_top_set_delta_falls_through_to_reps_when_load_and_sets_tie() -> None:
    subject = [_set("subj", "Curl", 0, 12, 3.0)]
    prior = [_set("prior", "Curl", 0, 10, 3.0)]
    obs = top_set_delta("Curl", subject, prior)
    assert "reps" in obs.claim and "up from" in obs.claim


def test_top_set_delta_reports_an_exact_tie_as_held_steady() -> None:
    subject = [_set("subj", "Curl", 0, 10, 3.0)]
    prior = [_set("prior", "Curl", 0, 10, 3.0)]
    obs = top_set_delta("Curl", subject, prior)
    assert "held steady" in obs.claim


def test_top_set_delta_is_none_without_sets_on_either_side() -> None:
    assert top_set_delta("Curl", [], [_set("prior", "Curl", 0, 10, 3.0)]) is None
    assert top_set_delta("Curl", [_set("subj", "Curl", 0, 10, 3.0)], []) is None


# --- plan streak (ADR 0004) -----------------------------------------------


def test_plan_streak_is_zero_with_no_prior_plans() -> None:
    result = plan_streak([], [], "2026-07-30T10:00:00.000Z")
    assert result == StreakResult(weeks=0, milestones=())


def test_plan_streak_counts_consecutive_hit_weeks() -> None:
    plans = [
        _plan("p1", "2026-07-20", "historical", ["Upper A", "Lower A"]),
        _plan("p2", "2026-07-13", "historical", ["Upper A", "Lower A"]),
    ]
    sessions = [
        _plan_session("s1", "p1", "Upper A", "2026-07-20T10:00:00.000Z"),
        _plan_session("s2", "p1", "Lower A", "2026-07-22T10:00:00.000Z"),
        _plan_session("s3", "p2", "Upper A", "2026-07-13T10:00:00.000Z"),
        _plan_session("s4", "p2", "Lower A", "2026-07-15T10:00:00.000Z"),
    ]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 2
    assert [m["id"] for m in result.milestones] == ["s4", "s2"]  # oldest hit week first


def test_plan_streak_stops_at_a_week_that_was_not_hit() -> None:
    plans = [
        _plan("p1", "2026-07-20", "historical", ["Upper A"]),
        _plan("p2", "2026-07-13", "historical", ["Upper A"]),
    ]
    sessions = [
        _plan_session("s1", "p1", "Upper A", "2026-07-20T10:00:00.000Z"),
        # p2's Upper A never logged.
    ]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 1


def test_plan_streak_stops_at_a_gap_week_with_no_plan() -> None:
    plans = [_plan("p1", "2026-07-20", "historical", ["Upper A"])]  # nothing for 2026-07-13
    sessions = [_plan_session("s1", "p1", "Upper A", "2026-07-20T10:00:00.000Z")]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 1


def test_plan_streak_ignores_draft_plans() -> None:
    plans = [_plan("p1", "2026-07-20", "draft", ["Upper A"])]
    sessions = [_plan_session("s1", "p1", "Upper A", "2026-07-20T10:00:00.000Z")]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 0  # a draft is not "the" week's Plan


def test_plan_streak_ignores_abandoned_sessions() -> None:
    plans = [_plan("p1", "2026-07-20", "historical", ["Upper A"])]
    sessions = [_plan_session("s1", "p1", "Upper A", "2026-07-20T10:00:00.000Z", status="abandoned")]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 0


def test_plan_streak_excludes_the_current_still_in_progress_week() -> None:
    # A fully-hit Plan for the subject's OWN week must not count — it isn't over yet.
    plans = [_plan("p1", "2026-07-27", "active", ["Upper A"])]
    sessions = [_plan_session("s1", "p1", "Upper A", "2026-07-28T10:00:00.000Z")]
    result = plan_streak(plans, sessions, "2026-07-30T10:00:00.000Z")
    assert result.weeks == 0


def test_streak_observation_is_none_for_a_zero_streak() -> None:
    assert streak_observation(StreakResult(weeks=0, milestones=())) is None


def test_streak_observation_cites_one_evidence_row_per_hit_week() -> None:
    milestones = (
        {"id": "s1", "date": "2026-07-15T10:00:00.000Z"},
        {"id": "s2", "date": "2026-07-22T10:00:00.000Z"},
    )
    obs = streak_observation(StreakResult(weeks=2, milestones=milestones))
    assert obs.category == "adherence"
    assert "2 weeks" in obs.claim
    assert len(obs.evidence) == 2
    assert all(isinstance(row, FactEvidence) and row.field == "date" for row in obs.evidence)


# --- extraction pre-filter (ADR 0004: qualitative-only) -------------------


def test_needs_extraction_false_for_a_boring_exercise() -> None:
    activity = _activity("Cable Squat", [(10, 3.0), (10, 3.0), (10, 3.0)])
    assert _needs_extraction("Cable Squat", activity, malformed={}) is False


def test_needs_extraction_true_when_activity_has_a_note() -> None:
    activity = _activity("Cable Squat", [(10, 3.0), (10, 3.0)], note="felt strong")
    assert _needs_extraction("Cable Squat", activity, malformed={}) is True


def test_needs_extraction_true_when_the_exercise_is_malformed() -> None:
    activity = _activity("Cable Squat", [(10, 3.0), (10, 3.0)])
    assert _needs_extraction("Cable Squat", activity, malformed={"Cable Squat": 1}) is True


def test_needs_extraction_true_when_skipped() -> None:
    activity = _activity("Cable Squat", [(10, 3.0)], skipped=True)
    assert _needs_extraction("Cable Squat", activity, malformed={}) is True


def test_needs_extraction_ignores_within_session_numeric_changes() -> None:
    # Numeric-change detection moved entirely to the deterministic Top-set
    # delta (ADR 0004) — a load/rep change alone no longer triggers a call.
    activity = _activity("Seated Row", [(10, 5.0), (10, 6.0), (10, 6.0)])
    assert _needs_extraction("Seated Row", activity, malformed={}) is False


# --- candidate evidence (extraction only, no history — ADR 0004) ---------


def test_candidate_evidence_offers_only_subject_data_no_history() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="strong")])
    candidates = _candidate_evidence(subject)

    set_ids = {e.workout_id for e in candidates if isinstance(e, SetEvidence)}
    assert set_ids == {"subj"}  # only subject sets, never a prior's
    assert any(isinstance(e, FactEvidence) and e.field == "note" and e.value == "strong" for e in candidates)
    assert any(isinstance(e, FactEvidence) and e.exercise is None and e.field == "feel" for e in candidates)


def test_structured_activity_offers_no_derived_sets_completed_candidate() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0), (8, 3.0), (7, 3.0)])])
    candidates = _candidate_evidence(subject)

    assert not any(isinstance(e, FactEvidence) and e.field == "sets_completed" for e in candidates)
    grounding = Grounding([subject])
    assert all(grounding.resolves(e) for e in candidates)
    subject_sets = [e for e in candidates if isinstance(e, SetEvidence) and e.workout_id == "subj"]
    assert len(subject_sets) == 3


def test_freeform_activity_still_offers_its_literal_sets_completed() -> None:
    freeform = {"name": "Warmup Row", "spec": "5 min easy", "sets_completed": 1}
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [freeform])
    candidates = _candidate_evidence(subject)

    matches = [e for e in candidates if isinstance(e, FactEvidence) and e.field == "sets_completed"]
    assert [e.value for e in matches] == ["1"]
    assert Grounding([subject]).resolves(matches[0])


def test_resolvability_guard_drops_a_cited_row_that_does_not_ground() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])
    grounding = Grounding([subject])
    candidates = _candidate_evidence(subject)

    bogus = SetEvidence(workout_id="subj", exercise="Cable Squat", set_index=99, reps=1, load=999.0, loadType="plate")
    candidates = [*candidates, bogus]
    bogus_id = len(candidates)

    resolved = _resolve_evidence([1, bogus_id], candidates, grounding)

    assert bogus not in resolved
    assert resolved == [candidates[0]]


def test_malformed_set_notes_counts_dropped_subject_sets() -> None:
    subject = {
        "id": "subj",
        "status": "finished",
        "date": "2026-07-25T14:05:00.000Z",
        "workoutName": "Upper B",
        "type": "strength",
        "activities": [
            {
                "name": "Chest Fly",
                "performedSets": [
                    {"reps": 12, "measurement": {"value": 2, "unit": "plate"}},  # valid
                    {"reps": 12, "measurement": {"value": 2}},  # missing unit -> dropped
                    {"reps": 11, "measurement": None},  # null measurement -> dropped
                ],
            },
            _activity("Cable Squat", [(10, 3.0)]),  # all clean
        ],
    }
    notes = _malformed_set_notes(subject)
    assert notes == {"Chest Fly": 2}


def test_clean_session_has_no_malformed_notes() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0), (8, 3.0)])])
    assert _malformed_set_notes(subject) == {}


# --- extraction prompt (comparison history stays as context — ADR 0004) --


def test_extraction_prompt_shows_comparison_history_as_context_only() -> None:
    # Reverted from an initial "drop it entirely" cut: a live A/B showed the
    # model needs this context for judgment even though it can't cite it for
    # progression (c11's malformed-vs-missing regression, reproduced 4/4).
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Chest Fly", [(10, 3.0)], note="ok")])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Chest Fly", [(10, 2.5)])])
    selected = select_comparison_history(subject, [subject, prior])
    candidates = _candidate_evidence(subject)
    partition = Partition(key="Chest Fly", candidate_ids=tuple(range(1, len(candidates) + 1)))

    prompt = build_extraction_prompt(partition, candidates, selected, {}, set())
    assert "COMPARISON HISTORY" in prompt
    assert "context only" in prompt
    assert "'progression'" in prompt  # the instruction telling it NOT to use the category


def test_extraction_prompt_omits_comparison_history_with_no_priors() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Chest Fly", [(10, 3.0)], note="ok")])
    selected = select_comparison_history(subject, [subject])
    candidates = _candidate_evidence(subject)
    partition = Partition(key="Chest Fly", candidate_ids=tuple(range(1, len(candidates) + 1)))

    prompt = build_extraction_prompt(partition, candidates, selected, {}, set())
    assert "COMPARISON HISTORY" not in prompt


def test_extraction_prompt_surfaces_data_quality_notes() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Chest Fly", [(10, 3.0)], note="ok")])
    selected = select_comparison_history(subject, [subject])
    candidates = _candidate_evidence(subject)
    partition = Partition(key="Chest Fly", candidate_ids=tuple(range(1, len(candidates) + 1)))

    header = "DATA QUALITY NOTE:"
    without = build_extraction_prompt(partition, candidates, selected, {}, set())
    assert header not in without

    with_notes = build_extraction_prompt(partition, candidates, selected, {"Chest Fly": 2}, set())
    assert header in with_notes
    assert "2 performed sets" in with_notes
    inline = "You MUST add a 'malformed_data'"
    assert inline in with_notes
    assert inline not in without


def test_extraction_prompt_surfaces_a_skipped_note() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Chest Fly", [(10, 3.0)], skipped=True)])
    selected = select_comparison_history(subject, [subject])
    candidates = _candidate_evidence(subject)
    partition = Partition(key="Chest Fly", candidate_ids=tuple(range(1, len(candidates) + 1)))

    without = build_extraction_prompt(partition, candidates, selected, {}, set())
    assert "was marked skipped" not in without

    with_note = build_extraction_prompt(partition, candidates, selected, {}, {"Chest Fly"})
    assert "'Chest Fly' was marked skipped" in with_note
    assert "You MUST add a 'missing_data' limitation naming 'Chest Fly'" in with_note


# --- partition_session (ADR 0004: no more baseline findings here) --------


def test_partition_session_skips_a_boring_exercise_entirely() -> None:
    # No note, no priors, not malformed: nothing that needs a model call. Its
    # baseline recap is now built directly in _generate_review, not here.
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0), (10, 3.0)])])
    selected = select_comparison_history(subject, [subject])
    candidates = _candidate_evidence(subject)

    partitions = partition_session(subject, selected, candidates, {})

    assert partitions == []


def test_partition_session_keeps_an_exercise_that_needs_extraction() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Seated Row", [(10, 5.0)], note="felt heavy")])
    selected = select_comparison_history(subject, [subject])
    candidates = _candidate_evidence(subject)

    partitions = partition_session(subject, selected, candidates, {})

    keys = [p.key for p in partitions]
    assert "Seated Row" in keys
    seated_row = next(p for p in partitions if p.key == "Seated Row")
    assert seated_row.candidate_ids


# --- review_workout end to end --------------------------------------------


def test_no_priors_yields_an_insufficient_history_limitation_and_a_baseline_observation() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        fetch_plans=lambda: [],
        model=object(),
        complete=_boom,  # a boring, no-history exercise needs no model call at all
    )

    kinds = [limitation.kind for limitation in review.limitations]
    assert "insufficient_history" in kinds
    assert any("Cable Squat" in limitation.detail for limitation in review.limitations)
    assert [o.category for o in review.observations] == ["performance"]
    assert "Cable Squat" in review.observations[0].claim


def test_an_exercise_with_history_gets_a_deterministic_progression_observation_and_no_model_call() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.5)])])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject, prior],
        fetch_plans=lambda: [],
        model=object(),
        complete=_boom,  # no note, no skip, no malformed data -> no extraction call needed
    )

    assert [o.category for o in review.observations] == ["progression"]
    assert "up from" in review.observations[0].claim
    assert not any(limitation.kind == "insufficient_history" for limitation in review.limitations)


def test_invented_or_unknown_evidence_ids_are_dropped() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="ok")])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        fetch_plans=lambda: [],
        model=object(),
        complete=_route_complete(
            {
                "Cable Squat": {
                    "observations": [
                        {
                            "kind": "fact",
                            "confidence": "firm",
                            "category": "adherence",
                            "claim": "cites a nonexistent row",
                            "evidence_ids": [9999],  # out of range -> finding dropped
                        }
                    ],
                    "limitations": [],
                }
            }
        ),
    )

    # No priors -> a baseline performance observation, but the invented-evidence
    # extraction finding never survives.
    assert [o.category for o in review.observations] == ["performance"]


def test_extraction_is_barred_from_proposing_progression() -> None:
    # Even with priors present, a model that (against instructions) tries to
    # emit category="progression" from extraction is filtered at decode time —
    # that comparison is deterministic-only now (ADR 0004).
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="ok")])
    prior = _session("prior", "2026-07-20T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])])

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject, prior],
        fetch_plans=lambda: [],
        model=object(),
        complete=_route_complete(
            {
                "Cable Squat": {
                    "observations": [
                        {
                            "kind": "inference",
                            "confidence": "tentative",
                            "category": "progression",
                            "claim": "model-inferred progression, should be dropped",
                            "evidence_ids": [1],
                        }
                    ],
                    "limitations": [],
                }
            }
        ),
    )

    categories = [o.category for o in review.observations]
    assert "progression" in categories  # from the deterministic Top-set delta
    assert categories.count("progression") == 1  # the extraction-sourced one was filtered out
    assert "model-inferred progression, should be dropped" not in [o.claim for o in review.observations]


def test_uncapped_extraction_findings_are_all_kept() -> None:
    # ADR 0004: no selection, no cap — every valid extraction finding survives.
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="ok")])
    one = {
        "kind": "fact",
        "confidence": "firm",
        "category": "adherence",
        "claim": "a set was logged",
        "evidence_ids": [1],
    }

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        fetch_plans=lambda: [],
        model=object(),
        complete=_route_complete({"Cable Squat": {"observations": [one] * 5, "limitations": []}}),
    )

    extraction_observations = [o for o in review.observations if o.category == "adherence"]
    assert len(extraction_observations) == 5


def test_a_model_failure_still_yields_the_deterministic_observations() -> None:
    subject = _session("subj", "2026-07-24T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)], note="ok")])

    def boom(_model, _prompt):
        raise RuntimeError("endpoint unreachable")

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: [subject],
        fetch_plans=lambda: [],
        model=object(),
        complete=boom,
    )

    assert isinstance(review, WorkoutReview)
    assert review.workout_id == "subj"
    # No priors -> the deterministic baseline observation survives a broken model call.
    assert [o.category for o in review.observations] == ["performance"]


def test_a_plan_streak_flows_end_to_end_into_the_review() -> None:
    subject = _session(
        "subj", "2026-07-30T10:00:00.000Z", [_activity("Cable Squat", [(10, 3.0)])], planId="current-plan"
    )
    plans = [_plan("p1", "2026-07-20", "historical", ["Cable Squat"])]
    pool = [subject, _plan_session("s1", "p1", "Cable Squat", "2026-07-20T10:00:00.000Z")]

    review = review_workout(
        "subj",
        fetch=lambda _id: subject,
        fetch_history=lambda: pool,
        fetch_plans=lambda: plans,
        model=object(),
        complete=_boom,
    )

    streaks = [o for o in review.observations if o.category == "adherence"]
    assert len(streaks) == 1
    assert "1 week" in streaks[0].claim
