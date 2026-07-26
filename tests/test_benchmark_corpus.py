"""Structural validation of the Workout Review benchmark corpus (ticket #22).

These are not the eventual deterministic *evaluator* (that scores a generated
``WorkoutReview``). They assert the frozen corpus itself is well-formed: every
case parses, points at a real subject, cites only Evidence that validates
against the #17 contract and resolves verbatim to the input, names only
in-vocabulary limitation kinds and observation categories, and feeds the real
``performed_sets`` extraction without error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from farm_system.kiln_coach.kiln_client import finished_sessions, performed_sets
from stengents.workout_review.contract import (
    Category,
    FactEvidence,
    LimitationKind,
    SetEvidence,
)
from stengents.workout_review.grounding import Grounding
from stengents.workout_review.review import _candidate_evidence, select_comparison_history

# Locate the corpus by filesystem path from this test file so discovery is
# independent of how the ``stengents`` package resolves on sys.path.
BENCHMARK = Path(__file__).resolve().parent.parent / "src" / "stengents" / "workout_review" / "benchmark"

_CATEGORIES = set(Category.__args__)
_LIMITATION_KINDS = set(LimitationKind.__args__)


def _case_dirs() -> list[Path]:
    return sorted(p for p in BENCHMARK.iterdir() if p.is_dir())


CASE_IDS = [p.name for p in _case_dirs()]


def test_corpus_has_ten_to_twenty_cases() -> None:
    assert 10 <= len(CASE_IDS) <= 20, f"corpus must hold 10-20 cases, found {len(CASE_IDS)}"


def test_coverage_note_exists() -> None:
    assert (BENCHMARK / "coverage.md").is_file()


def _load(case_id: str) -> tuple[list[dict], dict]:
    case = BENCHMARK / case_id
    pool = json.loads((case / "input.json").read_text())
    expectations = json.loads((case / "expectations.json").read_text())
    return pool, expectations


def _set_rows_by_session(pool: list[dict]) -> dict[str, dict[tuple[str, int], dict]]:
    """Every performed set in the pool, keyed by (session id) -> (exercise, index)."""
    index: dict[str, dict[tuple[str, int], dict]] = {}
    for session in pool:
        rows: dict[tuple[str, int], dict] = {}
        for activity in session.get("activities") or []:
            for row in performed_sets(activity):
                rows[(activity.get("name"), row["set_index"])] = row
        index[session["id"]] = rows
    return index


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_input_parses_to_a_list_of_sessions(case_id: str) -> None:
    pool, _ = _load(case_id)
    assert isinstance(pool, list) and pool, "input.json must be a non-empty list"
    assert all(isinstance(session, dict) and "id" in session for session in pool)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_subject_workout_id_exists_in_the_pool(case_id: str) -> None:
    pool, expectations = _load(case_id)
    ids = {session["id"] for session in pool}
    assert expectations["subject_workout_id"] in ids


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_performed_sets_runs_on_the_subject_without_error(case_id: str) -> None:
    pool, expectations = _load(case_id)
    subject = next(s for s in pool if s["id"] == expectations["subject_workout_id"])
    for activity in subject.get("activities") or []:
        # Must not raise, even for the malformed/missing-field cases.
        assert isinstance(performed_sets(activity), list)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_required_evidence_validates_against_the_contract(case_id: str) -> None:
    _, expectations = _load(case_id)
    for row in expectations["required_evidence"]:
        kind = row["kind"]
        if kind == "set":
            SetEvidence.model_validate(row)
        elif kind == "fact":
            FactEvidence.model_validate(row)
        else:  # pragma: no cover - guards a typo'd fixture
            pytest.fail(f"unknown evidence kind {kind!r} in {case_id}")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_required_set_evidence_resolves_verbatim_to_the_input(case_id: str) -> None:
    pool, expectations = _load(case_id)
    by_session = _set_rows_by_session(pool)
    for row in expectations["required_evidence"]:
        if row["kind"] != "set":
            continue
        rows = by_session.get(row["workout_id"], {})
        source = rows.get((row["exercise"], row["set_index"]))
        assert source is not None, f"{case_id}: no set for {row['exercise']}#{row['set_index']}"
        assert source["reps"] == row["reps"]
        assert source["load"] == row["load"]
        assert source["loadType"] == row["loadType"]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_required_fact_evidence_targets_a_real_session(case_id: str) -> None:
    pool, expectations = _load(case_id)
    ids = {session["id"] for session in pool}
    for row in expectations["required_evidence"]:
        if row["kind"] != "fact":
            continue
        assert row["workout_id"] in ids


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_required_limitations_use_the_contract_vocab(case_id: str) -> None:
    _, expectations = _load(case_id)
    for limitation in expectations["required_limitations"]:
        assert limitation["kind"] in _LIMITATION_KINDS
        assert limitation.get("exercise")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_forbidden_observations_use_the_contract_vocab(case_id: str) -> None:
    _, expectations = _load(case_id)
    for forbidden in expectations["forbidden_observations"]:
        assert forbidden["category"] in _CATEGORIES
        assert forbidden.get("exercise")
        assert forbidden.get("why"), "a forbidden observation must say why"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_every_generator_candidate_grounds_for_each_case(case_id: str) -> None:
    # The generator must offer the model only Evidence that grounds verbatim.
    # A candidate that cannot resolve (the old synthesised ``sets_completed``
    # count on structured activities) leads the model to cite what the evaluator
    # scores as an invented fact — the entire source of the 0.1.0 baseline's
    # unsupported claims. Grounding every candidate the generator builds for each
    # real subject guards against that class of divergence returning.
    pool, expectations = _load(case_id)
    subject = next(s for s in pool if s["id"] == expectations["subject_workout_id"])
    selected = select_comparison_history(subject, finished_sessions(pool))
    candidates = _candidate_evidence(subject, selected)
    grounding = Grounding(
        [subject, *(prior["session"] for priors in selected.values() for prior in priors)]
    )

    assert candidates, f"{case_id}: subject offers no citable evidence"
    ungrounded = [e for e in candidates if not grounding.resolves(e)]
    assert not ungrounded, f"{case_id}: generator offered ungroundable candidates: {ungrounded}"
