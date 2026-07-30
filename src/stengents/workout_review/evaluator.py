"""The deterministic factual evaluator for Workout Reviews (ticket #24).

Scores a generated ``WorkoutReview`` against one benchmark case with **no model
judgment** — every verdict is a verbatim comparison against the case's
``input.json`` pool and its ``expectations.json``. This mirrors the deterministic
style of ``stengents.harness`` / ``stengents.run_record``: pure functions,
structured records, and a repeatable outcome that never depends on a model call.

The checks (fixed in #17/#19/#20):

* **schema-valid** — the review is a valid ``WorkoutReview`` (observations
  uncapped as of ADR 0004, each still needing ≥1 evidence row; re-asserted
  here rather than trusted).
* **evidence grounding** — every cited ``Evidence`` row resolves verbatim to real
  data derivable from the pool: a ``SetEvidence`` row matches a row produced by
  ``performed_sets`` over the referenced Session's activity; a ``FactEvidence``
  row matches a real session- or activity-level field. A row that does not
  resolve is an invented fact.
* **required-detail recall** — the fraction of ``required_evidence`` rows the
  review cites.
* **required limitations present** — every ``{kind, exercise}`` in
  ``required_limitations`` is matched by a review ``Limitation`` of that kind
  whose ``detail`` names the exercise.
* **forbidden observations absent** — no observation whose ``(category,
  derived-exercise)`` matches a ``forbidden_observations`` entry, where an
  observation's exercise is derived from its evidence rows.
* **comparison claims have comparison evidence** — every ``progression``
  observation cites at least one evidence row from a *prior* Session (a different,
  earlier ``workout_id`` than the subject); a progression claim resting only on
  the subject's own sets is unsupported (#19).

The LLM rubric is out of scope: this module never calls a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .contract import (
    Evidence,
    Observation,
    SetEvidence,
    WorkoutReview,
)
from .grounding import resolves as _resolves
from .grounding import sessions_by_id as _sessions_by_id
from .grounding import set_rows_by_session as _set_rows_by_session

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    """One loaded benchmark case: the raw Session pool plus its expectations."""

    case_id: str
    pool: tuple[dict, ...]
    subject_workout_id: str
    required_evidence: tuple[dict, ...]
    required_limitations: tuple[dict, ...]
    forbidden_observations: tuple[dict, ...]


def load_case(case_dir: Path) -> Case:
    """Load a case directory (``input.json`` + ``expectations.json``)."""
    pool = json.loads((case_dir / "input.json").read_text())
    expectations = json.loads((case_dir / "expectations.json").read_text())
    return Case(
        case_id=case_dir.name,
        pool=tuple(pool),
        subject_workout_id=expectations["subject_workout_id"],
        required_evidence=tuple(expectations.get("required_evidence", [])),
        required_limitations=tuple(expectations.get("required_limitations", [])),
        forbidden_observations=tuple(expectations.get("forbidden_observations", [])),
    )


def load_corpus(benchmark_dir: Path = BENCHMARK_DIR) -> list[Case]:
    """Load every case directory under ``benchmark_dir``, sorted by id."""
    return [load_case(p) for p in sorted(benchmark_dir.iterdir()) if p.is_dir()]


# Verbatim evidence resolution lives in :mod:`grounding`, shared with generation
# so the two sides cannot disagree about what "grounds" (imported above as
# ``_resolves`` / ``_sessions_by_id`` / ``_set_rows_by_session``).


# --------------------------------------------------------------------------- #
# Per-observation and per-case scoring
# --------------------------------------------------------------------------- #
def _evidence_key(row: Evidence) -> tuple:
    """A hashable identity for one evidence row, for required-detail matching."""
    if isinstance(row, SetEvidence):
        return ("set", row.workout_id, row.exercise, row.set_index, row.reps, row.load, row.loadType)
    return ("fact", row.workout_id, row.exercise, row.field, row.value)


def _required_key(row: dict) -> tuple:
    if row["kind"] == "set":
        return ("set", row["workout_id"], row["exercise"], row["set_index"], row["reps"], row["load"], row["loadType"])
    return ("fact", row["workout_id"], row.get("exercise"), row["field"], row["value"])


def _derived_exercises(observation: Observation) -> set[str]:
    """The exercise(s) an observation is about, derived from its evidence rows."""
    return {row.exercise for row in observation.evidence if getattr(row, "exercise", None) is not None}


def _session_date(session: dict | None) -> str | None:
    return None if session is None else session.get("date")


@dataclass(frozen=True)
class CaseResult:
    """The deterministic score of one review against one case."""

    case_id: str
    subject_workout_id: str
    schema_valid: bool
    schema_error: str | None
    cited_evidence_total: int
    cited_evidence_valid: int
    invalid_evidence: tuple[tuple, ...]
    required_detail_total: int
    required_detail_matched: int
    missing_required_evidence: tuple[tuple, ...]
    required_limitations_total: int
    required_limitations_matched: int
    missing_limitations: tuple[tuple[str, str], ...]
    forbidden_present: tuple[tuple[str, str], ...]
    observation_total: int
    unsupported_observations: int
    unsupported_reasons: tuple[str, ...] = field(default_factory=tuple)

    # -- per-case metrics (issue #8) -------------------------------------- #
    @property
    def required_detail_recall(self) -> float:
        if self.required_detail_total == 0:
            return 1.0
        return self.required_detail_matched / self.required_detail_total

    @property
    def evidence_validity_rate(self) -> float:
        if self.cited_evidence_total == 0:
            return 1.0
        return self.cited_evidence_valid / self.cited_evidence_total

    @property
    def unsupported_claim_rate(self) -> float:
        if self.observation_total == 0:
            return 0.0
        return self.unsupported_observations / self.observation_total

    @property
    def all_evidence_valid(self) -> bool:
        return self.cited_evidence_valid == self.cited_evidence_total

    @property
    def all_required_detail_recalled(self) -> bool:
        return self.required_detail_matched == self.required_detail_total

    @property
    def all_required_limitations_present(self) -> bool:
        return self.required_limitations_matched == self.required_limitations_total

    @property
    def no_forbidden_observations(self) -> bool:
        return not self.forbidden_present

    @property
    def no_unsupported_claims(self) -> bool:
        return self.unsupported_observations == 0

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "schema_valid": self.schema_valid,
            "all_evidence_valid": self.all_evidence_valid,
            "all_required_detail_recalled": self.all_required_detail_recalled,
            "all_required_limitations_present": self.all_required_limitations_present,
            "no_forbidden_observations": self.no_forbidden_observations,
            "no_unsupported_claims": self.no_unsupported_claims,
        }

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _coerce_review(review: WorkoutReview | dict) -> tuple[WorkoutReview | None, str | None]:
    """Re-assert schema-validity; a dict is validated, a model round-tripped."""
    try:
        if isinstance(review, WorkoutReview):
            return WorkoutReview.model_validate(review.model_dump()), None
        return WorkoutReview.model_validate(review), None
    except ValidationError as error:
        return None, str(error)


def evaluate_review(review: WorkoutReview | dict, case: Case) -> CaseResult:
    """Score one ``WorkoutReview`` against one loaded ``Case`` (pure, no model)."""
    valid_review, schema_error = _coerce_review(review)
    if valid_review is None:
        # Schema-invalid: nothing downstream can be trusted, so the case fails
        # every check without inspecting evidence.
        required_total = len(case.required_evidence)
        return CaseResult(
            case_id=case.case_id,
            subject_workout_id=case.subject_workout_id,
            schema_valid=False,
            schema_error=schema_error,
            cited_evidence_total=0,
            cited_evidence_valid=0,
            invalid_evidence=(),
            required_detail_total=required_total,
            required_detail_matched=0,
            missing_required_evidence=tuple(_required_key(r) for r in case.required_evidence),
            required_limitations_total=len(case.required_limitations),
            required_limitations_matched=0,
            missing_limitations=tuple((r["kind"], r["exercise"]) for r in case.required_limitations),
            forbidden_present=(),
            observation_total=0,
            unsupported_observations=0,
            unsupported_reasons=(),
        )

    sessions = _sessions_by_id(case.pool)
    set_index = _set_rows_by_session(case.pool)
    subject_date = _session_date(sessions.get(case.subject_workout_id))

    # ---- evidence grounding ------------------------------------------- #
    cited_total = 0
    cited_valid = 0
    invalid: list[tuple] = []
    cited_keys: set[tuple] = set()
    for observation in valid_review.observations:
        for row in observation.evidence:
            cited_total += 1
            cited_keys.add(_evidence_key(row))
            if _resolves(row, set_index, sessions):
                cited_valid += 1
            else:
                invalid.append(_evidence_key(row))

    # ---- required-detail recall --------------------------------------- #
    required_keys = [_required_key(r) for r in case.required_evidence]
    matched = [k for k in required_keys if k in cited_keys]
    missing_required = [k for k in required_keys if k not in cited_keys]

    # ---- required limitations present --------------------------------- #
    limitations_matched = 0
    missing_limitations: list[tuple[str, str]] = []
    for required in case.required_limitations:
        kind, exercise = required["kind"], required["exercise"]
        present = any(
            limitation.kind == kind and exercise in limitation.detail
            for limitation in valid_review.limitations
        )
        if present:
            limitations_matched += 1
        else:
            missing_limitations.append((kind, exercise))

    # ---- forbidden observations absent -------------------------------- #
    forbidden_present: list[tuple[str, str]] = []
    for forbidden in case.forbidden_observations:
        category, exercise = forbidden["category"], forbidden["exercise"]
        for observation in valid_review.observations:
            if observation.category == category and exercise in _derived_exercises(observation):
                forbidden_present.append((category, exercise))
                break

    # ---- unsupported claims ------------------------------------------- #
    # An observation is unsupported when it cites an unresolvable row, or when it
    # is a progression claim with no evidence from a prior (earlier) Session.
    unsupported = 0
    reasons: list[str] = []
    for i, observation in enumerate(valid_review.observations):
        row_unresolved = any(not _resolves(row, set_index, sessions) for row in observation.evidence)
        progression_unsupported = False
        if observation.category == "progression":
            has_prior = False
            for row in observation.evidence:
                if row.workout_id == case.subject_workout_id:
                    continue
                other_date = _session_date(sessions.get(row.workout_id))
                if other_date is not None and subject_date is not None and other_date < subject_date:
                    has_prior = True
                    break
            progression_unsupported = not has_prior
        if row_unresolved or progression_unsupported:
            unsupported += 1
            if row_unresolved:
                reasons.append(f"observation[{i}] cites an unresolvable evidence row")
            if progression_unsupported:
                reasons.append(f"observation[{i}] is a progression claim with no prior-Session evidence")

    return CaseResult(
        case_id=case.case_id,
        subject_workout_id=case.subject_workout_id,
        schema_valid=True,
        schema_error=None,
        cited_evidence_total=cited_total,
        cited_evidence_valid=cited_valid,
        invalid_evidence=tuple(invalid),
        required_detail_total=len(required_keys),
        required_detail_matched=len(matched),
        missing_required_evidence=tuple(missing_required),
        required_limitations_total=len(case.required_limitations),
        required_limitations_matched=limitations_matched,
        missing_limitations=tuple(missing_limitations),
        forbidden_present=tuple(forbidden_present),
        observation_total=len(valid_review.observations),
        unsupported_observations=unsupported,
        unsupported_reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------- #
# Aggregate metrics (issue #8)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregateMetrics:
    """The corpus-level roll-up of per-case results — deterministic, repeatable."""

    case_count: int
    schema_valid_rate: float
    required_detail_recall: float
    required_limitation_recall: float
    evidence_validity_rate: float
    unsupported_claim_rate: float
    cases_passing: int

    @property
    def cases_passing_rate(self) -> float:
        if self.case_count == 0:
            return 1.0
        return self.cases_passing / self.case_count


def aggregate_results(results: Iterable[CaseResult]) -> AggregateMetrics:
    """Roll per-case results up into the deterministic corpus metrics (#8)."""
    results = list(results)
    count = len(results)
    if count == 0:
        return AggregateMetrics(0, 1.0, 1.0, 1.0, 1.0, 0.0, 0)

    schema_valid_rate = sum(1 for r in results if r.schema_valid) / count
    required_detail_recall = sum(r.required_detail_recall for r in results) / count

    # required-limitation recall is **micro-averaged over only the cases that
    # require a limitation** (#33): summing matched/total across the corpus, which
    # naturally excludes no-requirement cases (they add 0 to both sums) instead of
    # diluting the rate with a free 1.0 the way a per-case mean would. This keeps
    # the metric tracking exactly the data-quality-limitation failure mode the
    # floor gates. Undefined (→ 1.0) only if no case in the corpus requires one.
    required_limitation_total = sum(r.required_limitations_total for r in results)
    required_limitation_matched = sum(r.required_limitations_matched for r in results)
    required_limitation_recall = (
        1.0 if required_limitation_total == 0 else required_limitation_matched / required_limitation_total
    )

    cited_total = sum(r.cited_evidence_total for r in results)
    cited_valid = sum(r.cited_evidence_valid for r in results)
    evidence_validity_rate = 1.0 if cited_total == 0 else cited_valid / cited_total

    observation_total = sum(r.observation_total for r in results)
    unsupported_total = sum(r.unsupported_observations for r in results)
    unsupported_claim_rate = 0.0 if observation_total == 0 else unsupported_total / observation_total

    cases_passing = sum(1 for r in results if r.passed)

    return AggregateMetrics(
        case_count=count,
        schema_valid_rate=schema_valid_rate,
        required_detail_recall=required_detail_recall,
        required_limitation_recall=required_limitation_recall,
        evidence_validity_rate=evidence_validity_rate,
        unsupported_claim_rate=unsupported_claim_rate,
        cases_passing=cases_passing,
    )
