"""Run the Workout Review capability over the whole benchmark corpus.

The runner wires the three finished pieces into one pass — corpus (#22) →
`review_workout` (#23) → `evaluate_review` (#24) → `aggregate_results` — and
records the result as a minimal, stamped metrics artifact. Only the model call
inside `review_workout` is live; everything else runs on the case's frozen
`input.json`, so the runner is unit-testable offline by injecting a fake
``complete``.

This is the "minimal recording" cut (decided while charting #29): the artifact
carries the aggregate, the per-case results, each generated review, and the
stamps needed to make a score repeatable — the corpus content-hash (#20), the
model record, and the capability version. The fuller `run_record`/`turn_log`
integration stays fog until the baseline shows what is worth capturing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stengents.utilities.model_source import ModelConnection

from . import CAPABILITY_VERSION
from .contract import WorkoutReview
from .evaluator import (
    BENCHMARK_DIR,
    AggregateMetrics,
    Case,
    CaseResult,
    aggregate_results,
    evaluate_review,
    load_corpus,
)
from .review import Complete, review_workout
from .threshold import GateReport, RetestReport, Threshold, evaluate_floors

# On a safety-gate violation, each offending case is re-generated up to this many
# times to tell a reproducible defect from a one-off flake (#33).
SAFETY_RETEST_N = 3

# Where a benchmark run's full artifact is written, mirroring `.stengents/runs/`.
# Transient and gitignored — one file per run, carrying the generated reviews.
ARTIFACT_DIR = Path(".stengents/benchmark")

# Where a *recorded* baseline is committed — tracked, one per (capability, model,
# corpus). Trimmed of the model-nondeterministic reviews so it reads as a stable
# reference the tune loop can diff future runs against.
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


def corpus_hash(benchmark_dir: Path = BENCHMARK_DIR) -> str:
    """The corpus content-hash (#20): a SHA-256 over every case's normalized
    ``input.json`` + ``expectations.json``, sorted by case-id.

    Normalizing with sorted keys and compact separators means whitespace or key
    ordering never changes the hash — only the actual case content does. Any
    added or edited case yields a different hash, correctly marking a different
    corpus.
    """
    digest = hashlib.sha256()
    for case_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        digest.update(case_dir.name.encode())
        for filename in ("input.json", "expectations.json"):
            content = json.loads((case_dir / filename).read_text())
            normalized = json.dumps(content, sort_keys=True, separators=(",", ":"))
            digest.update(filename.encode())
            digest.update(normalized.encode())
    return digest.hexdigest()


def run_benchmark(
    cases: list[Case],
    *,
    model: ModelConnection | None = None,
    complete: Complete | None = None,
) -> tuple[list[CaseResult], AggregateMetrics, dict[str, WorkoutReview]]:
    """Review and score every case, feeding each its own frozen Session pool.

    Each case's `input.json` pool is handed to `review_workout` through its
    `fetch`/`fetch_history` seams, so the corpus data never touches Kiln; only
    the model call is live (or the injected ``complete``). Returns the per-case
    results, the aggregate, and the generated reviews keyed by case-id.
    """
    results: list[CaseResult] = []
    reviews: dict[str, WorkoutReview] = {}
    for case in cases:
        review = _review_case(case, model=model, complete=complete)
        reviews[case.case_id] = review
        results.append(evaluate_review(review, case))
    return results, aggregate_results(results), reviews


def _review_case(
    case: Case, *, model: ModelConnection | None = None, complete: Complete | None = None
) -> WorkoutReview:
    """Generate one review for ``case``, feeding it only its own frozen pool.

    The single place the review's `fetch`/`fetch_history` seams are wired to a
    case's `input.json`, so `run_benchmark` and the safety retest generate
    identically.
    """
    by_id = {session.get("id"): session for session in case.pool}
    return review_workout(
        case.subject_workout_id,
        fetch=(lambda workout_id, _by_id=by_id: _by_id.get(workout_id)),
        fetch_history=(lambda _pool=case.pool: list(_pool)),
        model=model,
        complete=complete,
    )


# --------------------------------------------------------------------------- #
# The passing gate + confirm-before-failing retest (#33/#37)
# --------------------------------------------------------------------------- #
def _violates(result: CaseResult, metric: str) -> bool:
    """Whether one case violates a given safety metric."""
    if metric == "evidence_validity_rate":
        return not result.all_evidence_valid
    if metric == "unsupported_claim_rate":
        return not result.no_unsupported_claims
    return False


def gate_benchmark(
    cases: list[Case],
    results: list[CaseResult],
    aggregate: AggregateMetrics,
    *,
    model: ModelConnection | None = None,
    complete: Complete | None = None,
    threshold: Threshold = Threshold(),
    retest_n: int = SAFETY_RETEST_N,
) -> GateReport:
    """Turn a scored run into a single pass/fail verdict.

    Applies the #33 floors to ``aggregate``. If a **safety** floor fails, each
    offending case is re-reviewed up to ``retest_n`` times: a violation that
    recurs even once is a confirmed defect (the floor stays failed); a case whose
    every confirmation run is clean is a flake (it clears). A safety floor is
    lifted to passing only when *all* its offending cases clear — this is
    confirm-before-failing, not retry-until-pass. Quality/structural floors are
    never retested.
    """
    floors = evaluate_floors(aggregate, threshold)
    failed_safety = [f for f in floors if f.tier == "safety" and not f.raw_passed]
    if not failed_safety:
        return GateReport(tuple(floors), RetestReport(performed=False, n=retest_n, cases={}))

    cases_by_id = {case.case_id: case for case in cases}
    retest_cases: dict[str, dict[str, str]] = {}
    cleared_metrics: set[str] = set()
    for floor in failed_safety:
        metric = floor.metric
        offenders = [r.case_id for r in results if _violates(r, metric)]
        verdicts: dict[str, str] = {}
        for case_id in offenders:
            reproduced = _confirm_violation(
                cases_by_id[case_id], metric, model=model, complete=complete, n=retest_n
            )
            verdicts[case_id] = "reproduced" if reproduced else "cleared"
        retest_cases[metric] = verdicts
        # The floor clears only if every offender was a flake.
        if offenders and all(v == "cleared" for v in verdicts.values()):
            cleared_metrics.add(metric)

    final_floors = tuple(
        floor.cleared_by_retest() if floor.metric in cleared_metrics else floor
        for floor in floors
    )
    return GateReport(final_floors, RetestReport(performed=True, n=retest_n, cases=retest_cases))


def _confirm_violation(
    case: Case,
    metric: str,
    *,
    model: ModelConnection | None = None,
    complete: Complete | None = None,
    n: int = SAFETY_RETEST_N,
) -> bool:
    """Re-review ``case`` up to ``n`` times; return ``True`` if ``metric`` is
    violated in any confirmation run (reproduced), ``False`` if all ``n`` are
    clean (a flake)."""
    for _ in range(n):
        review = _review_case(case, model=model, complete=complete)
        if _violates(evaluate_review(review, case), metric):
            return True
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aggregate_to_dict(aggregate: AggregateMetrics) -> dict:
    data = asdict(aggregate)
    data["cases_passing_rate"] = aggregate.cases_passing_rate
    return data


def _case_result_to_dict(result: CaseResult, review: WorkoutReview) -> dict:
    data = asdict(result)
    data["metrics"] = {
        "required_detail_recall": result.required_detail_recall,
        "evidence_validity_rate": result.evidence_validity_rate,
        "unsupported_claim_rate": result.unsupported_claim_rate,
    }
    data["checks"] = result.checks
    data["passed"] = result.passed
    data["review"] = review.model_dump(mode="json")
    return data


def build_artifact(
    *,
    results: list[CaseResult],
    aggregate: AggregateMetrics,
    reviews: dict[str, WorkoutReview],
    model_record: dict[str, str],
    run_id: str,
    gate: GateReport | None = None,
    benchmark_dir: Path = BENCHMARK_DIR,
    capability_version: str = CAPABILITY_VERSION,
    generated_at: str | None = None,
) -> dict:
    """Assemble the JSON-serializable metrics artifact for one benchmark run.

    When a ``gate`` is supplied its verdict (#33 pass/fail, per-floor breakdown,
    and the confirm-before-failing retest record) is stamped under ``gate``.
    """
    artifact = {
        "run_id": run_id,
        "generated_at": generated_at or _now_iso(),
        "capability_version": capability_version,
        "model": model_record,
        "corpus": {"hash": corpus_hash(benchmark_dir), "case_count": len(results)},
        "aggregate": _aggregate_to_dict(aggregate),
        "cases": [_case_result_to_dict(result, reviews[result.case_id]) for result in results],
    }
    if gate is not None:
        artifact["gate"] = gate.to_dict()
    return artifact


def write_artifact(artifact: dict, *, run_dir: Path = ARTIFACT_DIR) -> Path:
    """Write the artifact to ``<run_dir>/<run_id>.json`` and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{artifact['run_id']}.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return path


def trimmed_baseline(artifact: dict) -> dict:
    """A committable baseline: the full artifact minus each case's generated
    review. The per-case deterministic results (checks, metrics, missing/forbidden
    detail) and the stamps stay, so it's a stable reference; the reviews — which
    vary run to run — are dropped so a committed baseline isn't noise."""
    trimmed = {key: value for key, value in artifact.items() if key != "cases"}
    trimmed["cases"] = [
        {key: value for key, value in case.items() if key != "review"}
        for case in artifact["cases"]
    ]
    return trimmed


def _model_slug(name: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in name).strip("-")


def write_baseline(artifact: dict, *, baselines_dir: Path = BASELINES_DIR) -> Path:
    """Write the trimmed baseline to a tracked, self-identifying path — one file
    per (capability version, model, corpus hash), overwritten on re-record."""
    baselines_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"baseline-{artifact['capability_version']}"
        f"-{_model_slug(artifact['model']['name'])}"
        f"-{artifact['corpus']['hash'][:8]}.json"
    )
    path = baselines_dir / name
    path.write_text(json.dumps(trimmed_baseline(artifact), indent=2) + "\n")
    return path
