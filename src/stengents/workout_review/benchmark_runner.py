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

# Where a benchmark run's artifact is written, mirroring `.stengents/runs/`.
ARTIFACT_DIR = Path(".stengents/benchmark")


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
        by_id = {session.get("id"): session for session in case.pool}
        review = review_workout(
            case.subject_workout_id,
            fetch=(lambda workout_id, _by_id=by_id: _by_id.get(workout_id)),
            fetch_history=(lambda _pool=case.pool: list(_pool)),
            model=model,
            complete=complete,
        )
        reviews[case.case_id] = review
        results.append(evaluate_review(review, case))
    return results, aggregate_results(results), reviews


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
    benchmark_dir: Path = BENCHMARK_DIR,
    capability_version: str = CAPABILITY_VERSION,
    generated_at: str | None = None,
) -> dict:
    """Assemble the JSON-serializable metrics artifact for one benchmark run."""
    return {
        "run_id": run_id,
        "generated_at": generated_at or _now_iso(),
        "capability_version": capability_version,
        "model": model_record,
        "corpus": {"hash": corpus_hash(benchmark_dir), "case_count": len(results)},
        "aggregate": _aggregate_to_dict(aggregate),
        "cases": [_case_result_to_dict(result, reviews[result.case_id]) for result in results],
    }


def write_artifact(artifact: dict, *, run_dir: Path = ARTIFACT_DIR) -> Path:
    """Write the artifact to ``<run_dir>/<run_id>.json`` and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{artifact['run_id']}.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return path
