import json

import pytest

from stengents import cli
from stengents.workout_review import CAPABILITY_VERSION
from stengents.workout_review import benchmark_runner
from stengents.workout_review.benchmark_runner import (
    build_artifact,
    corpus_hash,
    gate_benchmark,
    run_benchmark,
    trimmed_baseline,
    write_artifact,
    write_baseline,
)
from stengents.workout_review.contract import Observation, SetEvidence, WorkoutReview
from stengents.workout_review.evaluator import AggregateMetrics, Case, CaseResult, load_corpus


# A model call is never made: the fake returns text the generator can't parse, so
# review_workout degrades to a valid, claim-free review. The runner only needs
# valid reviews to score — richness of generation is #23's concern, not the runner's.
def _fake_complete(_model, _prompt):
    return "{}"


def test_run_benchmark_scores_every_case_offline() -> None:
    cases = load_corpus()
    assert len(cases) == 12

    results, aggregate, reviews = run_benchmark(cases, complete=_fake_complete)

    assert len(results) == 12
    assert aggregate.case_count == 12
    assert set(reviews) == {case.case_id for case in cases}
    # Every degraded review is still schema-valid.
    assert all(result.schema_valid for result in results)


def test_corpus_hash_is_deterministic_and_hex() -> None:
    first = corpus_hash()
    second = corpus_hash()
    assert first == second
    assert len(first) == 64
    int(first, 16)  # is hex


def test_corpus_hash_tracks_content(tmp_path) -> None:
    case = tmp_path / "c01"
    case.mkdir()
    (case / "input.json").write_text('[{"id": "a"}]')
    (case / "expectations.json").write_text('{"subject_workout_id": "a"}')
    before = corpus_hash(tmp_path)

    (case / "expectations.json").write_text('{"subject_workout_id": "a", "required_evidence": []}')
    after = corpus_hash(tmp_path)

    assert before != after
    # Reformatting without changing content does not change the hash.
    (case / "expectations.json").write_text('{\n  "required_evidence": [],\n  "subject_workout_id": "a"\n}')
    assert corpus_hash(tmp_path) == after


def test_build_artifact_is_stamped_and_json_serializable() -> None:
    cases = load_corpus()
    results, aggregate, reviews = run_benchmark(cases, complete=_fake_complete)

    artifact = build_artifact(
        results=results,
        aggregate=aggregate,
        reviews=reviews,
        model_record={"provider": "openai-compatible", "name": "qwen2.5:7b-8k"},
        run_id="run-123",
    )

    assert artifact["run_id"] == "run-123"
    assert artifact["capability_version"] == CAPABILITY_VERSION
    assert artifact["model"]["name"] == "qwen2.5:7b-8k"
    assert len(artifact["corpus"]["hash"]) == 64
    assert artifact["corpus"]["case_count"] == 12
    assert artifact["aggregate"]["case_count"] == 12
    assert len(artifact["cases"]) == 12
    assert "review" in artifact["cases"][0]
    assert "checks" in artifact["cases"][0]
    # The whole artifact round-trips through JSON.
    json.loads(json.dumps(artifact))


def test_write_artifact_lands_at_run_id_path(tmp_path) -> None:
    artifact = {"run_id": "abc", "aggregate": {}}
    path = write_artifact(artifact, run_dir=tmp_path / "benchmark")
    assert path.name == "abc.json"
    assert json.loads(path.read_text())["run_id"] == "abc"


@pytest.mark.parametrize("argv", [[], ["bogus"], ["run"], ["review-benchmark", "extra"], ["run", "a", "b"]])
def test_cli_rejects_malformed_invocations(argv, capsys) -> None:
    assert cli.main(argv) == 2
    assert "usage:" in capsys.readouterr().err


def _sample_artifact() -> dict:
    cases = load_corpus()
    results, aggregate, reviews = run_benchmark(cases, complete=_fake_complete)
    return build_artifact(
        results=results,
        aggregate=aggregate,
        reviews=reviews,
        model_record={"provider": "openai-compatible", "name": "qwen2.5:7b-8k"},
        run_id="run-123",
    )


def test_trimmed_baseline_drops_reviews_but_keeps_results() -> None:
    trimmed = trimmed_baseline(_sample_artifact())

    assert trimmed["capability_version"] == CAPABILITY_VERSION
    assert trimmed["corpus"]["hash"]
    assert all("review" not in case for case in trimmed["cases"])
    assert all("checks" in case for case in trimmed["cases"])


def test_write_baseline_lands_at_self_identifying_path(tmp_path) -> None:
    path = write_baseline(_sample_artifact(), baselines_dir=tmp_path)

    hash8 = corpus_hash()[:8]
    assert path.name == f"baseline-{CAPABILITY_VERSION}-qwen2-5-7b-8k-{hash8}.json"
    assert "review" not in json.loads(path.read_text())["cases"][0]


def test_parse_recognizes_write_baseline_flag() -> None:
    positional, model_override, write_baseline_flag, port_override = cli._parse(["review-benchmark", "--write-baseline"])
    assert positional == ["review-benchmark"]
    assert model_override is None
    assert write_baseline_flag is True
    assert port_override is None

    _, _, unflagged, _ = cli._parse(["review-benchmark"])
    assert unflagged is False


def test_parse_recognizes_serve_coach_port() -> None:
    positional, model_override, _, port_override = cli._parse(["serve-coach", "--port", "9001", "--model", "qwen2.5:7b-8k"])
    assert positional == ["serve-coach"]
    assert model_override == "qwen2.5:7b-8k"
    assert port_override == 9001

    _, _, _, unflagged_port = cli._parse(["serve-coach"])
    assert unflagged_port is None


def _install_offline_benchmark(monkeypatch, tmp_path):
    """Stub the endpoint-bound pieces of review-benchmark so the command runs
    offline, and return a list that records every write_baseline call."""
    from stengents.workout_review import benchmark_runner

    class _FakeConnection:
        name = "qwen2.5:7b-8k"
        base_url = "http://endpoint"

        def preflight(self):
            return None

        def as_record(self):
            return {"provider": "openai-compatible", "name": self.name}

    class _FakeGate:
        passed = True

    monkeypatch.setattr(cli, "resolve_model", lambda default_name, name=None: _FakeConnection())
    monkeypatch.setattr(benchmark_runner, "run_benchmark", lambda cases, model: ("results", "aggregate", "reviews"))
    monkeypatch.setattr(benchmark_runner, "gate_benchmark", lambda cases, results, aggregate, model: _FakeGate())
    monkeypatch.setattr(benchmark_runner, "build_artifact", lambda **kwargs: {"aggregate": {"case_count": 12}})
    monkeypatch.setattr(benchmark_runner, "write_artifact", lambda artifact: tmp_path / "run.json")

    baseline_calls: list[dict] = []

    def _spy_write_baseline(artifact):
        baseline_calls.append(artifact)
        return tmp_path / "baseline.json"

    monkeypatch.setattr(benchmark_runner, "write_baseline", _spy_write_baseline)
    return baseline_calls


def test_review_benchmark_writes_baseline_with_flag(tmp_path, monkeypatch, capsys) -> None:
    baseline_calls = _install_offline_benchmark(monkeypatch, tmp_path)

    assert cli.main(["review-benchmark", "--write-baseline"]) == 0

    assert len(baseline_calls) == 1
    assert "baseline.json" in capsys.readouterr().out


def test_review_benchmark_skips_baseline_by_default(tmp_path, monkeypatch, capsys) -> None:
    baseline_calls = _install_offline_benchmark(monkeypatch, tmp_path)

    assert cli.main(["review-benchmark"]) == 0

    assert baseline_calls == []
    assert "baseline.json" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The passing gate + confirm-before-failing retest (#33/#37)
# --------------------------------------------------------------------------- #
def _synthetic_case() -> Case:
    """A trivial one-Session case; the retest reviews are injected, so its only
    job is to exist for `evaluate_review` to score the injected reviews against."""
    return Case(
        case_id="cX",
        pool=({"id": "w1", "date": "2026-01-01"},),
        subject_workout_id="w1",
        required_evidence=(),
        required_limitations=(),
        forbidden_observations=(),
    )


def _result(case_id: str, *, evidence_valid: bool = True, unsupported: int = 0) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        subject_workout_id="w1",
        schema_valid=True,
        schema_error=None,
        cited_evidence_total=2,
        cited_evidence_valid=2 if evidence_valid else 1,
        invalid_evidence=(),
        required_detail_total=0,
        required_detail_matched=0,
        missing_required_evidence=(),
        required_limitations_total=0,
        required_limitations_matched=0,
        missing_limitations=(),
        forbidden_present=(),
        observation_total=1,
        unsupported_observations=unsupported,
    )


def _aggregate(*, evidence=1.0, unsupported=0.0, detail=0.66, limitation=0.333) -> AggregateMetrics:
    return AggregateMetrics(
        case_count=1,
        schema_valid_rate=1.0,
        required_detail_recall=detail,
        required_limitation_recall=limitation,
        evidence_validity_rate=evidence,
        unsupported_claim_rate=unsupported,
        cases_passing=1,
    )


def _clean_review() -> WorkoutReview:
    return WorkoutReview(workout_id="w1", summary="clean", observations=[], limitations=[])


def _ungrounded_review() -> WorkoutReview:
    """A review citing a set that isn't in the pool → unresolvable → a safety
    violation on both evidence-validity and unsupported-claim."""
    bogus = SetEvidence(workout_id="w1", exercise="Ghost", set_index=0, reps=5, load=10.0, loadType="plate")
    observation = Observation(kind="fact", confidence="firm", category="performance", claim="up", evidence=[bogus])
    return WorkoutReview(workout_id="w1", summary="bad", observations=[observation], limitations=[])


def test_gate_passes_with_no_retest_when_floors_met() -> None:
    results = [_result("cX", evidence_valid=True)]
    report = gate_benchmark([_synthetic_case()], results, _aggregate(), complete=lambda *_: "{}")

    assert report.passed
    assert report.retest.performed is False


def test_gate_fails_a_quality_floor_without_retesting() -> None:
    # Safety is clean; a quality floor is below bound. No safety failure → no retest.
    results = [_result("cX", evidence_valid=True)]
    report = gate_benchmark([_synthetic_case()], results, _aggregate(detail=0.5), complete=lambda *_: "{}")

    assert not report.passed
    assert report.retest.performed is False
    failed = {f.metric for f in report.floors if not f.passed}
    assert failed == {"required_detail_recall"}


def test_gate_fails_when_safety_violation_reproduces(monkeypatch) -> None:
    # The offending case re-generates the same ungrounded review every time.
    monkeypatch.setattr(benchmark_runner, "_review_case", lambda case, **_: _ungrounded_review())
    results = [_result("cX", evidence_valid=False, unsupported=1)]
    aggregate = _aggregate(evidence=0.5, unsupported=0.5)

    report = gate_benchmark([_synthetic_case()], results, aggregate, retest_n=3)

    assert report.retest.performed is True
    assert report.retest.cases["evidence_validity_rate"]["cX"] == "reproduced"
    assert not report.passed
    validity = next(f for f in report.floors if f.metric == "evidence_validity_rate")
    assert validity.raw_passed is False and validity.retest_cleared is False


def test_gate_clears_a_safety_flake_and_passes(monkeypatch) -> None:
    # Retest re-generations all come back clean → the initial miss was a flake.
    monkeypatch.setattr(benchmark_runner, "_review_case", lambda case, **_: _clean_review())
    results = [_result("cX", evidence_valid=False, unsupported=1)]
    aggregate = _aggregate(evidence=0.5, unsupported=0.5)

    report = gate_benchmark([_synthetic_case()], results, aggregate, retest_n=3)

    assert report.retest.performed is True
    assert report.retest.cases["evidence_validity_rate"]["cX"] == "cleared"
    validity = next(f for f in report.floors if f.metric == "evidence_validity_rate")
    assert validity.raw_passed is False and validity.retest_cleared is True
    assert report.passed


def test_confirm_before_failing_is_not_retry_until_pass(monkeypatch) -> None:
    # A reproduction anywhere in the N confirmation runs confirms the defect —
    # earlier clean runs do NOT rescue it (that would be retry-until-pass).
    reviews = iter([_clean_review(), _clean_review(), _ungrounded_review()])
    monkeypatch.setattr(benchmark_runner, "_review_case", lambda case, **_: next(reviews))
    # Only evidence-validity fails, so only that one metric is retested (3 runs).
    results = [_result("cX", evidence_valid=False, unsupported=0)]

    report = gate_benchmark([_synthetic_case()], results, _aggregate(evidence=0.5), retest_n=3)

    assert report.retest.cases["evidence_validity_rate"]["cX"] == "reproduced"
    assert not report.passed


def test_gate_verdict_lands_in_artifact() -> None:
    cases = load_corpus()
    results, aggregate, reviews = run_benchmark(cases, complete=_fake_complete)
    gate = gate_benchmark(cases, results, aggregate, complete=_fake_complete)

    artifact = build_artifact(
        results=results,
        aggregate=aggregate,
        reviews=reviews,
        model_record={"provider": "openai-compatible", "name": "qwen2.5:7b-8k"},
        run_id="run-gate",
        gate=gate,
    )

    assert "gate" in artifact
    assert artifact["gate"]["passed"] == gate.passed
    assert {f["metric"] for f in artifact["gate"]["floors"]} == {
        "evidence_validity_rate",
        "unsupported_claim_rate",
        "schema_valid_rate",
        "required_detail_recall",
        "required_limitation_recall",
    }
    # The new aggregate metric is stamped too.
    assert "required_limitation_recall" in artifact["aggregate"]
    json.loads(json.dumps(artifact))
