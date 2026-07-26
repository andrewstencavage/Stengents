import json

import pytest

from stengents import cli
from stengents.workout_review import CAPABILITY_VERSION
from stengents.workout_review.benchmark_runner import (
    build_artifact,
    corpus_hash,
    run_benchmark,
    write_artifact,
)
from stengents.workout_review.evaluator import load_corpus


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
