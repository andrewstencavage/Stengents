from stengents.utilities.observation import build_call_record, reduce_value


def test_reduce_value_keeps_scalars_and_collapses_collections() -> None:
    reduced = reduce_value({"workout_count": 16, "workouts": [1, 2, 3], "meta": {"a": 1, "b": 2}, "note": "x" * 250})

    assert reduced["workout_count"] == 16
    assert reduced["workouts"] == "[3 items]"
    assert reduced["meta"] == "{2 keys}"
    assert reduced["note"].endswith("…(+50)") and len(reduced["note"]) == 206


def test_build_call_record_reduces_args_and_result_on_success() -> None:
    record = build_call_record("read_file", {"path": "a.py"}, 12, result={"content": "x" * 250})

    assert record == {
        "name": "read_file",
        "args": {"path": "a.py"},
        "duration_ms": 12,
        "outcome": "ok",
        "result_summary": {"content": ("x" * 200) + "…(+50)"},
    }


def test_build_call_record_names_the_error_on_failure() -> None:
    record = build_call_record("run_tests", {}, 5, error=ValueError("boom"))

    assert record == {
        "name": "run_tests",
        "args": {},
        "duration_ms": 5,
        "outcome": "error",
        "error": "ValueError: boom",
    }
