from stengents.utilities.observation import reduce_value


def test_reduce_value_keeps_scalars_and_collapses_collections() -> None:
    reduced = reduce_value({"workout_count": 16, "workouts": [1, 2, 3], "meta": {"a": 1, "b": 2}, "note": "x" * 250})

    assert reduced["workout_count"] == 16
    assert reduced["workouts"] == "[3 items]"
    assert reduced["meta"] == "{2 keys}"
    assert reduced["note"].endswith("…(+50)") and len(reduced["note"]) == 206
