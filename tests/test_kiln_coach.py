import farm_system.kiln_coach.kiln_client as kiln_client
from farm_system.kiln_coach import root_agent
from farm_system.kiln_coach.kiln_client import compact_sessions, fetch_workout, finished_sessions, performed_sets


def test_kiln_coach_exports_a_standard_adk_root_agent() -> None:
    assert root_agent.name == "kiln_coach"
    assert "get_workouts" in {tool.__name__ for tool in root_agent.tools}


def test_finished_sessions_drops_abandoned_run_throughs() -> None:
    sessions = [
        {"id": "a", "status": "finished"},
        {"id": "b", "status": "abandoned"},
        {"id": "c", "status": "finished"},
        {"id": "d"},
    ]

    assert [session["id"] for session in finished_sessions(sessions)] == ["a", "c"]


def test_compact_sessions_trims_to_a_lean_readable_shape() -> None:
    session = {
        "id": "s1",
        "planId": "p1",
        "workoutName": "Lower B",
        "type": "strength",
        "date": "2026-07-23T10:50:14.623Z",
        "minutes": 33,
        "feel": "easy",
        "activities": [
            {
                "name": "Cable Squat",
                "note": "felt strong",
                "feel": "right",
                "plannedActivity": {"muscles": {"primary": ["quads"]}, "prescribedSets": [1, 2, 3]},
                "performedSets": [
                    {"reps": 10, "measurement": {"value": 3, "unit": "plate"}},
                    {"reps": 10, "measurement": {"value": 3, "unit": "plate"}},
                    {"reps": 8, "measurement": {"value": 3, "unit": "plate"}},
                ],
            }
        ],
    }

    (compact,) = compact_sessions([session])

    assert compact == {
        "date": "2026-07-23",
        "workout": "Lower B",
        "type": "strength",
        "minutes": 33,
        "feel": "easy",
        "activities": [
            {"exercise": "Cable Squat", "sets": "2×10@3plate, 8@3plate", "note": "felt strong", "feel": "right"}
        ],
    }


def test_performed_sets_projects_the_set_evidence_shape() -> None:
    activity = {
        "name": "Bench Press",
        "performedSets": [
            {"reps": 12, "measurement": {"value": 3, "unit": "plate"}},
            {"reps": 10, "measurement": {"value": 3, "unit": "plate"}},
        ],
    }

    assert performed_sets(activity) == [
        {"set_index": 0, "reps": 12, "load": 3, "loadType": "plate"},
        {"set_index": 1, "reps": 10, "load": 3, "loadType": "plate"},
    ]


def test_performed_sets_reads_a_timed_hold_as_reps_zero() -> None:
    plank = {"name": "Plank", "performedSets": [{"reps": 0, "measurement": {"value": 45, "unit": "sec"}}]}

    assert performed_sets(plank) == [{"set_index": 0, "reps": 0, "load": 45, "loadType": "sec"}]


def test_performed_sets_is_empty_for_a_freeform_activity() -> None:
    warmup = {"name": "Warmup: easy row", "spec": "5 min easy stroke", "sets_completed": 1}

    assert performed_sets(warmup) == []


def test_fetch_workout_returns_the_finished_session_with_that_id(monkeypatch) -> None:
    pool = [{"id": "w1", "status": "finished"}, {"id": "w2", "status": "finished"}]
    monkeypatch.setattr(kiln_client, "fetch_sessions", lambda **_: pool)

    assert fetch_workout("w2") == {"id": "w2", "status": "finished"}


def test_fetch_workout_returns_none_for_an_unknown_id(monkeypatch) -> None:
    monkeypatch.setattr(kiln_client, "fetch_sessions", lambda **_: [{"id": "w1", "status": "finished"}])

    assert fetch_workout("missing") is None
