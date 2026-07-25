from farm_system.kiln_coach import root_agent
from farm_system.kiln_coach.kiln_client import compact_sessions, finished_sessions


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
