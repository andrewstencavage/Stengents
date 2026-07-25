import json
from types import SimpleNamespace

from farm_system.kiln_coach import root_agent
from farm_system.kiln_coach.kiln_client import compact_sessions, finished_sessions
from farm_system.kiln_coach.turn_log import TurnLogger, build_turn_record, reduce_value


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


def test_reduce_value_keeps_scalars_and_collapses_collections() -> None:
    reduced = reduce_value({"workout_count": 16, "workouts": [1, 2, 3], "meta": {"a": 1, "b": 2}, "note": "x" * 250})

    assert reduced["workout_count"] == 16
    assert reduced["workouts"] == "[3 items]"
    assert reduced["meta"] == "{2 keys}"
    assert reduced["note"].endswith("…(+50)") and len(reduced["note"]) == 206


def test_build_turn_record_derives_operational_outcome() -> None:
    base = dict(
        turn_id="t1", session_id="s1", started_at="2026-07-25T09:41:43Z", duration_ms=10,
        model={"provider": "openai-compatible", "name": "qwen2.5:7b"}, query="how many workouts",
        tool_calls=[{"name": "get_workouts", "outcome": "ok"}],
    )

    errored_tool_calls = {**base, "tool_calls": [{"name": "get_workouts", "outcome": "error"}]}

    completed = build_turn_record(**base, answer="You've logged 16 workouts.", model_error=None)
    no_answer = build_turn_record(**base, answer=None, model_error=None)
    model_errored = build_turn_record(**{**base, "tool_calls": []}, answer=None, model_error="ConnectionError: boom")
    tool_errored = build_turn_record(**errored_tool_calls, answer=None, model_error=None)
    degraded = build_turn_record(**errored_tool_calls, answer="couldn't reach Kiln", model_error=None)

    assert completed["outcome"] == "completed" and "error" not in completed
    assert no_answer["outcome"] == "no_answer" and "error" not in no_answer
    assert model_errored["outcome"] == "errored" and model_errored["error"] == "ConnectionError: boom"
    assert tool_errored["outcome"] == "errored" and tool_errored["error"] == "tool_error"
    assert degraded["outcome"] == "degraded" and degraded["error"] == "tool_error"


def test_turn_logger_appends_one_jsonl_line_per_turn(tmp_path) -> None:
    logger = TurnLogger(model={"provider": "openai-compatible", "name": "qwen2.5:7b"}, log_path=tmp_path / "turns.jsonl")
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(id="sess-1"),
        user_content=SimpleNamespace(parts=[SimpleNamespace(text="how many workouts have I logged?")]),
    )
    tool = SimpleNamespace(name="get_workouts")
    tool_ctx = SimpleNamespace(invocation_id="inv-1", function_call_id="call-1")

    logger.before_agent(callback_context=ctx)
    logger.before_tool(tool=tool, args={}, tool_context=tool_ctx)
    logger.after_tool(tool=tool, args={}, tool_context=tool_ctx, tool_response={"workout_count": 16, "workouts": [0] * 16})
    logger.after_model(callback_context=ctx, llm_response=SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="You've logged 16 workouts.")])))
    logger.after_agent(callback_context=ctx)

    lines = (tmp_path / "turns.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query"] == "how many workouts have I logged?"
    assert record["session_id"] == "sess-1"
    assert record["outcome"] == "completed"
    assert record["answer"] == "You've logged 16 workouts."
    assert record["tool_calls"][0] == {
        "name": "get_workouts", "args": {}, "outcome": "ok",
        "duration_ms": record["tool_calls"][0]["duration_ms"], "result_summary": {"workout_count": 16, "workouts": "[16 items]"},
    }
