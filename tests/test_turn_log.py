import json
from types import SimpleNamespace

from stengents.utilities.turn_log import TurnLogger, build_turn_record


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


def test_turn_logger_records_a_tool_error_with_its_message(tmp_path) -> None:
    logger = TurnLogger(model={"provider": "openai-compatible", "name": "qwen2.5:7b"}, log_path=tmp_path / "turns.jsonl")
    ctx = SimpleNamespace(invocation_id="inv-1", session=None, user_content=None)
    tool = SimpleNamespace(name="get_workouts")
    tool_ctx = SimpleNamespace(invocation_id="inv-1", function_call_id="call-1")

    logger.before_agent(callback_context=ctx)
    logger.before_tool(tool=tool, args={}, tool_context=tool_ctx)
    logger.on_tool_error(tool=tool, args={}, tool_context=tool_ctx, error=ConnectionError("kiln unreachable"))
    logger.after_agent(callback_context=ctx)

    record = json.loads((tmp_path / "turns.jsonl").read_text().splitlines()[0])
    assert record["tool_calls"][0] == {
        "name": "get_workouts", "args": {}, "outcome": "error",
        "duration_ms": record["tool_calls"][0]["duration_ms"], "error": "ConnectionError: kiln unreachable",
    }
    assert record["outcome"] == "errored" and record["error"] == "tool_error"
