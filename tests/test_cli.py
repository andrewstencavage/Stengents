"""Coach server wiring (#65): `serve-coach` must call `review_workout` with
`kiln_mcp_client`'s MCP-backed fetch seams, not `kiln_client`'s HTTP ones.

`_serve_coach_command` itself starts a real (blocking) HTTP server, so this
fakes `resolve_model`, `serve`, and `review_workout` to capture exactly what
the command wires together, without ever binding a socket or reaching a
model endpoint or Kiln.
"""

from __future__ import annotations

from stengents import cli
from stengents.auto_replan.contract import ReplanDecision
from stengents.utilities.model_source import ModelConnection
from stengents.workout_review import kiln_mcp_client


class _FakeServer:
    """A `serve()` stand-in: reports back an address, then exits its
    `serve_forever` loop immediately, mirroring a Ctrl-C shutdown."""

    server_address = ("127.0.0.1", 8787)

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass


def test_serve_coach_wires_review_workout_to_kiln_mcp_client_reads(monkeypatch, capsys) -> None:
    connection = ModelConnection(name="qwen2.5:7b-8k", base_url="http://gym:11434", api_key="local")
    monkeypatch.setattr(ModelConnection, "preflight", lambda self, **_: None)
    monkeypatch.setattr(cli, "resolve_model", lambda *_args, **_kwargs: connection)

    captured_review_closures: list = []
    monkeypatch.setattr(
        "stengents.workout_review.server.serve",
        lambda *, host, port, review_workout, auto_replan=None: (
            captured_review_closures.append(review_workout),
            _FakeServer(),
        )[1],
    )

    captured_calls: list[dict] = []

    def fake_review_workout(workout_id, **kwargs):
        captured_calls.append({"workout_id": workout_id, **kwargs})
        return object()

    monkeypatch.setattr("stengents.workout_review.review.review_workout", fake_review_workout)

    exit_code = cli._serve_coach_command(None, None)
    capsys.readouterr()

    assert exit_code == 0
    assert len(captured_review_closures) == 1

    # Drive the closure `serve()` was handed, the same way an incoming
    # `GET /review/<id>` request would.
    captured_review_closures[0]("w1")

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["workout_id"] == "w1"
    assert call["fetch"] is kiln_mcp_client.fetch_workout
    assert call["fetch_history"] is kiln_mcp_client.fetch_sessions
    assert call["fetch_plans"] is kiln_mcp_client.fetch_plans
    assert call["model"] is connection


def test_serve_coach_wires_auto_replan_alongside_the_review(monkeypatch, capsys) -> None:
    """Issue #66: `serve-coach` must hand `serve()` an `auto_replan` closure
    that calls `run_auto_replan` for the same `workout_id` — the post-Session
    hook's second, independent best-effort track."""
    connection = ModelConnection(name="qwen2.5:7b-8k", base_url="http://gym:11434", api_key="local")
    monkeypatch.setattr(ModelConnection, "preflight", lambda self, **_: None)
    monkeypatch.setattr(cli, "resolve_model", lambda *_args, **_kwargs: connection)

    captured_auto_replan_closures: list = []
    monkeypatch.setattr(
        "stengents.workout_review.server.serve",
        lambda *, host, port, review_workout, auto_replan=None: (
            captured_auto_replan_closures.append(auto_replan),
            _FakeServer(),
        )[1],
    )
    monkeypatch.setattr("stengents.workout_review.review.review_workout", lambda workout_id, **kwargs: object())

    captured_calls: list[str] = []

    def fake_run_auto_replan(workout_id, **kwargs):
        captured_calls.append(workout_id)
        return ReplanDecision(selected_template=None, draft=None, activate=False, template_updates=[], reason="no strategy")

    monkeypatch.setattr("stengents.auto_replan.kiln_adapter.run_auto_replan", fake_run_auto_replan)

    exit_code = cli._serve_coach_command(None, None)
    capsys.readouterr()

    assert exit_code == 0
    assert len(captured_auto_replan_closures) == 1
    captured_auto_replan_closures[0]("w1")
    assert captured_calls == ["w1"]

    # The decision's reason is logged as Auto-replan's light-touch audit trail.
    logged = capsys.readouterr().out
    assert '"auto_replan": "w1"' in logged
    assert '"reason": "no strategy"' in logged
