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


# --- strava-sync (#172): plain Python + Playwright, no model connection at all --


def test_strava_sync_once_prints_one_summary_and_never_resolves_a_model(monkeypatch, capsys) -> None:
    """`strava-sync` must not preflight or even resolve a model — unlike every
    other command here, `resolve_model` must simply never be called."""
    resolve_calls: list = []
    monkeypatch.setattr(cli, "resolve_model", lambda *args, **kwargs: resolve_calls.append((args, kwargs)))

    captured_run_sync_calls: list[dict] = []

    def fake_run_sync(*, playwright_base_url, dry_run, max_attempts):
        captured_run_sync_calls.append({"playwright_base_url": playwright_base_url, "dry_run": dry_run, "max_attempts": max_attempts})
        return {"dry_run": dry_run, "pending_count": 0, "uploaded": 0, "failed": 0, "skipped_at_max_attempts": 0, "items": []}

    monkeypatch.setattr("farm_system.strava_uploader.sync.run_sync", fake_run_sync)

    exit_code = cli.main(["strava-sync", "--once", "--dry-run"])
    logged = capsys.readouterr().out

    assert exit_code == 0
    assert resolve_calls == []
    assert captured_run_sync_calls == [{"playwright_base_url": None, "dry_run": True, "max_attempts": 5}]
    assert '"pending_count": 0' in logged


def test_strava_sync_loops_until_interrupted_and_respects_interval_and_max_attempts(monkeypatch, capsys) -> None:
    calls: list[dict] = []

    def fake_run_sync(*, playwright_base_url, dry_run, max_attempts):
        calls.append({"playwright_base_url": playwright_base_url, "dry_run": dry_run, "max_attempts": max_attempts})
        return {"dry_run": dry_run, "pending_count": 0, "uploaded": 0, "failed": 0, "skipped_at_max_attempts": 0, "items": []}

    monkeypatch.setattr("farm_system.strava_uploader.sync.run_sync", fake_run_sync)

    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    exit_code = cli._strava_sync_command(once=False, interval=30, dry_run=False, max_attempts=3, sleep=fake_sleep)

    assert exit_code == 0
    assert sleeps == [30, 30]
    assert len(calls) == 2  # one pass runs immediately before each sleep
    assert all(call["max_attempts"] == 3 for call in calls)


def test_strava_sync_reads_strava_upload_base_url_env_and_survives_a_failed_pass(monkeypatch, capsys) -> None:
    """A failed pass (e.g. Kiln unreachable) must be reported, not raised —
    a periodic loop that dies on the first transient Kiln outage defeats the
    point of running periodically."""
    monkeypatch.setenv("STRAVA_UPLOAD_BASE_URL", "http://mock-strava.example")

    def failing_run_sync(*, playwright_base_url, dry_run, max_attempts):
        assert playwright_base_url == "http://mock-strava.example"
        raise RuntimeError("Kiln unreachable")

    monkeypatch.setattr("farm_system.strava_uploader.sync.run_sync", failing_run_sync)

    exit_code = cli._strava_sync_command(once=True, interval=None, dry_run=False, max_attempts=None)
    logged = capsys.readouterr().out

    assert exit_code == 1
    assert "Kiln unreachable" in logged


# --- strava-download: plain Python + Playwright, no model connection at all --


def test_strava_download_once_prints_one_summary_and_never_resolves_a_model(monkeypatch, capsys) -> None:
    """`strava-download` must not preflight or even resolve a model — same
    reasoning as `strava-sync`: `resolve_model` must simply never be called."""
    resolve_calls: list = []
    monkeypatch.setattr(cli, "resolve_model", lambda *args, **kwargs: resolve_calls.append((args, kwargs)))

    captured_run_sync_calls: list[dict] = []

    def fake_run_sync(*, playwright_base_url, dry_run, limit):
        captured_run_sync_calls.append({"playwright_base_url": playwright_base_url, "dry_run": dry_run, "limit": limit})
        return {"dry_run": dry_run, "scraped_count": 0, "recorded": 0, "failed": 0, "items": []}

    monkeypatch.setattr("farm_system.strava_downloader.sync.run_sync", fake_run_sync)

    exit_code = cli.main(["strava-download", "--once", "--dry-run"])
    logged = capsys.readouterr().out

    assert exit_code == 0
    assert resolve_calls == []
    assert captured_run_sync_calls == [{"playwright_base_url": None, "dry_run": True, "limit": 10}]
    assert '"scraped_count": 0' in logged


def test_strava_download_loops_until_interrupted_and_respects_interval_and_limit(monkeypatch, capsys) -> None:
    calls: list[dict] = []

    def fake_run_sync(*, playwright_base_url, dry_run, limit):
        calls.append({"playwright_base_url": playwright_base_url, "dry_run": dry_run, "limit": limit})
        return {"dry_run": dry_run, "scraped_count": 0, "recorded": 0, "failed": 0, "items": []}

    monkeypatch.setattr("farm_system.strava_downloader.sync.run_sync", fake_run_sync)

    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    exit_code = cli._strava_download_command(once=False, interval=30, dry_run=False, limit=3, sleep=fake_sleep)

    assert exit_code == 0
    assert sleeps == [30, 30]
    assert len(calls) == 2  # one pass runs immediately before each sleep
    assert all(call["limit"] == 3 for call in calls)


def test_strava_download_reads_strava_download_base_url_env_and_survives_a_failed_pass(monkeypatch, capsys) -> None:
    """A failed pass (e.g. Kiln or Strava unreachable) must be reported, not
    raised — a periodic loop that dies on the first transient outage defeats
    the point of running periodically."""
    monkeypatch.setenv("STRAVA_DOWNLOAD_BASE_URL", "http://mock-strava.example")

    def failing_run_sync(*, playwright_base_url, dry_run, limit):
        assert playwright_base_url == "http://mock-strava.example"
        raise RuntimeError("Kiln unreachable")

    monkeypatch.setattr("farm_system.strava_downloader.sync.run_sync", failing_run_sync)

    exit_code = cli._strava_download_command(once=True, interval=None, dry_run=False, limit=None)
    logged = capsys.readouterr().out

    assert exit_code == 1
    assert "Kiln unreachable" in logged


def test_strava_download_reports_a_scrape_error_as_a_failed_pass(monkeypatch, capsys) -> None:
    """Unlike an exception, a scrape failure comes back as a normal summary
    dict with `scrape_error` set — `--once` must still exit 1 for it, the
    same "something needs a look" signal a raised exception gives."""

    def fake_run_sync(*, playwright_base_url, dry_run, limit):
        return {"dry_run": dry_run, "scraped_count": 0, "recorded": 0, "failed": 0, "items": [], "scrape_error": "TimeoutError: no entries"}

    monkeypatch.setattr("farm_system.strava_downloader.sync.run_sync", fake_run_sync)

    exit_code = cli._strava_download_command(once=True, interval=None, dry_run=False, limit=None)
    logged = capsys.readouterr().out

    assert exit_code == 1
    assert "TimeoutError" in logged
