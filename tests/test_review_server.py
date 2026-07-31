import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from stengents.workout_review import FactEvidence, Limitation, Observation, WorkoutReview
from stengents.workout_review.server import serve


def _make_review(workout_id: str) -> WorkoutReview:
    return WorkoutReview(
        workout_id=workout_id,
        observations=[
            Observation(
                kind="fact",
                confidence="firm",
                category="performance",
                claim="Bench Press was performed for 3 sets.",
                evidence=[FactEvidence(workout_id=workout_id, exercise=None, field="minutes", value="33")],
            )
        ],
        limitations=[Limitation(kind="insufficient_history", detail="No prior Bench Press sets.")],
    )


@pytest.fixture
def running_server():
    calls: list[str] = []

    def fake_review_workout(workout_id: str) -> WorkoutReview:
        calls.append(workout_id)
        if workout_id == "boom":
            raise RuntimeError("model unavailable")
        return _make_review(workout_id)

    server = serve(port=0, review_workout=fake_review_workout)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", calls
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def server_with_auto_replan():
    """A running server whose `auto_replan` records calls and can be told to
    raise on demand — for exercising issue #66's "alongside, best-effort,
    never blocking" wiring."""
    review_calls: list[str] = []
    replan_calls: list[str] = []
    should_raise = {"boom-replan": True}

    def fake_review_workout(workout_id: str) -> WorkoutReview:
        review_calls.append(workout_id)
        return _make_review(workout_id)

    def fake_auto_replan(workout_id: str) -> None:
        replan_calls.append(workout_id)
        if should_raise.get(workout_id):
            raise RuntimeError("kiln unreachable")

    server = serve(port=0, review_workout=fake_review_workout, auto_replan=fake_auto_replan)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", review_calls, replan_calls
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_review_route_returns_the_workout_review_as_json(running_server) -> None:
    base, calls = running_server

    with urlopen(f"{base}/review/w1", timeout=5) as response:
        assert response.status == 200
        payload = json.load(response)

    assert calls == ["w1"]
    assert payload["workout_id"] == "w1"
    assert len(payload["observations"]) == 1
    assert payload["observations"][0]["claim"] == "Bench Press was performed for 3 sets."
    assert payload["limitations"][0]["kind"] == "insufficient_history"


def test_unknown_route_is_a_404(running_server) -> None:
    base, _ = running_server

    with pytest.raises(HTTPError) as excinfo:
        urlopen(f"{base}/nope", timeout=5)
    assert excinfo.value.code == 404


def test_a_review_failure_is_a_502_not_a_crash(running_server) -> None:
    base, _ = running_server

    with pytest.raises(HTTPError) as excinfo:
        urlopen(f"{base}/review/boom", timeout=5)
    assert excinfo.value.code == 502
    body = json.load(excinfo.value)
    assert "model unavailable" in body["error"]

    # The server survives a failed request and serves the next one fine.
    with urlopen(f"{base}/review/w2", timeout=5) as response:
        assert json.load(response)["workout_id"] == "w2"


# --- Auto-replan runs alongside the review, on its own thread (issue #66) --


def _wait_until(condition, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Poll `condition` until it's truthy or `timeout` elapses — `auto_replan`
    now runs on its own background thread (never awaited by the request
    thread, see `server.py`'s `_run_in_background`), so a test asserting on
    its side effects has to wait for it rather than assume it's already
    finished the instant `urlopen` returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    assert condition(), f"condition not met within {timeout}s"


def test_auto_replan_runs_alongside_the_review_for_the_same_workout_id(server_with_auto_replan) -> None:
    base, review_calls, replan_calls = server_with_auto_replan

    with urlopen(f"{base}/review/w1", timeout=5) as response:
        assert response.status == 200
        assert json.load(response)["workout_id"] == "w1"

    assert review_calls == ["w1"]
    _wait_until(lambda: replan_calls == ["w1"])


def test_a_slow_auto_replan_does_not_delay_the_review_response() -> None:
    """`auto_replan` runs on its own background thread — a slow run must not
    add latency to the review response, not just a failing one avoid blocking
    it (a stronger guarantee than `_best_effort`'s try/except alone would
    give if `auto_replan` still ran inline)."""
    replan_started = threading.Event()
    replan_finished = threading.Event()

    def slow_auto_replan(workout_id: str) -> None:
        replan_started.set()
        time.sleep(0.5)
        replan_finished.set()

    server = serve(port=0, review_workout=lambda workout_id: _make_review(workout_id), auto_replan=slow_auto_replan)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        started_at = time.monotonic()
        with urlopen(f"http://{host}:{port}/review/w1", timeout=5) as response:
            assert response.status == 200
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.5, f"the review response waited on the slow auto_replan run ({elapsed:.2f}s)"
        assert replan_started.is_set()
        assert not replan_finished.is_set()  # still running in the background
        _wait_until(lambda: replan_finished.is_set(), timeout=2.0)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_a_failed_auto_replan_does_not_block_the_review_response(server_with_auto_replan) -> None:
    base, review_calls, replan_calls = server_with_auto_replan

    with urlopen(f"{base}/review/boom-replan", timeout=5) as response:
        assert response.status == 200
        assert json.load(response)["workout_id"] == "boom-replan"

    assert review_calls == ["boom-replan"]
    _wait_until(lambda: replan_calls == ["boom-replan"])

    # The server survives it and serves the next request fine too.
    with urlopen(f"{base}/review/w2", timeout=5) as response:
        assert json.load(response)["workout_id"] == "w2"
    _wait_until(lambda: replan_calls == ["boom-replan", "w2"])


def test_default_auto_replan_is_a_noop_when_the_caller_does_not_supply_one(running_server) -> None:
    base, calls = running_server

    with urlopen(f"{base}/review/w1", timeout=5) as response:
        assert response.status == 200
    assert calls == ["w1"]
