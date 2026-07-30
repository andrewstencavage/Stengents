import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from stengents.workout_review import FactEvidence, Limitation, Observation, WorkoutReview
from stengents.workout_review.server import serve


@pytest.fixture
def running_server():
    calls: list[str] = []

    def fake_review_workout(workout_id: str) -> WorkoutReview:
        calls.append(workout_id)
        if workout_id == "boom":
            raise RuntimeError("model unavailable")
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
