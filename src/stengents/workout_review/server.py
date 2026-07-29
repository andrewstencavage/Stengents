"""A minimal loopback HTTP wrapper around ``review_workout``.

Kiln's Train console wants a Coaching reflection right after a Session
finishes. ``review_workout`` is a plain Python function with no HTTP surface,
so this module exposes exactly one route — ``GET /review/<workout_id>`` —
returning a :class:`~stengents.workout_review.contract.WorkoutReview` as JSON.

Stdlib-only (``http.server``), matching Kiln's own browser server, which uses
only ``node:http``: this is a spike, not a service that should need its own
dependency footprint. The model connection is resolved once at startup (see
``cli.py``'s ``serve-coach`` command) and passed to every request rather than
re-resolved per call.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .contract import WorkoutReview
from .review import review_workout as _default_review_workout

ReviewWorkout = Callable[..., WorkoutReview]


def make_handler(review_workout: ReviewWorkout) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one ``review_workout`` callable.

    Injectable so tests can bind a fake reviewer and never touch a model.
    """

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            pass  # Quiet by default; the CLI prints its own startup/record lines.

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) != 2 or parts[0] != "review" or not parts[1]:
                self._reply(404, {"error": f"no such route: {self.path}"})
                return
            workout_id = parts[1]
            try:
                review = review_workout(workout_id)
            except Exception as error:  # noqa: BLE001 - surface any failure as a 502, never crash the server
                self._reply(502, {"error": f"{type(error).__name__}: {error}"})
                return
            self._reply(200, review.model_dump(mode="json"))

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ReviewHandler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    review_workout: ReviewWorkout = _default_review_workout,
) -> ThreadingHTTPServer:
    """Build and start (but not block on) the review server.

    Returns the live server; call ``.serve_forever()`` to block, or
    ``.shutdown()`` from another thread to stop it. ``port=0`` binds an
    ephemeral port, read back from the returned server's ``server_address``.
    """
    server = ThreadingHTTPServer((host, port), make_handler(review_workout))
    return server
