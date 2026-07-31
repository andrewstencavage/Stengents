"""A minimal loopback HTTP wrapper around ``review_workout``.

Kiln's Train console wants a Coaching reflection right after a Session
finishes. ``review_workout`` is a plain Python function with no HTTP surface,
so this module exposes exactly one route — ``GET /review/<workout_id>`` —
returning a :class:`~stengents.workout_review.contract.WorkoutReview` as JSON.

This one route is also Kiln's whole post-Session hook (see
``docs/kiln/CLAUDE.md``'s "Coaching reflection" section, mirrored on the
``kiln-issue-170`` worktree): Kiln's browser server proxies
``GET /api/coach/review/:id`` here right after a Session is saved. Issue #66
(Auto-replan) piggybacks on that same one call rather than adding any new
trigger — see ``auto_replan``, kicked off on its own background thread
alongside (never instead of, and never able to delay) ``review_workout``
below.

Stdlib-only (``http.server``), matching Kiln's own browser server, which uses
only ``node:http``: this is a spike, not a service that should need its own
dependency footprint. The model connection is resolved once at startup (see
``cli.py``'s ``serve-coach`` command) and passed to every request rather than
re-resolved per call.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .contract import WorkoutReview
from .review import review_workout as _default_review_workout

ReviewWorkout = Callable[..., WorkoutReview]
# Auto-replan's whole entry point, addressed by workout_id — anything it
# returns is ignored here; only whether it raises matters (see `_best_effort`).
AutoReplan = Callable[[str], object]


def _noop_auto_replan(workout_id: str) -> None:
    """The default `auto_replan`: nothing to run, e.g. for a caller (or a
    test) that only cares about `review_workout`."""


def _best_effort(auto_replan: AutoReplan, workout_id: str) -> None:
    """Run `auto_replan` for one Session, never letting a failure propagate —
    same fault-tolerance policy `do_GET` already applies to a `review_workout`
    failure (a logged, non-crashing miss), applied here too since Auto-replan
    has no HTTP response of its own to degrade into: a failed or errored
    Auto-replan run must never block Session save or Workout Review
    generation (issue #66's acceptance criterion), mirroring how a failed
    Coaching reflection already can't block Kiln from saving or exiting a
    Session (see module docstring).
    """
    try:
        auto_replan(workout_id)
    except Exception as error:  # noqa: BLE001 - best-effort: log, never raise.
        print(f"auto-replan failed for {workout_id!r}: {type(error).__name__}: {error}", file=sys.stderr)


def _run_in_background(auto_replan: AutoReplan, workout_id: str) -> None:
    """Kick off `_best_effort` on its own daemon thread rather than inline in
    the request-handling thread.

    A "real" Auto-replan run is several sequential Kiln MCP round trips
    (`get_planning_context`, `list_plan_templates`, a `create_plan_template`
    per template update, `create_plan_draft`, `select_current_plan`), each
    presently its own subprocess spawn (`kiln_mcp_client`'s one-connection-
    per-call design) — multiple seconds, easily. Running it synchronously
    would make "does not block ... Workout Review" only true in the narrow
    sense of "doesn't prevent it from eventually happening": the review
    response — and so Kiln's PATCH of it onto the Session — would still sit
    behind however long Auto-replan takes. A daemon thread (never joined; the
    server's own shutdown doesn't wait on it either) makes that literally
    true instead: `do_GET` moves on to `review_workout` immediately, and this
    thread's own result is never awaited by anything.
    """
    threading.Thread(target=_best_effort, args=(auto_replan, workout_id), daemon=True).start()


def make_handler(
    review_workout: ReviewWorkout, *, auto_replan: AutoReplan = _noop_auto_replan
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one ``review_workout`` callable (and,
    optionally, one ``auto_replan`` callable run alongside it).

    Both are injectable so tests can bind fakes and never touch a model or
    Kiln.
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
            # Alongside, not instead of, Workout Review — on its own
            # background thread so it can't add latency to this response
            # either, not just avoid blocking it on failure (issue #66).
            _run_in_background(auto_replan, workout_id)
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
    auto_replan: AutoReplan = _noop_auto_replan,
) -> ThreadingHTTPServer:
    """Build and start (but not block on) the review server.

    Returns the live server; call ``.serve_forever()`` to block, or
    ``.shutdown()`` from another thread to stop it. ``port=0`` binds an
    ephemeral port, read back from the returned server's ``server_address``.
    """
    server = ThreadingHTTPServer((host, port), make_handler(review_workout, auto_replan=auto_replan))
    return server
