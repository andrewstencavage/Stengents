"""A local stand-in for strava.com's upload flow, for testing
``strava_playwright.upload_fit`` end to end without real Strava credentials
(none are available in this environment — see the package README's
"What's validated" section).

Stdlib-only (``http.server``), mirroring ``workout_review.server``'s own
``make_handler``/``serve`` shape: a bound handler factory plus a thin
``serve()`` that returns a live, not-yet-blocking server so tests can run it
on an ephemeral port in a background thread.

Route shape mirrors what real Strava's upload flow is believed to look like
(best current knowledge, unverified — see the package README): a form page at
``GET /upload/select``, a multipart ``POST`` to the same path carrying the FIT
file plus an activity name, and — on success — a redirect to a confirmation
page at ``GET /activities/<id>`` that ``upload_fit`` detects by URL pattern.
A too-small/garbage upload (the deliberate failure path a test drives) is
rejected: the form re-renders with an error banner instead of redirecting, so
nothing at ``/activities/<id>`` ever appears and ``upload_fit`` times out
waiting for that navigation, exactly like a real failed save would.
"""

from __future__ import annotations

import html
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A FIT file's minimum plausible size; anything shorter is treated as the
# deliberate "garbage upload" failure path (see module docstring and the
# package README's note on exercising `upload_fit`'s failure/screenshot path).
_MIN_FIT_BYTES = 10

_UPLOAD_FORM = """<!doctype html>
<html><head><title>Upload an Activity | Mock Strava</title></head>
<body>
<h1>Upload Activity</h1>
{error_banner}
<form method="post" action="/upload/select" enctype="multipart/form-data">
  <input type="file" name="file" />
  <input type="text" name="name" placeholder="Activity name" />
  <button type="submit">Save</button>
</form>
</body></html>"""

_ERROR_BANNER = '<p data-upload-error="{message}">{message}</p>'

_CONFIRMATION = """<!doctype html>
<html><head><title>Your Activity | Mock Strava</title></head>
<body>
<div data-activity-url="/activities/{activity_id}" data-activity-id="{activity_id}">
  Uploaded {name} ({size} bytes)
</div>
</body></html>"""

_ACTIVITY_PATH = re.compile(r"^/activities/([\w-]+)$")


def _parse_multipart(content_type: str, body: bytes) -> dict[str, bytes]:
    """Hand-rolled ``multipart/form-data`` parsing — the stdlib ``cgi``
    module this used to lean on was removed in Python 3.13, and pulling in a
    new dependency just to parse a browser's own form submission would be
    disproportionate for a test-only mock server."""
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        return {}
    delimiter = ("--" + boundary_match.group(1)).encode()
    fields: dict[str, bytes] = {}
    for chunk in body.split(delimiter):
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if not chunk or chunk == b"--":
            continue
        header_bytes, sep, content = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        name_match = re.search(r'name="([^"]*)"', header_bytes.decode("latin-1"))
        if name_match:
            fields[name_match.group(1)] = content
    return fields


def make_handler() -> type[BaseHTTPRequestHandler]:
    """Build a request handler with its own in-memory activity store, so
    each ``serve()`` call (and so each test) gets an isolated mock Strava."""
    activities: dict[str, dict] = {}

    class MockStravaHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            pass  # Quiet by default; tests read responses, not server logs.

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            path = self.path.split("?", 1)[0]
            if path == "/upload/select":
                self._send_html(200, _UPLOAD_FORM.format(error_banner=""))
                return
            match = _ACTIVITY_PATH.match(path)
            if match:
                activity = activities.get(match.group(1))
                if activity is None:
                    self._send_html(404, "<h1>Activity not found</h1>")
                    return
                self._send_html(
                    200,
                    _CONFIRMATION.format(
                        activity_id=match.group(1),
                        name=html.escape(activity["name"] or "(untitled)"),
                        size=activity["size"],
                    ),
                )
                return
            self._send_html(404, "<h1>Not found</h1>")

        def do_POST(self) -> None:  # noqa: N802 - stdlib method name
            if self.path != "/upload/select":
                self._send_html(404, "<h1>Not found</h1>")
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            fields = _parse_multipart(self.headers.get("Content-Type", ""), body)
            file_bytes = fields.get("file", b"")
            name = fields.get("name", b"").decode("utf-8", errors="replace")

            if len(file_bytes) < _MIN_FIT_BYTES:
                banner = _ERROR_BANNER.format(message=html.escape("Invalid or empty FIT file."))
                self._send_html(200, _UPLOAD_FORM.format(error_banner=banner))
                return

            activity_id = uuid.uuid4().hex[:8]
            activities[activity_id] = {"name": name, "size": len(file_bytes)}
            self.send_response(303)
            self.send_header("Location", f"/activities/{activity_id}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_html(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return MockStravaHandler


def serve(*, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Build and start (but not block on) the mock Strava server.

    Returns the live server; call ``.serve_forever()`` to block, or
    ``.shutdown()`` from another thread to stop it. ``port=0`` (the default)
    binds an ephemeral port, read back from the returned server's
    ``server_address`` — exactly ``workout_review.server.serve``'s own shape.
    """
    return ThreadingHTTPServer((host, port), make_handler())
