"""A local stand-in for strava.com's training-log page, for testing
``strava_playwright.list_recent_activities`` end to end without real Strava
credentials.

Stdlib-only (``http.server``), mirroring
``strava_uploader.mock_strava.server``'s own ``make_handler``/``serve`` shape.
Route/DOM shape mirrors ``strava_playwright.py``'s **confirmed** (2026-08-03,
against real production strava.com — see that module's docstring) scrape
target: a single ``GET /athlete/training`` page rendering each recent
Activity as a plain server-rendered ``<table class="... activities ...">``
row (``tr.training-activity-row``) with ``.col-type``/``.col-date``/
``.col-title`` (containing the ``/activities/<id>`` link)/``.col-time``/
``.col-dist``/``.col-elev`` cells — kept in sync with that module by hand,
the same way the uploader's mock keeps ``/upload/select`` and
``/activities/<id>`` in sync with ``strava_uploader.strava_playwright`` by
hand rather than a shared import (this is meant to stand in for Strava's own
page, not reuse Kiln-side code). Distance/elevation render as pre-formatted
display text (``"3.53 mi"``, ``"101 ft"``) with a nested ``<abbr>`` unit tag,
exactly as confirmed on the real page — there is deliberately no
heart-rate column, since the real page doesn't have one either.
"""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TRAINING_LOG_PATH = "/athlete/training"

# A confirmed-shape default fixture. The first entry is the same real
# Activity (id, name, duration) `strava_uploader`'s own 2026-08-01
# confirmation produced by uploading a real finished Kiln Session — see that
# package's README. The other two exercise the mm:ss vs. h:mm:ss duration
# formats both confirmed present on the real page.
_DEFAULT_ACTIVITIES: list[dict] = [
    {
        "id": "19558762653",
        "type": "Workout",
        "name": "Morning Weight Training",
        "date_display": "Sat, 8/1/2026",
        "time_display": "46:00",
        "distance_display": "0 mi",
        "elevation_display": "0 ft",
    },
    {
        "id": "19540878515",
        "type": "Run",
        "name": "Morning Run",
        "date_display": "Fri, 7/31/2026",
        "time_display": "32:37",
        "distance_display": "3.53 mi",
        "elevation_display": "101 ft",
    },
    {
        "id": "19301112233",
        "type": "Run",
        "name": "Long Sunday Run",
        "date_display": "Sun, 7/12/2026",
        "time_display": "1:03:42",
        "distance_display": "6.32 mi",
        "elevation_display": "254 ft",
    },
]

_PAGE = """<!doctype html>
<html><head><title>My Activities | Mock Strava</title></head>
<body>
<h1>My Activities</h1>
<table class="table table-striped table-padded activities table-sortable">
  <thead><tr>
    <th class="col-type">Sport</th><th class="col-date">Date</th><th class="col-title">Title</th>
    <th class="col-time">Time</th><th class="col-dist">Distance</th><th class="col-elev">Elevation</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
</body></html>"""

_ROW = """    <tr class="training-activity-row">
      <td class="view-col col-type">{type}</td>
      <td class="view-col col-date">{date_display}</td>
      <td class="view-col col-title"><a href="https://www.strava.com/activities/{id}">{name}</a></td>
      <td class="view-col col-time">{time_display}</td>
      <td class="view-col col-dist">{distance_value} <abbr class="unit" title="{distance_unit}">{distance_unit_short}</abbr></td>
      <td class="view-col col-elev">{elevation_value} <abbr class="unit" title="{elevation_unit}">{elevation_unit_short}</abbr></td>
    </tr>"""

_UNIT_TITLES = {"mi": "miles", "km": "kilometers", "ft": "feet", "m": "meters"}


def _split_measurement(display: str) -> tuple[str, str, str]:
    value, _, unit_short = display.strip().partition(" ")
    return value, _UNIT_TITLES.get(unit_short, unit_short), unit_short


def _render_row(activity: dict) -> str:
    distance_value, distance_unit, distance_unit_short = _split_measurement(activity["distance_display"])
    elevation_value, elevation_unit, elevation_unit_short = _split_measurement(activity["elevation_display"])
    return _ROW.format(
        id=html.escape(str(activity["id"])),
        type=html.escape(activity["type"]),
        name=html.escape(activity["name"]),
        date_display=html.escape(activity["date_display"]),
        time_display=html.escape(activity["time_display"]),
        distance_value=html.escape(distance_value),
        distance_unit=html.escape(distance_unit),
        distance_unit_short=html.escape(distance_unit_short),
        elevation_value=html.escape(elevation_value),
        elevation_unit=html.escape(elevation_unit),
        elevation_unit_short=html.escape(elevation_unit_short),
    )


def make_handler(activities: list[dict] | None = None) -> type[BaseHTTPRequestHandler]:
    """Build a request handler serving a fixed list of Activities (the
    default confirmed-shape fixture, or a caller-supplied one) — no mutable
    state, since the training log here is read-only, unlike the uploader
    mock's `/activities/<id>` confirmation pages which record what was
    actually uploaded."""
    data = _DEFAULT_ACTIVITIES if activities is None else activities

    class MockStravaTrainingLogHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            pass  # Quiet by default; tests read responses, not server logs.

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            path = self.path.split("?", 1)[0]
            if path == TRAINING_LOG_PATH:
                rows = "\n".join(_render_row(activity) for activity in data)
                self._send_html(200, _PAGE.format(rows=rows))
                return
            self._send_html(404, "<h1>Not found</h1>")

        def _send_html(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return MockStravaTrainingLogHandler


def serve(*, host: str = "127.0.0.1", port: int = 0, activities: list[dict] | None = None) -> ThreadingHTTPServer:
    """Build and start (but not block on) the mock Strava training-log
    server. ``port=0`` (the default) binds an ephemeral port, read back from
    the returned server's ``server_address`` — exactly
    ``strava_uploader.mock_strava.server.serve``'s own shape."""
    return ThreadingHTTPServer((host, port), make_handler(activities))
