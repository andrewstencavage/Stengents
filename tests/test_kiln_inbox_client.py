"""HTTP client tests for `kiln_inbox_client` against a real loopback stub
server, mirroring `test_kiln_outbox_client.py`'s "spin up a real server on an
ephemeral port" pattern rather than mocking `urlopen`."""

from __future__ import annotations

import json
import threading

import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer

from farm_system.strava_downloader import kiln_inbox_client
from farm_system.strava_downloader.kiln_inbox_client import StravaActivityNotFound


class _StubInboxHandler(BaseHTTPRequestHandler):
    """A tiny stand-in for Kiln's `/api/strava-inbox*` routes (see
    `local/server.js` on the kiln side): records every request it receives so
    tests can assert on method/path/body, and serves fixed canned responses."""

    activities: list[dict] = []
    requests: list[dict] = []
    known_ids: set[str] = set()

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        self.requests.append({"method": "GET", "path": self.path})
        if self.path.startswith("/api/strava-inbox"):
            self._reply_json(200, self.activities)
            return
        self._reply_json(404, {"error": "no such route"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append({"method": "POST", "path": self.path, "body": body})
        if self.path == "/api/strava-inbox":
            self._reply_json(201, {**body, "sessionId": None})
            return
        if self.path.startswith("/api/strava-inbox/") and self.path.endswith("/attach"):
            activity_id = self.path.split("/")[3]
            if activity_id not in self.known_ids:
                self._reply_json(400, {"error": "Strava activity not found"})
                return
            self._reply_json(200, {"id": activity_id, "sessionId": body.get("sessionId")})
            return
        self._reply_json(404, {"error": "no such route"})

    def _reply_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_server():
    _StubInboxHandler.activities = []
    _StubInboxHandler.requests = []
    _StubInboxHandler.known_ids = set()
    server = HTTPServer(("127.0.0.1", 0), _StubInboxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", _StubInboxHandler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_list_activities_returns_the_stubbed_entries(stub_server):
    base, handler = stub_server
    handler.activities = [{"id": "strava-1", "sessionId": None, "type": "Run"}]

    entries = kiln_inbox_client.list_activities(base=base)

    assert entries == handler.activities
    assert handler.requests == [{"method": "GET", "path": "/api/strava-inbox"}]


def test_list_activities_passes_status_through_as_a_query_param(stub_server):
    base, handler = stub_server

    kiln_inbox_client.list_activities(status="unattached", base=base)

    assert handler.requests == [{"method": "GET", "path": "/api/strava-inbox?status=unattached"}]


def test_record_activity_posts_the_activity_and_returns_the_stored_entry(stub_server):
    base, handler = stub_server
    activity = {"id": "strava-1", "type": "Run", "name": "Morning Run", "date": "2026-08-02T12:00:00Z", "movingSeconds": 1800}

    result = kiln_inbox_client.record_activity(activity, base=base)

    assert result == {**activity, "sessionId": None}
    assert handler.requests == [{"method": "POST", "path": "/api/strava-inbox", "body": activity}]


def test_attach_activity_posts_the_session_id_and_returns_the_entry(stub_server):
    base, handler = stub_server
    handler.known_ids = {"strava-1"}

    result = kiln_inbox_client.attach_activity("strava-1", "session-1", base=base)

    assert result == {"id": "strava-1", "sessionId": "session-1"}
    assert handler.requests == [{"method": "POST", "path": "/api/strava-inbox/strava-1/attach", "body": {"sessionId": "session-1"}}]


def test_attach_activity_raises_strava_activity_not_found_when_kiln_rejects_it(stub_server):
    base, _ = stub_server

    with pytest.raises(StravaActivityNotFound):
        kiln_inbox_client.attach_activity("missing", "session-1", base=base)


def test_base_url_defaults_and_reads_kiln_base_url_env(monkeypatch):
    monkeypatch.delenv("KILN_BASE_URL", raising=False)
    assert kiln_inbox_client.base_url() == kiln_inbox_client.DEFAULT_BASE_URL

    monkeypatch.setenv("KILN_BASE_URL", "http://example:4173/")
    assert kiln_inbox_client.base_url() == "http://example:4173"
