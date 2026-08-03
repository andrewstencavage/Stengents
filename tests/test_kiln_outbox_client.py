"""HTTP client tests for `kiln_outbox_client` against a real loopback stub
server, mirroring `test_review_server.py`'s "spin up a real server on an
ephemeral port" pattern rather than mocking `urlopen`."""

from __future__ import annotations

import json
import threading

import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer

from farm_system.strava_uploader import kiln_outbox_client
from farm_system.strava_uploader.kiln_outbox_client import OutboxEntryNotFound


class _StubOutboxHandler(BaseHTTPRequestHandler):
    """A tiny stand-in for Kiln's `/api/strava-outbox*` routes (see
    `local/server.js` on the kiln side): records every request it receives so
    tests can assert on method/path/body, and serves fixed canned responses."""

    pending_entries: list[dict] = []
    fit_bytes: dict[str, bytes] = {}
    requests: list[dict] = []

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        self.requests.append({"method": "GET", "path": self.path})
        if self.path == "/api/strava-outbox?status=pending":
            self._reply_json(200, self.pending_entries)
            return
        if self.path.startswith("/api/strava-outbox/") and self.path.endswith("/fit"):
            session_id = self.path.split("/")[3]
            body = self.fit_bytes.get(session_id)
            if body is None:
                self._reply_json(404, {"error": "Strava outbox entry not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{session_id}.fit"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._reply_json(404, {"error": "no such route"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append({"method": "POST", "path": self.path, "body": body})
        session_id = self.path.split("/")[3]
        if self.path.endswith("/uploaded"):
            self._reply_json(200, {"sessionId": session_id, "status": "uploaded", "stravaUrl": body.get("stravaUrl")})
            return
        if self.path.endswith("/failed"):
            self._reply_json(200, {"sessionId": session_id, "status": "pending", "attempts": 1, "lastError": body.get("error")})
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
    _StubOutboxHandler.pending_entries = []
    _StubOutboxHandler.fit_bytes = {}
    _StubOutboxHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StubOutboxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", _StubOutboxHandler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_list_pending_returns_the_stubbed_entries(stub_server):
    base, handler = stub_server
    handler.pending_entries = [{"sessionId": "s1", "status": "pending", "attempts": 0, "session": {"workoutName": "Lower B"}}]

    entries = kiln_outbox_client.list_pending(base=base)

    assert entries == handler.pending_entries
    assert handler.requests == [{"method": "GET", "path": "/api/strava-outbox?status=pending"}]


def test_fetch_fit_returns_the_raw_bytes(stub_server):
    base, handler = stub_server
    handler.fit_bytes = {"s1": b"\x0e\x10FIT-BYTES-HERE"}

    assert kiln_outbox_client.fetch_fit("s1", base=base) == b"\x0e\x10FIT-BYTES-HERE"


def test_fetch_fit_raises_outbox_entry_not_found_on_404(stub_server):
    base, _ = stub_server

    with pytest.raises(OutboxEntryNotFound):
        kiln_outbox_client.fetch_fit("missing", base=base)


def test_mark_uploaded_posts_the_strava_url_and_returns_the_entry(stub_server):
    base, handler = stub_server

    result = kiln_outbox_client.mark_uploaded("s1", strava_url="https://strava.example/activities/1", base=base)

    assert result == {"sessionId": "s1", "status": "uploaded", "stravaUrl": "https://strava.example/activities/1"}
    assert handler.requests == [
        {"method": "POST", "path": "/api/strava-outbox/s1/uploaded", "body": {"stravaUrl": "https://strava.example/activities/1"}}
    ]


def test_mark_uploaded_with_no_strava_url(stub_server):
    base, handler = stub_server

    kiln_outbox_client.mark_uploaded("s1", base=base)

    assert handler.requests[0]["body"] == {"stravaUrl": None}


def test_mark_failed_posts_the_error_and_returns_the_entry(stub_server):
    base, handler = stub_server

    result = kiln_outbox_client.mark_failed("s1", "login wall", base=base)

    assert result == {"sessionId": "s1", "status": "pending", "attempts": 1, "lastError": "login wall"}
    assert handler.requests == [{"method": "POST", "path": "/api/strava-outbox/s1/failed", "body": {"error": "login wall"}}]


def test_base_url_defaults_and_reads_kiln_base_url_env(monkeypatch):
    monkeypatch.delenv("KILN_BASE_URL", raising=False)
    assert kiln_outbox_client.base_url() == kiln_outbox_client.DEFAULT_BASE_URL

    monkeypatch.setenv("KILN_BASE_URL", "http://example:4173/")
    assert kiln_outbox_client.base_url() == "http://example:4173"
