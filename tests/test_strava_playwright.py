"""`upload_fit` driven against the real local `mock_strava` server with a real
Playwright browser — no mocking of Playwright itself, since the whole point is
to prove the automation actually drives a browser through a form. Skipped
(not failed) if Playwright's Chromium binary isn't installed in this
environment; `python -m playwright install chromium` was run once to provide
it here (see the package README for the exact command)."""

from __future__ import annotations

import threading

import pytest

from farm_system.strava_uploader import strava_playwright
from farm_system.strava_uploader.mock_strava import server as mock_strava_server

pytest.importorskip("playwright.sync_api")

try:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as _pw:
        _browser = _pw.chromium.launch(headless=True)
        _browser.close()
    _CHROMIUM_AVAILABLE = True
except Exception:
    _CHROMIUM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _CHROMIUM_AVAILABLE, reason="Playwright chromium binary is not installed")

_VALID_FIT = b"\x0e\x10\x43\x08" + b"FAKE-FIT-BYTES-FOR-TESTING-ONLY" * 4
_GARBAGE_FIT = b"\x00\x01"  # shorter than mock_strava's minimum plausible FIT size


@pytest.fixture
def running_mock_strava():
    server = mock_strava_server.serve()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_upload_fit_succeeds_against_the_mock_strava_server(running_mock_strava, tmp_path):
    result = strava_playwright.upload_fit(
        _VALID_FIT,
        "session-1.fit",
        base_url=running_mock_strava,
        activity_name="Lower B",
        screenshot_dir=tmp_path,
        timeout_ms=5000,
    )

    assert result.success is True
    assert result.error is None
    assert result.screenshot_path is None
    assert result.strava_url is not None
    assert "/activities/" in result.strava_url


def test_upload_fit_captures_a_screenshot_on_a_forced_failure(running_mock_strava, tmp_path):
    """A too-small/garbage FIT upload is the mock server's deliberate failure
    path (see mock_strava/server.py): the form re-renders with an error
    instead of redirecting, so `upload_fit`'s wait for the post-save URL times
    out — the failure + diagnostics-screenshot behavior issue #172 calls for."""
    result = strava_playwright.upload_fit(
        _GARBAGE_FIT,
        "session-2.fit",
        base_url=running_mock_strava,
        screenshot_dir=tmp_path,
        timeout_ms=2000,
    )

    assert result.success is False
    assert result.strava_url is None
    assert result.error
    assert result.screenshot_path is not None

    from pathlib import Path

    assert Path(result.screenshot_path).exists()


def test_upload_fit_dry_run_never_touches_the_network(tmp_path):
    """`dry_run=True` must not even attempt to reach `base_url` — pointed at
    an address nothing is listening on to prove it."""
    result = strava_playwright.upload_fit(
        _VALID_FIT,
        "session-3.fit",
        base_url="http://127.0.0.1:1",  # nothing listens here
        dry_run=True,
        screenshot_dir=tmp_path,
    )

    assert result == strava_playwright.UploadResult(success=True, strava_url=None, error=None, screenshot_path=None)


def test_upload_fit_against_an_unreachable_server_fails_without_raising(tmp_path):
    result = strava_playwright.upload_fit(
        _VALID_FIT,
        "session-4.fit",
        base_url="http://127.0.0.1:1",  # nothing listens here
        screenshot_dir=tmp_path,
        timeout_ms=2000,
    )

    assert result.success is False
    assert result.error is not None
