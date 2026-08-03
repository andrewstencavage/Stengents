"""`list_recent_activities` driven against the real local `mock_strava`
training-log server with a real Playwright browser — no mocking of
Playwright itself, mirroring `test_strava_playwright.py`'s own approach for
the uploader. The mock's default fixture mirrors the DOM shape confirmed
against real production strava.com on 2026-08-03 (see
`strava_playwright.py`'s module docstring), including the same real
`Morning Weight Training` Activity `strava_uploader`'s own 2026-08-01
confirmation produced.

Skipped (not failed) if Playwright's Chromium binary isn't installed in this
environment."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from farm_system.strava_downloader import strava_playwright
from farm_system.strava_downloader.mock_strava import server as mock_strava_server

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


def _running_mock_strava(activities=None):
    server = mock_strava_server.serve(activities=activities)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


@pytest.fixture
def running_mock_strava():
    server, thread, base_url = _running_mock_strava()
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_list_recent_activities_scrapes_the_default_fixture(running_mock_strava, tmp_path):
    result = strava_playwright.list_recent_activities(base_url=running_mock_strava, screenshot_dir=tmp_path, timeout_ms=5000)

    assert result.success is True
    assert result.error is None
    assert result.screenshot_path is None
    assert len(result.activities) == 3

    first = result.activities[0]
    assert first["id"] == "19558762653"
    assert first["type"] == "Workout"
    assert first["name"] == "Morning Weight Training"
    assert first["date"] == "2026-08-01T00:00:00Z"
    assert first["movingSeconds"] == 46 * 60
    assert first["distanceMeters"] == 0.0
    assert first["elevationMeters"] == 0.0
    assert first["averageHeartRate"] is None


def test_list_recent_activities_parses_miles_feet_and_hmmss_duration(running_mock_strava, tmp_path):
    result = strava_playwright.list_recent_activities(base_url=running_mock_strava, screenshot_dir=tmp_path, timeout_ms=5000)

    run = next(activity for activity in result.activities if activity["id"] == "19540878515")
    assert run["movingSeconds"] == 32 * 60 + 37
    assert run["distanceMeters"] == round(3.53 * 1609.344, 2)
    assert run["elevationMeters"] == round(101 * 0.3048, 2)

    long_run = next(activity for activity in result.activities if activity["id"] == "19301112233")
    assert long_run["date"] == "2026-07-12T00:00:00Z"
    assert long_run["movingSeconds"] == 1 * 3600 + 3 * 60 + 42
    assert long_run["distanceMeters"] == round(6.32 * 1609.344, 2)


def test_list_recent_activities_never_reports_a_heart_rate(running_mock_strava, tmp_path):
    """Confirmed against the real page (2026-08-03): the training log has no
    heart-rate column at all, so every scraped Activity's averageHeartRate
    is None — not "missing sometimes," always absent from this source."""
    result = strava_playwright.list_recent_activities(base_url=running_mock_strava, screenshot_dir=tmp_path, timeout_ms=5000)

    assert all(activity["averageHeartRate"] is None for activity in result.activities)


def test_list_recent_activities_respects_limit(running_mock_strava, tmp_path):
    result = strava_playwright.list_recent_activities(base_url=running_mock_strava, limit=2, screenshot_dir=tmp_path, timeout_ms=5000)

    assert len(result.activities) == 2


def test_list_recent_activities_captures_a_screenshot_on_a_forced_failure(tmp_path):
    """An empty training log (the mock's deliberate failure fixture here,
    mirroring the uploader test's garbage-FIT failure path) means no
    `tr.training-activity-row` element ever appears, so the wait for it
    times out — the failure + diagnostics-screenshot behavior this mirrors
    from `strava_uploader.strava_playwright`."""
    server, thread, base_url = _running_mock_strava(activities=[])
    try:
        result = strava_playwright.list_recent_activities(base_url=base_url, screenshot_dir=tmp_path, timeout_ms=2000)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert result.success is False
    assert result.activities == []
    assert result.error
    assert result.screenshot_path is not None
    assert Path(result.screenshot_path).exists()


def test_list_recent_activities_dry_run_never_touches_the_network(tmp_path):
    """`dry_run=True` must not even attempt to reach `base_url` — pointed at
    an address nothing is listening on to prove it."""
    result = strava_playwright.list_recent_activities(
        base_url="http://127.0.0.1:1",  # nothing listens here
        dry_run=True,
        screenshot_dir=tmp_path,
    )

    assert result == strava_playwright.ScrapeResult(success=True, activities=[])


def test_list_recent_activities_against_an_unreachable_server_fails_without_raising(tmp_path):
    result = strava_playwright.list_recent_activities(
        base_url="http://127.0.0.1:1",  # nothing listens here
        screenshot_dir=tmp_path,
        timeout_ms=2000,
    )

    assert result.success is False
    assert result.error is not None
