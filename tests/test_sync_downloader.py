"""`run_sync` end to end against fakes for both halves it orchestrates:
`strava_playwright.list_recent_activities` (a scripted stand-in — the real
browser-driving path is `test_strava_playwright_downloader.py`'s job, not
this module's) and `kiln_inbox_client.record_activity` (an in-memory Kiln
inbox double, following `test_sync.py`'s injected-fake pattern). Proves the
orchestration rules this mirrors from the uploader's own `run_sync`: a
successful scrape records every Activity, one bad item never aborts the
batch, a scrape failure is reported as `scrape_error` rather than raised, and
`dry_run` touches neither Strava nor Kiln."""

from __future__ import annotations

from farm_system.strava_downloader import sync
from farm_system.strava_downloader.strava_playwright import ScrapeResult


class _FakeInbox:
    """An in-memory stand-in for `kiln_inbox_client.record_activity` — same
    shape/signature, no HTTP involved."""

    def __init__(self, failing_ids: set[str] | None = None) -> None:
        self.failing_ids = failing_ids or set()
        self.recorded: list[dict] = []

    def record_activity(self, activity, *, base=None, timeout=10):
        if activity.get("id") in self.failing_ids:
            raise RuntimeError(f"kiln unreachable for {activity.get('id')}")
        self.recorded.append(activity)
        return {**activity, "sessionId": None}


def _install_fake_inbox(monkeypatch, fake: _FakeInbox) -> None:
    monkeypatch.setattr(sync.kiln_inbox_client, "record_activity", fake.record_activity)


def _install_fake_scrape(monkeypatch, result: ScrapeResult) -> list[dict]:
    """Scripted `list_recent_activities`: returns the canned `ScrapeResult`
    and records every call it saw."""
    calls: list[dict] = []

    def fake_list_recent_activities(*, base_url, limit, dry_run=False, screenshot_dir=None):
        calls.append({"base_url": base_url, "limit": limit, "dry_run": dry_run})
        return result

    monkeypatch.setattr(sync.strava_playwright, "list_recent_activities", fake_list_recent_activities)
    return calls


def test_scraped_activities_are_recorded_into_kiln(monkeypatch):
    activities = [{"id": "s1", "type": "Run"}, {"id": "s2", "type": "Ride"}]
    fake_inbox = _FakeInbox()
    _install_fake_inbox(monkeypatch, fake_inbox)
    calls = _install_fake_scrape(monkeypatch, ScrapeResult(success=True, activities=activities))

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", limit=5)

    assert result["scraped_count"] == 2
    assert result["recorded"] == 2
    assert result["failed"] == 0
    assert result["items"] == [{"id": "s1", "outcome": "recorded"}, {"id": "s2", "outcome": "recorded"}]
    assert fake_inbox.recorded == activities
    assert calls == [{"base_url": "http://mock.example", "limit": 5, "dry_run": False}]


def test_one_bad_activity_never_aborts_the_rest_of_the_batch(monkeypatch):
    activities = [{"id": "bad", "type": "Run"}, {"id": "good", "type": "Ride"}]
    fake_inbox = _FakeInbox(failing_ids={"bad"})
    _install_fake_inbox(monkeypatch, fake_inbox)
    _install_fake_scrape(monkeypatch, ScrapeResult(success=True, activities=activities))

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example")

    assert result["recorded"] == 1
    assert result["failed"] == 1
    assert result["items"][0]["outcome"] == "failed"
    assert "kiln unreachable" in result["items"][0]["error"]
    assert result["items"][1] == {"id": "good", "outcome": "recorded"}
    assert fake_inbox.recorded == [{"id": "good", "type": "Ride"}]


def test_a_scrape_failure_is_reported_without_recording_anything(monkeypatch):
    fake_inbox = _FakeInbox()
    _install_fake_inbox(monkeypatch, fake_inbox)
    _install_fake_scrape(monkeypatch, ScrapeResult(success=False, activities=[], error="TimeoutError: no entries", screenshot_path="/tmp/shot.png"))

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example")

    assert result["scraped_count"] == 0
    assert result["recorded"] == 0
    assert result["failed"] == 0
    assert result["items"] == []
    assert result["scrape_error"] == "TimeoutError: no entries"
    assert result["screenshot_path"] == "/tmp/shot.png"
    assert fake_inbox.recorded == []


def test_dry_run_never_touches_kiln(monkeypatch):
    fake_inbox = _FakeInbox()
    _install_fake_inbox(monkeypatch, fake_inbox)
    calls = _install_fake_scrape(monkeypatch, ScrapeResult(success=True, activities=[]))

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", dry_run=True)

    assert result["dry_run"] is True
    assert result["items"] == []
    assert fake_inbox.recorded == []
    assert calls[0]["dry_run"] is True


def test_default_limit_and_base_url_are_used_when_not_overridden(monkeypatch):
    fake_inbox = _FakeInbox()
    _install_fake_inbox(monkeypatch, fake_inbox)
    calls = _install_fake_scrape(monkeypatch, ScrapeResult(success=True, activities=[]))

    sync.run_sync()

    assert calls == [{"base_url": sync.strava_playwright.STRAVA_BASE_URL, "limit": sync.DEFAULT_LIMIT, "dry_run": False}]
