"""`run_sync` end to end against fakes for both halves it orchestrates:
`kiln_outbox_client` (an in-memory outbox double, following
`test_kiln_mcp_client.py`'s injected-fake pattern) and `strava_playwright.upload_fit`
(a scripted stand-in — the real browser-driving path is `test_strava_playwright.py`'s
job, not this module's). Proves the three orchestration rules issue #172 cares
about: pending -> uploaded, a failure increments attempts and stays pending
(never deleted), and an entry already at max_attempts is skipped untouched."""

from __future__ import annotations

from farm_system.strava_uploader import sync
from farm_system.strava_uploader.strava_playwright import UploadResult


class _FakeOutbox:
    """An in-memory stand-in for `kiln_outbox_client`'s module functions —
    same shape/signature, no HTTP involved."""

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self.fit_bytes: dict[str, bytes] = {}
        self.fetch_calls: list[str] = []
        self.uploaded_calls: list[tuple[str, str | None]] = []
        self.failed_calls: list[tuple[str, str]] = []

    def list_pending(self, *, base=None, timeout=10):
        return self.entries

    def fetch_fit(self, session_id, *, base=None, timeout=10):
        self.fetch_calls.append(session_id)
        if session_id not in self.fit_bytes:
            raise RuntimeError(f"no fit queued for {session_id}")
        return self.fit_bytes[session_id]

    def mark_uploaded(self, session_id, *, strava_url=None, base=None, timeout=10):
        self.uploaded_calls.append((session_id, strava_url))
        return {"sessionId": session_id, "status": "uploaded", "stravaUrl": strava_url}

    def mark_failed(self, session_id, error, *, base=None, timeout=10):
        self.failed_calls.append((session_id, error))
        return {"sessionId": session_id, "status": "pending", "lastError": error}


def _install_fake_outbox(monkeypatch, fake: _FakeOutbox) -> None:
    monkeypatch.setattr(sync.kiln_outbox_client, "list_pending", fake.list_pending)
    monkeypatch.setattr(sync.kiln_outbox_client, "fetch_fit", fake.fetch_fit)
    monkeypatch.setattr(sync.kiln_outbox_client, "mark_uploaded", fake.mark_uploaded)
    monkeypatch.setattr(sync.kiln_outbox_client, "mark_failed", fake.mark_failed)


def _install_fake_upload(monkeypatch, outcomes: dict[str, UploadResult]) -> list[dict]:
    """Scripted `upload_fit`: returns each session's canned `UploadResult` by
    filename, and records every call it saw."""
    calls: list[dict] = []

    def fake_upload_fit(fit_bytes, filename, *, base_url, activity_name=None, dry_run=False, screenshot_dir=None):
        session_id = filename.rsplit(".", 1)[0]
        calls.append({"session_id": session_id, "base_url": base_url, "activity_name": activity_name, "dry_run": dry_run})
        return outcomes[session_id]

    monkeypatch.setattr(sync.strava_playwright, "upload_fit", fake_upload_fit)
    return calls


def test_a_pending_entry_uploads_and_is_reported_back_to_kiln(monkeypatch):
    fake = _FakeOutbox([{"sessionId": "s1", "attempts": 0, "session": {"workoutName": "Lower B", "date": "2026-07-30T10:00:00Z"}}])
    fake.fit_bytes["s1"] = b"FIT-BYTES"
    _install_fake_outbox(monkeypatch, fake)
    calls = _install_fake_upload(monkeypatch, {"s1": UploadResult(True, "https://strava.example/activities/1", None, None)})

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", max_attempts=5)

    assert result["uploaded"] == 1
    assert result["failed"] == 0
    assert result["skipped_at_max_attempts"] == 0
    assert result["items"] == [{"sessionId": "s1", "outcome": "uploaded", "strava_url": "https://strava.example/activities/1"}]
    assert fake.uploaded_calls == [("s1", "https://strava.example/activities/1")]
    assert fake.failed_calls == []
    assert calls == [{"session_id": "s1", "base_url": "http://mock.example", "activity_name": "Lower B — 2026-07-30", "dry_run": False}]


def test_a_failed_upload_increments_attempts_and_stays_pending(monkeypatch):
    fake = _FakeOutbox([{"sessionId": "s2", "attempts": 1, "session": None}])
    fake.fit_bytes["s2"] = b"FIT-BYTES"
    _install_fake_outbox(monkeypatch, fake)
    _install_fake_upload(monkeypatch, {"s2": UploadResult(False, None, "login wall", "/tmp/shot.png")})

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", max_attempts=5)

    assert result["failed"] == 1
    assert result["uploaded"] == 0
    assert fake.failed_calls == [("s2", "login wall")]
    assert fake.uploaded_calls == []
    assert result["items"] == [{"sessionId": "s2", "outcome": "failed", "error": "login wall", "screenshot_path": "/tmp/shot.png"}]


def test_an_entry_at_max_attempts_is_skipped_and_left_untouched(monkeypatch):
    fake = _FakeOutbox([{"sessionId": "s3", "attempts": 5, "session": None}])
    _install_fake_outbox(monkeypatch, fake)
    calls = _install_fake_upload(monkeypatch, {})

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", max_attempts=5)

    assert result["skipped_at_max_attempts"] == 1
    assert result["uploaded"] == 0
    assert result["failed"] == 0
    assert result["items"] == [{"sessionId": "s3", "outcome": "skipped_at_max_attempts", "attempts": 5}]
    # Never even fetched, let alone uploaded or reported — the FIT file stays
    # in Kiln's outbox exactly as it was.
    assert fake.fetch_calls == []
    assert calls == []
    assert fake.uploaded_calls == []
    assert fake.failed_calls == []


def test_dry_run_never_reports_back_to_kiln(monkeypatch):
    fake = _FakeOutbox([{"sessionId": "s4", "attempts": 0, "session": None}])
    fake.fit_bytes["s4"] = b"FIT-BYTES"
    _install_fake_outbox(monkeypatch, fake)
    calls = _install_fake_upload(monkeypatch, {"s4": UploadResult(True, "https://strava.example/activities/4", None, None)})

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example", dry_run=True)

    assert result["dry_run"] is True
    assert result["uploaded"] == 1
    assert fake.uploaded_calls == []  # nothing reported back to Kiln
    assert fake.failed_calls == []
    assert calls[0]["dry_run"] is True


def test_one_bad_entry_never_aborts_the_rest_of_the_batch(monkeypatch):
    """`fetch_fit` raising for one entry (e.g. the FIT file went missing
    between listing and fetching) must not stop later entries in the same
    pass from being processed."""
    fake = _FakeOutbox(
        [
            {"sessionId": "bad", "attempts": 0, "session": None},
            {"sessionId": "good", "attempts": 0, "session": None},
        ]
    )
    fake.fit_bytes["good"] = b"FIT-BYTES"  # "bad" is deliberately missing
    _install_fake_outbox(monkeypatch, fake)
    _install_fake_upload(monkeypatch, {"good": UploadResult(True, "https://strava.example/activities/9", None, None)})

    result = sync.run_sync(kiln_base="http://kiln.example", playwright_base_url="http://mock.example")

    assert result["items"][0]["outcome"] == "failed"
    assert result["items"][1] == {"sessionId": "good", "outcome": "uploaded", "strava_url": "https://strava.example/activities/9"}
    assert fake.uploaded_calls == [("good", "https://strava.example/activities/9")]
    assert fake.failed_calls == [("bad", "RuntimeError: no fit queued for bad")]


def test_pending_count_reflects_the_full_pending_list_regardless_of_outcome(monkeypatch):
    fake = _FakeOutbox(
        [
            {"sessionId": "s5", "attempts": 5, "session": None},
            {"sessionId": "s6", "attempts": 0, "session": None},
        ]
    )
    fake.fit_bytes["s6"] = b"FIT-BYTES"
    _install_fake_outbox(monkeypatch, fake)
    _install_fake_upload(monkeypatch, {"s6": UploadResult(True, "https://strava.example/activities/6", None, None)})

    result = sync.run_sync(max_attempts=5)

    assert result["pending_count"] == 2
