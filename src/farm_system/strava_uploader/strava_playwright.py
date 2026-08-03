"""Playwright automation for Strava's upload form (issue #172).

Kiln queues a FIT file per finished Session (``kiln_outbox_client``); this
module is the other half — driving a real browser through Strava's upload
flow so no paid Strava API dependency is needed. ``upload_fit`` is the one
entry point ``sync.py`` calls per outbox entry.

**Selectors confirmed against production strava.com** (2026-08-01, by the
maintainer, uploading a real FIT file generated from a real finished Session
via an already-authenticated browser — see the package README's "What's
validated" section for the activity URL and exact evidence): the file input,
``input[name="name"]`` title field, and ``button[type="submit"]`` save button
below match the real page as-is, and a successful save does redirect to
``strava.com/activities/<id>``, exactly as this module assumed even before
that confirmation.

**Still open: automated login.** That confirmation used a browser already
logged into Strava by a human — this module still has no way to *establish*
that session itself. ``storage_state`` (below) is how an already-captured
session gets reused, but nothing here produces one; see the README for the
capture steps and why that remains a manual, periodic step rather than
something ``stengents strava-sync`` can do unattended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

STRAVA_BASE_URL = "https://www.strava.com"
DEFAULT_SCREENSHOT_DIR = Path(".stengents/strava_uploader/screenshots")

# Confirmed against production strava.com (2026-08-01): a successful save
# does redirect here, e.g. strava.com/activities/19558762653. The mock server
# matches this exactly (see mock_strava/server.py).
_ACTIVITY_URL_PATTERN = re.compile(r"/activities/[\w-]+")


@dataclass(frozen=True)
class UploadResult:
    """The outcome of one ``upload_fit`` call. ``screenshot_path`` is set
    whenever ``success`` is ``False`` and a screenshot could be captured —
    the diagnostics trail issue #172 asks upload failures to leave behind."""

    success: bool
    strava_url: str | None
    error: str | None
    screenshot_path: str | None


def _capture_screenshot(page: object, screenshot_dir: Path, filename: str) -> str | None:
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = screenshot_dir / f"{filename}-{stamp}.png"
        page.screenshot(path=str(path))  # type: ignore[attr-defined]
        return str(path)
    except Exception:  # noqa: BLE001 - a screenshot is best-effort diagnostics, never a second failure
        return None


def upload_fit(
    fit_bytes: bytes,
    filename: str,
    *,
    base_url: str = STRAVA_BASE_URL,
    activity_name: str | None = None,
    dry_run: bool = False,
    screenshot_dir: str | Path | None = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    storage_state: str | None = None,
) -> UploadResult:
    """Upload one FIT file through Strava's (or a compatible mock's) browser
    upload form.

    ``base_url`` is the upload target's origin — point it at
    ``https://www.strava.com`` (the default) for the real site, or at a local
    ``mock_strava.serve()`` instance for testing. ``dry_run=True`` skips
    Playwright and the network entirely, returning an immediate synthetic
    success — used by ``sync.py --dry-run`` to preview a batch with no
    browser, no target, and no side effects at all.

    ``storage_state`` is a Playwright storage-state JSON file path (cookies +
    local storage) for a pre-authenticated session — Strava's login wall is
    not automated here (see the package README); without it, a real-site run
    will simply see the login page and fail with a diagnostic screenshot,
    exactly like any other upload failure.

    Never raises: any failure (navigation, a missing selector, a timed-out
    wait for the post-save redirect) is caught, a screenshot is attempted, and
    a failed :class:`UploadResult` is returned — ``sync.py`` relies on this to
    keep one bad upload from aborting the rest of a batch.
    """
    if dry_run:
        return UploadResult(success=True, strava_url=None, error=None, screenshot_path=None)

    from playwright.sync_api import sync_playwright  # deferred: only the real upload path needs it

    root = base_url.rstrip("/")
    shot_dir = Path(screenshot_dir) if screenshot_dir is not None else DEFAULT_SCREENSHOT_DIR

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=storage_state) if storage_state else browser.new_context()
        page = context.new_page()
        try:
            page.goto(f"{root}/upload/select", timeout=timeout_ms)

            # --- confirmed against production strava.com, 2026-08-01 (module docstring) ---
            page.set_input_files(
                'input[type="file"]',
                files=[{"name": filename, "mimeType": "application/octet-stream", "buffer": fit_bytes}],
                timeout=timeout_ms,
            )
            # Strava renders the Title/Save form only after the file finishes
            # processing (an in-page async step, no navigation) — Playwright's
            # .fill()/.click() auto-wait for the elements to appear, so no
            # explicit wait is needed here. Leaving activity_name unset keeps
            # Strava's own auto-generated title (confirmed: it infers a
            # sensible one, e.g. "Morning Weight Training", from the FIT
            # file's sport/sub_sport and time of day).
            if activity_name:
                page.fill('input[name="name"]', activity_name, timeout=timeout_ms)
            page.click('button[type="submit"]', timeout=timeout_ms)
            page.wait_for_url(_ACTIVITY_URL_PATTERN, timeout=timeout_ms)
            # --- end confirmed block ---

            return UploadResult(success=True, strava_url=page.url, error=None, screenshot_path=None)
        except Exception as error:  # noqa: BLE001 - any failure becomes a diagnosable, non-raising result
            screenshot_path = _capture_screenshot(page, shot_dir, filename)
            return UploadResult(success=False, strava_url=None, error=f"{type(error).__name__}: {error}", screenshot_path=screenshot_path)
        finally:
            browser.close()
