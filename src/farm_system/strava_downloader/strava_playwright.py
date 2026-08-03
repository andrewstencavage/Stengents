"""Playwright automation for Strava's activity list (the download half of
kiln's "download the runs from strava" ask).

``strava_uploader.strava_playwright`` drives a browser through a Strava
*form* (one write per FIT file); this module drives one through a Strava
*page* (a read: scrape the athlete's recent Activities) and hands back a
plain list of dicts shaped for ``kiln_inbox_client.record_activity``.
``list_recent_activities`` is the one entry point ``sync.py`` calls per pass.

**Confirmed against production strava.com (2026-08-03)**, via an
already-authenticated browser (Chrome DevTools MCP, not this module's own
Playwright — no automated login exists here or in the sibling uploader). The
training log at ``/athlete/training`` is a plain, server-rendered
``<table class="... activities ...">`` (not the JS-driven
``[data-testid="activity-entry"]`` div structure this module originally
guessed): each Activity is a ``tr.training-activity-row`` with
``.col-type``/``.col-date``/``.col-title`` (containing the
``/activities/<id>`` link)/``.col-time``/``.col-dist``/``.col-elev`` cells,
20 rows per page. Confirmed against real rows including the same
``Morning Weight Training`` activity (``/activities/19558762653``) the
uploader's own 2026-08-01 confirmation produced.

Two real gaps this confirmation turned up, neither of which was knowable
from a guess:

- **No average-heart-rate column exists on this page at all.**
  ``averageHeartRate`` is always ``None`` from this scrape; getting it would
  require an extra page load per Activity (its detail page), not attempted
  here — see the package README.
- **Distance/elevation are unit-formatted display text** (e.g. ``"3.53 mi"``,
  ``"101 ft"``), confirming the module's own prior guess about the likeliest
  gap. ``_parse_measurement_meters`` below converts miles/kilometers/feet/
  meters to meters; an account using different display units needs
  verifying separately (this confirmation's account is imperial).
- **The date cell has no time-of-day** (``"Sat, 8/1/2026"`` only) — every
  recorded Activity's ``date`` is midnight UTC on that calendar day, not a
  real start time. Kiln's own Session dates come from FIT data with a real
  time; a Strava-inbox Activity's ``date`` is coarser by design until/unless
  a detail-page visit is added.

Still open: automated login. This confirmation used a browser already
authenticated by a human (via the Chrome extension, not Playwright);
``storage_state`` below is how an already-captured *Playwright* session
would get reused for an unattended run, but nothing here produces one yet —
see the package README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

STRAVA_BASE_URL = "https://www.strava.com"
TRAINING_LOG_PATH = "/athlete/training"
DEFAULT_SCREENSHOT_DIR = Path(".stengents/strava_downloader/screenshots")
DEFAULT_LIMIT = 10

# Confirmed against production strava.com (2026-08-03): a training-log row's
# cells always render as "<value> <unit-word>" (e.g. "3.53 mi", "101 ft"),
# the unit word being the visible text of a nested `<abbr class="unit"
# title="...">` tag. Extending this map is how a metric-configured account
# (km/m) would be supported; only mi/ft were observed here.
_UNIT_TO_METERS = {"mi": 1609.344, "km": 1000.0, "ft": 0.3048, "m": 1.0}
_MEASUREMENT_PATTERN = re.compile(r"([\d.]+)\s*([a-zA-Z]+)")
_DATE_PATTERN = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


@dataclass(frozen=True)
class ScrapeResult:
    """The outcome of one ``list_recent_activities`` call. ``screenshot_path``
    is set whenever ``success`` is ``False`` and a screenshot could be
    captured — the same failure-diagnostics trail
    ``strava_uploader.strava_playwright.UploadResult`` leaves behind."""

    success: bool
    activities: list[dict] = field(default_factory=list)
    error: str | None = None
    screenshot_path: str | None = None


def _capture_screenshot(page: object, screenshot_dir: Path, filename: str) -> str | None:
    try:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = screenshot_dir / f"{filename}-{stamp}.png"
        page.screenshot(path=str(path))  # type: ignore[attr-defined]
        return str(path)
    except Exception:  # noqa: BLE001 - a screenshot is best-effort diagnostics, never a second failure
        return None


def _cell_text(row: object, selector: str) -> str:
    cell = row.locator(selector)  # type: ignore[attr-defined]
    if cell.count() == 0:  # type: ignore[attr-defined]
        return ""
    text = cell.first.inner_text()  # type: ignore[attr-defined]
    return text.strip() if text else ""


def _parse_duration_seconds(text: str) -> int | None:
    """``"46:00"`` -> 2760, ``"1:03:42"`` -> 3822. Confirmed against
    production: both mm:ss and h:mm:ss appear depending on Activity length."""
    if not text:
        return None
    try:
        parts = [int(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _parse_measurement_meters(text: str) -> float | None:
    """``"3.53 mi"`` -> 5680.99, ``"101 ft"`` -> 30.78. See
    ``_UNIT_TO_METERS``'s docstring note above for the confirmed shape."""
    match = _MEASUREMENT_PATTERN.search(text)
    if not match:
        return None
    value, unit = match.groups()
    factor = _UNIT_TO_METERS.get(unit.lower())
    if factor is None:
        return None
    return round(float(value) * factor, 2)


def _parse_date_iso(text: str) -> str | None:
    """``"Sat, 8/1/2026"`` -> ``"2026-08-01T00:00:00Z"``. No time-of-day is
    present on the training log (module docstring) — every Activity's date
    is midnight UTC on the calendar day shown."""
    match = _DATE_PATTERN.search(text)
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}T00:00:00Z"


def _scrape_row(row: object) -> dict:
    """One ``tr.training-activity-row`` -> a dict shaped for
    ``kiln_inbox_client.record_activity``. See the module docstring for the
    confirmed DOM shape this assumes."""
    # Confirmed against production (2026-08-03): this link's `href` renders as an
    # *absolute* URL (`https://www.strava.com/activities/<id>`), not the relative
    # `/activities/<id>` originally guessed — an `a[href^="/activities/"]` selector
    # would silently match nothing on the real page. Selecting on `.col-title a`
    # alone (there is only ever one link in that cell) and taking the id from the
    # last path segment works for either form.
    title_link = row.locator(".col-title a")  # type: ignore[attr-defined]
    href = title_link.first.get_attribute("href") if title_link.count() else None  # type: ignore[attr-defined]
    activity_id = href.rstrip("/").rsplit("/", 1)[-1] if href else None
    return {
        "id": activity_id,
        "type": _cell_text(row, ".col-type") or None,
        "name": _cell_text(row, ".col-title") or None,
        "date": _parse_date_iso(_cell_text(row, ".col-date")),
        "movingSeconds": _parse_duration_seconds(_cell_text(row, ".col-time")),
        "distanceMeters": _parse_measurement_meters(_cell_text(row, ".col-dist")),
        "elevationMeters": _parse_measurement_meters(_cell_text(row, ".col-elev")),
        "averageHeartRate": None,  # not present on the training-log list view (module docstring)
    }


def list_recent_activities(
    *,
    base_url: str = STRAVA_BASE_URL,
    limit: int = DEFAULT_LIMIT,
    dry_run: bool = False,
    screenshot_dir: str | Path | None = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    storage_state: str | None = None,
) -> ScrapeResult:
    """Scrape the athlete's ``limit`` most recent Activities from Strava's (or
    a compatible mock's) training-log page.

    ``base_url`` is the site's origin — point it at ``https://www.strava.com``
    (the default) for the real site, or at a local ``mock_strava.serve()``
    instance for testing. ``dry_run=True`` skips Playwright and the network
    entirely, returning an immediate synthetic empty result — mirrors
    ``strava_uploader.strava_playwright.upload_fit``'s own ``dry_run`` shape:
    it proves the calling loop's machinery with no browser, no target, and no
    side effects, not a preview of real content.

    ``storage_state`` is a Playwright storage-state JSON file path for a
    pre-authenticated session — Strava's login wall is not automated here
    (see the package README); without it, a real-site run will simply see the
    login page and scrape nothing useful, failing with a diagnostic
    screenshot like any other failure here.

    Never raises: any failure (navigation, a missing selector, a timed-out
    wait for the page to render) is caught, a screenshot is attempted, and a
    failed :class:`ScrapeResult` is returned — ``sync.py`` relies on this to
    keep a scrape failure from crashing the whole pass.

    Only the first page of the training log is scraped (20 rows, confirmed —
    see module docstring) — pagination / "load more" for activities beyond
    that page is not handled (out of scope for this first version, same as
    kiln's own map issue #74 punted a dedicated backfill experience for
    anything older than the recent window).
    """
    if dry_run:
        return ScrapeResult(success=True, activities=[])

    from playwright.sync_api import sync_playwright  # deferred: only the real scrape path needs it

    root = base_url.rstrip("/")
    shot_dir = Path(screenshot_dir) if screenshot_dir is not None else DEFAULT_SCREENSHOT_DIR

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=storage_state) if storage_state else browser.new_context()
        page = context.new_page()
        try:
            page.goto(f"{root}{TRAINING_LOG_PATH}", timeout=timeout_ms)
            rows = page.locator("table.activities tbody tr.training-activity-row")
            rows.first.wait_for(timeout=timeout_ms)
            count = min(rows.count(), limit)
            activities = [_scrape_row(rows.nth(index)) for index in range(count)]
            return ScrapeResult(success=True, activities=activities)
        except Exception as error:  # noqa: BLE001 - any failure becomes a diagnosable, non-raising result
            screenshot_path = _capture_screenshot(page, shot_dir, "training-log")
            return ScrapeResult(success=False, activities=[], error=f"{type(error).__name__}: {error}", screenshot_path=screenshot_path)
        finally:
            browser.close()
