"""Deterministic Strava-inbox fill (kiln's "download the runs from strava"
ask): no LLM in this loop.

Mirrors ``strava_uploader.sync``'s shape — a plain function over plain dicts,
callable from a CLI subcommand or a test with no model connection at all.
``run_sync`` is the whole pipeline for one pass: scrape the athlete's recent
Strava Activities via Playwright, then record each one into Kiln's Strava
inbox. ``agent.py``'s ``download_now`` tool is a thin conversational wrapper
around this same function, not a second implementation.

Unlike the uploader (which reports both success *and* failure back to Kiln
per entry, since Kiln's outbox tracks retry state), there is nothing to
report back for a scrape or record failure here — Kiln was never told about
that Activity in the first place, so a failed item just stays out of the
inbox until the next pass tries again. ``record_activity`` is an idempotent
upsert on Kiln's side, so re-posting an already-known Activity on every pass
needs no dedup logic here.
"""

from __future__ import annotations

from . import kiln_inbox_client, strava_playwright

DEFAULT_LIMIT = strava_playwright.DEFAULT_LIMIT


def _record_one(activity: dict, *, kiln_base: str | None) -> dict:
    """Record one scraped Activity into Kiln's inbox. Never raises: a failed
    record (Kiln unreachable, a malformed activity, ...) is caught and folded
    into a ``failed`` item, so one bad Activity can never abort the rest of
    the batch — the same best-effort philosophy
    ``strava_uploader.sync._sync_one`` uses for uploads."""
    activity_id = activity.get("id")
    try:
        kiln_inbox_client.record_activity(activity, base=kiln_base)
        return {"id": activity_id, "outcome": "recorded"}
    except Exception as error:  # noqa: BLE001 - best-effort: one item's failure must not abort the batch
        return {"id": activity_id, "outcome": "failed", "error": f"{type(error).__name__}: {error}"}


def run_sync(
    *,
    kiln_base: str | None = None,
    playwright_base_url: str | None = None,
    limit: int = DEFAULT_LIMIT,
    dry_run: bool = False,
    screenshot_dir: str | None = None,
) -> dict:
    """Fill Kiln's Strava inbox once: scrape the athlete's ``limit`` most
    recent Activities, then record each into Kiln.

    Returns a JSON-serializable summary: counts plus one ``items`` entry per
    scraped Activity, each with its ``id`` and ``outcome`` (``"recorded"`` or
    ``"failed"``). A scrape failure itself (Playwright/navigation trouble, not
    a per-item record failure) is reported as ``{"scrape_error": ...}`` with
    no items, mirroring how ``strava_uploader.sync`` never lets a bad pass
    (e.g. Kiln unreachable) crash the periodic loop above it — that's this
    function's caller's job (``cli.py``'s ``_strava_download_command``) to
    catch, same as the uploader's own periodic loop.
    """
    target = (playwright_base_url or strava_playwright.STRAVA_BASE_URL).rstrip("/")
    scraped = strava_playwright.list_recent_activities(base_url=target, limit=limit, dry_run=dry_run, screenshot_dir=screenshot_dir)

    if not scraped.success:
        return {"dry_run": dry_run, "scraped_count": 0, "recorded": 0, "failed": 0, "items": [], "scrape_error": scraped.error, "screenshot_path": scraped.screenshot_path}

    items: list[dict] = [] if dry_run else [_record_one(activity, kiln_base=kiln_base) for activity in scraped.activities]

    counts = {"recorded": 0, "failed": 0}
    for item in items:
        counts[item["outcome"]] += 1

    return {"dry_run": dry_run, "scraped_count": len(scraped.activities), **counts, "items": items}
