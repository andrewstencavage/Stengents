"""Deterministic Strava-outbox drain (issue #172): no LLM in this loop.

Mirrors ``stengents.auto_replan``'s shape — a plain function over plain
dicts, callable from a CLI subcommand or a test with no model connection at
all. ``run_sync`` is the whole pipeline for one pass: list Kiln's pending
Strava outbox, upload each entry's FIT file via Playwright, and report the
outcome back to Kiln. ``agent.py``'s ``sync_now`` tool is a thin conversational
wrapper around this same function, not a second implementation.
"""

from __future__ import annotations

from . import kiln_outbox_client, strava_playwright

DEFAULT_MAX_ATTEMPTS = 5


def _activity_name(entry: dict) -> str | None:
    """A human-readable Strava activity title from the outbox entry's Kiln
    Session, e.g. ``"Lower B — 2026-07-30"``. ``None`` when the Session was
    since deleted (``entry["session"]`` is ``None``) — the upload still
    proceeds untitled rather than being skipped."""
    session = entry.get("session") or {}
    name, date = session.get("workoutName"), (session.get("date") or "")[:10]
    if name and date:
        return f"{name} — {date}"
    return name or None


def _sync_one(entry: dict, *, kiln_base: str | None, playwright_base_url: str, dry_run: bool, screenshot_dir: str | None) -> dict:
    """Upload one outbox entry and report its outcome back to Kiln. Never
    raises: any exception (a fetch/upload/report failure alike) is caught and
    folded into a ``failed`` item, so one bad Session can never abort the rest
    of the batch — the same best-effort philosophy Kiln's own coach-review and
    Strava-outbox integrations use (never let a side integration take the main
    flow down with it)."""
    session_id = entry["sessionId"]
    try:
        fit_bytes = kiln_outbox_client.fetch_fit(session_id, base=kiln_base)
        result = strava_playwright.upload_fit(
            fit_bytes,
            f"{session_id}.fit",
            base_url=playwright_base_url,
            activity_name=_activity_name(entry),
            dry_run=dry_run,
            screenshot_dir=screenshot_dir,
        )
    except Exception as error:  # noqa: BLE001 - best-effort: one item's failure must not abort the batch
        result = strava_playwright.UploadResult(success=False, strava_url=None, error=f"{type(error).__name__}: {error}", screenshot_path=None)

    if result.success:
        if not dry_run:
            try:
                kiln_outbox_client.mark_uploaded(session_id, strava_url=result.strava_url, base=kiln_base)
            except Exception as error:  # noqa: BLE001 - the upload itself succeeded; a failed report is still worth surfacing, not raising
                return {"sessionId": session_id, "outcome": "failed", "error": f"upload ok but mark_uploaded failed: {error}"}
        return {"sessionId": session_id, "outcome": "uploaded", "strava_url": result.strava_url}

    if not dry_run:
        try:
            kiln_outbox_client.mark_failed(session_id, result.error or "unknown error", base=kiln_base)
        except Exception:  # noqa: BLE001 - best-effort: the failure is already captured below regardless
            pass
    return {"sessionId": session_id, "outcome": "failed", "error": result.error, "screenshot_path": result.screenshot_path}


def run_sync(
    *,
    kiln_base: str | None = None,
    playwright_base_url: str | None = None,
    dry_run: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    screenshot_dir: str | None = None,
) -> dict:
    """Drain Kiln's pending Strava outbox once: upload each entry, then
    report success or failure back to Kiln so it can archive or retry it.

    An entry already at ``max_attempts`` (Kiln's own ``attempts`` counter,
    incremented by a prior ``mark_failed``) is skipped untouched — its FIT
    file stays in Kiln's outbox either way, since Kiln never deletes on
    failure; ``attempts`` is the maintainer-visible record of "this one needs
    a look," not something this function papers over by giving up silently.

    Returns a JSON-serializable summary: counts plus one ``items`` entry per
    outbox entry considered, each with its ``sessionId`` and ``outcome``
    (``"uploaded"``, ``"failed"``, or ``"skipped_at_max_attempts"``).
    """
    target = (playwright_base_url or strava_playwright.STRAVA_BASE_URL).rstrip("/")
    pending = kiln_outbox_client.list_pending(base=kiln_base)

    items: list[dict] = []
    for entry in pending:
        if entry.get("attempts", 0) >= max_attempts:
            items.append({"sessionId": entry["sessionId"], "outcome": "skipped_at_max_attempts", "attempts": entry.get("attempts", 0)})
            continue
        items.append(_sync_one(entry, kiln_base=kiln_base, playwright_base_url=target, dry_run=dry_run, screenshot_dir=screenshot_dir))

    counts = {"uploaded": 0, "failed": 0, "skipped_at_max_attempts": 0}
    for item in items:
        counts[item["outcome"]] += 1

    return {"dry_run": dry_run, "pending_count": len(pending), **counts, "items": items}
