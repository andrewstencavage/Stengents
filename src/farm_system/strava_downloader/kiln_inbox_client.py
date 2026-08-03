"""Deterministic HTTP client for Kiln's Strava inbox (kiln issue: "download
the runs from strava", the reverse of the outbox this repo's
``strava_uploader`` drains).

Kiln stores downloaded Strava Activities standalone (never as a Session — see
kiln's ``docs/adr/0004-strava-inbox-as-standalone-activities.md``) and exposes
them only over its LAN HTTP API, same boundary as the outbox. This module is
the plain, model-free client for that surface, mirroring
``strava_uploader.kiln_outbox_client``'s style and its ``KILN_BASE_URL``
convention so every farm_system agent points at the same Kiln instance with no
new configuration.

Recording an Activity is an idempotent upsert on Kiln's side (keyed by
Strava's own activity id) — this client can safely re-post an already-known
Activity on every sync pass with no dedup logic needed here.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://192.168.40.161:4173"


class StravaActivityNotFound(RuntimeError):
    """No Strava inbox entry exists for a given Strava activity id."""


def base_url() -> str:
    """The configured Kiln base URL, defaulting to the home-gym LAN address —
    the same ``KILN_BASE_URL`` env var every other farm_system Kiln client
    reads, so every agent agrees on which Kiln instance they talk to with no
    new configuration."""
    return os.environ.get("KILN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def list_activities(*, status: str | None = None, base: str | None = None, timeout: int = 10) -> list[dict]:
    """Fetch Strava inbox entries, optionally filtered by ``status``
    (``"unattached"`` or ``"attached"``). Each entry carries the downloaded
    Activity's summary fields plus ``sessionId`` (``None`` until attached)."""
    root = (base or base_url()).rstrip("/")
    path = f"{root}/api/strava-inbox"
    if status:
        path += f"?{urlencode({'status': status})}"
    with urlopen(path, timeout=timeout) as response:
        return json.load(response)


def record_activity(activity: dict, *, base: str | None = None, timeout: int = 10) -> dict:
    """Record (or upsert) one downloaded Strava Activity. ``activity`` must
    carry at least ``id``, ``type``, ``name``, ``date``, and ``movingSeconds``
    — ``distanceMeters``/``elevationMeters``/``averageHeartRate``/``raw`` are
    optional. Safe to call again for an already-known ``id``: Kiln refreshes
    the Strava-owned fields in place rather than duplicating the entry."""
    root = (base or base_url()).rstrip("/")
    data = json.dumps(activity).encode()
    request = Request(f"{root}/api/strava-inbox", data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def attach_activity(activity_id: str, session_id: str, *, base: str | None = None, timeout: int = 10) -> dict:
    """Attach a downloaded Activity to an already-logged Kiln Session,
    confirming it was that planned Workout. Raises
    :class:`StravaActivityNotFound` for an unknown ``activity_id``."""
    root = (base or base_url()).rstrip("/")
    data = json.dumps({"sessionId": session_id}).encode()
    request = Request(f"{root}/api/strava-inbox/{activity_id}/attach", data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 400:
            raise StravaActivityNotFound(activity_id) from error
        raise
