"""Deterministic HTTP client for Kiln's Strava outbox (issue #172).

Kiln queues a FIT file per finished Session and exposes it only over its LAN
HTTP API (never a shared filesystem — see kiln's
``docs/adr/0003-strava-outbox-via-http.md``): list pending entries, stream one
entry's raw FIT bytes, and report an upload's outcome back. This module is the
plain, model-free client for that surface, mirroring ``kiln_coach.kiln_client``'s
style and its ``KILN_BASE_URL`` convention so both agents point at the same
Kiln instance with no new configuration.

Kiln itself never decides when to give up retrying a failed upload —
``mark_failed`` only increments the entry's ``attempts``/``lastError`` and
leaves it ``pending``; the give-up policy (a max-attempts cutoff) lives in
``sync.py``, the caller of this client.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://192.168.40.161:4173"


class OutboxEntryNotFound(RuntimeError):
    """No Strava outbox entry exists for a given Kiln Session id."""


def base_url() -> str:
    """The configured Kiln base URL, defaulting to the home-gym LAN address —
    the same ``KILN_BASE_URL`` env var ``kiln_coach.kiln_client`` reads, so
    both agents agree on which Kiln instance they talk to with no new
    configuration."""
    return os.environ.get("KILN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def list_pending(*, base: str | None = None, timeout: int = 10) -> list[dict]:
    """Fetch every pending Strava outbox entry, each carrying the queued FIT
    file's status/attempts and the full Kiln Session it was generated from
    (``session``, or ``None`` if that Session was since deleted)."""
    root = (base or base_url()).rstrip("/")
    query = urlencode({"status": "pending"})
    with urlopen(f"{root}/api/strava-outbox?{query}", timeout=timeout) as response:
        return json.load(response)


def fetch_fit(session_id: str, *, base: str | None = None, timeout: int = 10) -> bytes:
    """Fetch one Session's raw FIT Activity bytes, ready to hand to the
    Playwright uploader. Raises :class:`OutboxEntryNotFound` for a Session
    with no queued (or already-archived) FIT file."""
    root = (base or base_url()).rstrip("/")
    try:
        with urlopen(f"{root}/api/strava-outbox/{session_id}/fit", timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        if error.code == 404:
            raise OutboxEntryNotFound(session_id) from error
        raise


def _post_json(path: str, payload: dict, *, base: str | None, timeout: int) -> dict:
    root = (base or base_url()).rstrip("/")
    data = json.dumps(payload).encode()
    request = Request(f"{root}{path}", data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def mark_uploaded(session_id: str, *, strava_url: str | None = None, base: str | None = None, timeout: int = 10) -> dict:
    """Report a successful upload: Kiln moves the FIT file from
    ``pending/`` to ``archive/`` and records ``strava_url`` (when known) on
    the returned, now-``uploaded`` entry."""
    return _post_json(f"/api/strava-outbox/{session_id}/uploaded", {"stravaUrl": strava_url}, base=base, timeout=timeout)


def mark_failed(session_id: str, error: str, *, base: str | None = None, timeout: int = 10) -> dict:
    """Report a failed upload attempt: Kiln increments ``attempts`` and
    records ``lastError``, but the entry stays ``pending`` and its FIT file is
    never moved or deleted — Kiln never decides when to give up retrying;
    that policy is ``sync.py``'s ``max_attempts``."""
    return _post_json(f"/api/strava-outbox/{session_id}/failed", {"error": error}, base=base, timeout=timeout)
