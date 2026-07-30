"""Deterministic HTTP client for Kiln's local training history.

Kiln runs on the home-gym box and serves its browser API over the LAN on
``0.0.0.0:4173``. This module reads finished workout Sessions from that API
without a model in the loop, so it stays fast, free, and reproducible.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_BASE_URL = "http://192.168.40.161:4173"


def base_url() -> str:
    """The configured Kiln base URL, defaulting to the home-gym LAN address."""
    return os.environ.get("KILN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def finished_sessions(sessions: list[dict]) -> list[dict]:
    """Keep only completed Sessions; abandoned run-throughs are dropped as noise."""
    return [session for session in sessions if session.get("status") == "finished"]


def fetch_sessions(limit: int = 100, *, base: str | None = None, timeout: int = 10) -> list[dict]:
    """Fetch the most recent finished Kiln Sessions over the LAN HTTP API.

    Returns Kiln's full Session records. ``limit`` bounds the most-recent
    Sessions requested from Kiln before the finished-only filter is applied, so
    the result may hold fewer than ``limit`` entries when recent Sessions were
    abandoned.
    """
    root = (base or base_url()).rstrip("/")
    query = urlencode({"limit": limit})
    with urlopen(f"{root}/api/sessions?{query}", timeout=timeout) as response:
        sessions = json.load(response)
    return finished_sessions(sessions)


def fetch_workout(workout_id: str, *, limit: int = 100, base: str | None = None, timeout: int = 10) -> dict | None:
    """Fetch one finished Session by its stable ``id``, or ``None`` if not found.

    A Workout Review is addressed by the Session's ``id`` — Kiln's per-Session
    UUID, unique and stable across refetch. The review reads this one Session as
    the subject; its comparison history is selected separately from
    ``fetch_sessions`` (the finished-Session pool, each also carrying ``id``).
    Only finished Sessions are addressable; an abandoned ``id`` returns ``None``.
    """
    for session in fetch_sessions(limit=limit, base=base, timeout=timeout):
        if session.get("id") == workout_id:
            return session
    return None


def fetch_plans(*, base: str | None = None, timeout: int = 10) -> list[dict]:
    """Fetch every Plan (active, historical, and draft) over the LAN HTTP API.

    Kiln's own rule holds at most one activated (active/historical) Plan per
    calendar week, keyed by ``weekStart`` — the Plan streak computation relies
    on that to look up one Plan per week. Drafts are included but should be
    filtered out by callers that only care about weeks that were actually run.
    """
    root = (base or base_url()).rstrip("/")
    with urlopen(f"{root}/api/plans", timeout=timeout) as response:
        return json.load(response)


def performed_sets(activity: dict) -> list[dict]:
    """Project one Activity's performed sets onto the ``SetEvidence`` shape.

    Returns one row per performed set — ``set_index`` (its position), ``reps``,
    ``load`` (the measurement value, ``None`` for a bodyweight movement), and
    ``loadType`` (Kiln's own ``measurement.unit``, e.g. ``plate`` or ``sec``,
    passed through verbatim). This is the single source of truth reviews cite a
    set against. Freeform Activities (a warmup or cardio row carrying only a
    ``spec`` string and ``sets_completed``, no ``performedSets``) yield ``[]`` —
    they are citable only through non-set facts, never a set. ``reps`` is ``0``
    for a timed hold, where the work lives in ``load``/``loadType`` (a 45-second
    plank is ``reps=0, load=45, loadType="sec"``).
    """
    rows: list[dict] = []
    for set_index, entry in enumerate(activity.get("performedSets") or []):
        measurement = entry.get("measurement") or {}
        rows.append(
            {
                "set_index": set_index,
                "reps": entry.get("reps"),
                "load": measurement.get("value"),
                "loadType": measurement.get("unit"),
            }
        )
    return rows


def _collapse_sets(performed: list[dict]) -> str:
    """Render performed sets compactly, e.g. ``4×10@3plate`` or ``10@3plate, 8@3plate``."""
    rendered: list[str] = []
    for entry in performed:
        measurement = entry.get("measurement") or {}
        value, unit = measurement.get("value"), measurement.get("unit", "")
        load = f"@{value}{unit}" if value is not None else ""
        rendered.append(f"{entry.get('reps')}{load}")
    parts: list[str] = []
    for item in rendered:
        if parts and parts[-1][1] == item:
            parts[-1][0] += 1
        else:
            parts.append([1, item])
    return ", ".join(item if count == 1 else f"{count}×{item}" for count, item in parts)


def _compact_activity(activity: dict) -> dict:
    """Reduce one Activity to exercise, its performed sets, and any note/feel."""
    compact = {"exercise": activity.get("name"), "sets": _collapse_sets(activity.get("performedSets") or [])}
    if activity.get("skipped"):
        compact["skipped"] = True
    if activity.get("note"):
        compact["note"] = activity["note"]
    if activity.get("feel"):
        compact["feel"] = activity["feel"]
    return compact


def compact_sessions(sessions: list[dict]) -> list[dict]:
    """Project full Session records onto a lean shape a small model can read cheaply.

    Drops ids, plan linkage, muscle lists, and the prescribed/performed set
    duplication, keeping date, workout, type, minutes, feel, and a per-activity
    set summary with notes.
    """
    return [
        {
            "date": (session.get("date") or "")[:10],
            "workout": session.get("workoutName"),
            "type": session.get("type"),
            "minutes": session.get("minutes"),
            "feel": session.get("feel"),
            "activities": [_compact_activity(activity) for activity in session.get("activities") or []],
        }
        for session in sessions
    ]
