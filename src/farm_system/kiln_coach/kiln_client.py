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
