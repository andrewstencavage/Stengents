"""`strava_downloader`'s chat-usable wrapper (agent.py): the standard ADK
export plus its two thin tool wrappers around the deterministic
`kiln_inbox_client`/`sync` modules, mirroring `test_strava_uploader.py`'s
pattern for testing an agent's tool wiring without a live model."""

from __future__ import annotations

import farm_system.strava_downloader.agent as agent
from farm_system.strava_downloader import root_agent


def test_strava_downloader_exports_a_standard_adk_root_agent() -> None:
    assert root_agent.name == "strava_downloader"
    tool_names = {tool.__name__ for tool in root_agent.tools}
    assert tool_names == {"list_unattached", "download_now"}


def test_list_unattached_compacts_entries_to_a_lean_shape(monkeypatch) -> None:
    entries = [
        {
            "id": "strava-1",
            "sessionId": None,
            "type": "Run",
            "name": "Morning Run",
            "date": "2026-08-02T12:00:00Z",
            "movingSeconds": 1800,
            "distanceMeters": 5000,
            "elevationMeters": 42,
            "averageHeartRate": 152,
            "raw": {},
            "created": "2026-08-02T12:05:00Z",
        },
    ]
    monkeypatch.setattr(agent.kiln_inbox_client, "list_activities", lambda status: entries)

    result = agent.list_unattached()

    assert result == {
        "unattached_count": 1,
        "entries": [
            {
                "id": "strava-1",
                "type": "Run",
                "name": "Morning Run",
                "date": "2026-08-02T12:00:00Z",
                "movingSeconds": 1800,
                "distanceMeters": 5000,
                "elevationMeters": 42,
                "averageHeartRate": 152,
            },
        ],
    }


def test_download_now_delegates_to_run_sync(monkeypatch) -> None:
    summary = {"dry_run": False, "scraped_count": 0, "recorded": 0, "failed": 0, "items": []}
    monkeypatch.setattr(agent.sync, "run_sync", lambda: summary)

    assert agent.download_now() is summary
