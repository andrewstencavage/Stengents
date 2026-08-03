"""`strava_uploader`'s chat-usable wrapper (agent.py): the standard ADK export
plus its two thin tool wrappers around the deterministic `kiln_outbox_client`/
`sync` modules, following `test_kiln_coach.py`'s pattern for testing an
agent's tool wiring without a live model."""

from __future__ import annotations

import farm_system.strava_uploader.agent as agent
from farm_system.strava_uploader import root_agent


def test_strava_uploader_exports_a_standard_adk_root_agent() -> None:
    assert root_agent.name == "strava_uploader"
    tool_names = {tool.__name__ for tool in root_agent.tools}
    assert tool_names == {"list_pending", "sync_now"}


def test_list_pending_compacts_entries_to_a_lean_shape(monkeypatch) -> None:
    entries = [
        {
            "sessionId": "s1",
            "status": "pending",
            "attempts": 2,
            "lastError": "login wall",
            "created": "2026-07-30T10:00:00Z",
            "session": {"workoutName": "Lower B", "date": "2026-07-30T10:00:00Z", "type": "strength"},
        },
        {
            "sessionId": "s2",
            "status": "pending",
            "attempts": 0,
            "lastError": None,
            "created": "2026-07-29T10:00:00Z",
            "session": None,
        },
    ]
    monkeypatch.setattr(agent.kiln_outbox_client, "list_pending", lambda: entries)

    result = agent.list_pending()

    assert result == {
        "pending_count": 2,
        "entries": [
            {
                "sessionId": "s1",
                "attempts": 2,
                "lastError": "login wall",
                "created": "2026-07-30T10:00:00Z",
                "workout": "Lower B",
                "date": "2026-07-30",
                "type": "strength",
            },
            {
                "sessionId": "s2",
                "attempts": 0,
                "lastError": None,
                "created": "2026-07-29T10:00:00Z",
                "workout": None,
                "date": None,
                "type": None,
            },
        ],
    }


def test_sync_now_delegates_to_run_sync(monkeypatch) -> None:
    summary = {"dry_run": False, "pending_count": 0, "uploaded": 0, "failed": 0, "skipped_at_max_attempts": 0, "items": []}
    monkeypatch.setattr(agent.sync, "run_sync", lambda: summary)

    assert agent.sync_now() is summary
