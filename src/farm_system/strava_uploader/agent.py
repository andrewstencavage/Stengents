"""Strava Uploader: an ADK agent that reports on and drives Kiln's Strava
outbox (issue #172).

The deterministic work — draining the outbox via Playwright — lives entirely
in ``sync.run_sync``, with no model in its loop, the same way
``auto_replan``'s decision logic stays model-free. This agent is only the
chat-usable wrapper around it (report status conversationally, run a sync on
request), matching ``farm_system``'s established shape: ``agent.py`` exports
``root_agent``, model resolution goes through ``stengents.utilities.model_source``,
and the real work is delegated to plain, independently testable functions.

The periodic/CLI path (``stengents strava-sync``, see ``cli.py``) calls
``sync.run_sync`` directly and never loads a model at all — this agent exists
for the conversational "what's pending" / "sync now" case only.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from google.genai import types

from stengents.utilities.model_source import resolve_model
from stengents.utilities.turn_log import TurnLogger
from . import kiln_outbox_client, sync

MODEL = resolve_model("qwen2.5:7b-8k")


def list_pending() -> dict:
    """Return Kiln's pending Strava outbox: ``pending_count`` (exact, already
    counted for you) and ``entries``, each with ``sessionId``, ``attempts``,
    ``lastError``, ``created``, and — when the Session is still around — its
    ``workout`` name, ``date``, and ``type``. Ground every answer in this data;
    do not invent entries or attempt counts.
    """
    entries = kiln_outbox_client.list_pending()
    compact = []
    for entry in entries:
        session = entry.get("session") or {}
        compact.append(
            {
                "sessionId": entry.get("sessionId"),
                "attempts": entry.get("attempts"),
                "lastError": entry.get("lastError"),
                "created": entry.get("created"),
                "workout": session.get("workoutName"),
                "date": (session.get("date") or "")[:10] or None,
                "type": session.get("type"),
            }
        )
    return {"pending_count": len(compact), "entries": compact}


def sync_now() -> dict:
    """Drain the pending Strava outbox right now: upload each entry via
    Playwright and report success or failure back to Kiln. Returns the sync
    summary — counts of ``uploaded``/``failed``/``skipped_at_max_attempts``
    plus a per-item list. Call this only when asked to sync or upload, not
    just to check status (use ``list_pending`` for that).
    """
    return sync.run_sync()


_turn_logger = TurnLogger(model=MODEL.as_record(), log_path=Path(".stengents/strava_uploader/turns.jsonl"))


root_agent = LlmAgent(
    name="strava_uploader",
    description="Reports on and drains Kiln's Strava upload outbox.",
    model=MODEL.llm,
    instruction=(
        "You are Strava Uploader, an assistant for a single athlete's Kiln-to-Strava upload "
        "queue. Call list_pending to see what's queued — report the pending_count and any "
        "entries with a nonzero attempts count or lastError plainly, the way a status update "
        "reads. Call sync_now only when asked to sync, upload, or drain the queue, and report "
        "its uploaded/failed/skipped_at_max_attempts counts afterward. Reply in at most two "
        "short sentences unless asked for detail. Base every claim on the data and say so "
        "plainly when it does not contain an answer."
    ),
    tools=[list_pending, sync_now],
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    before_agent_callback=_turn_logger.before_agent,
    after_agent_callback=_turn_logger.after_agent,
    before_tool_callback=_turn_logger.before_tool,
    after_tool_callback=_turn_logger.after_tool,
    on_tool_error_callback=_turn_logger.on_tool_error,
    after_model_callback=_turn_logger.after_model,
    on_model_error_callback=_turn_logger.on_model_error,
)
