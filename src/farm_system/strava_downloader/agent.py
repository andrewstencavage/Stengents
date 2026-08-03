"""Strava Downloader: an ADK agent that reports on and fills Kiln's Strava
inbox (kiln's "download the runs from strava" ask, the reverse of issue
#172's uploader).

The deterministic work — scraping Strava and recording into Kiln — lives
entirely in ``sync.run_sync``, with no model in its loop, the same way
``strava_uploader``'s own upload work stays model-free. This agent is only
the chat-usable wrapper around it (report status conversationally, run a
download on request), matching ``farm_system``'s established shape:
``agent.py`` exports ``root_agent``, model resolution goes through
``stengents.utilities.model_source``, and the real work is delegated to
plain, independently testable functions.

The periodic/CLI path (``stengents strava-download``, see ``cli.py``) calls
``sync.run_sync`` directly and never loads a model at all — this agent exists
for the conversational "what's been downloaded" / "download now" case only.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from google.genai import types

from stengents.utilities.model_source import resolve_model
from stengents.utilities.turn_log import TurnLogger
from . import kiln_inbox_client, sync

MODEL = resolve_model("qwen2.5:7b-8k")


def list_unattached() -> dict:
    """Return Kiln's unattached Strava inbox entries — Activities downloaded
    from Strava that haven't yet been linked to a logged Kiln Session:
    ``unattached_count`` (exact, already counted for you) and ``entries``,
    each with ``id``, ``type``, ``name``, ``date``, ``movingSeconds``,
    ``distanceMeters``, ``elevationMeters``, and ``averageHeartRate``. Ground
    every answer in this data; do not invent entries or stats.
    """
    entries = kiln_inbox_client.list_activities(status="unattached")
    compact = [
        {
            "id": entry.get("id"),
            "type": entry.get("type"),
            "name": entry.get("name"),
            "date": entry.get("date"),
            "movingSeconds": entry.get("movingSeconds"),
            "distanceMeters": entry.get("distanceMeters"),
            "elevationMeters": entry.get("elevationMeters"),
            "averageHeartRate": entry.get("averageHeartRate"),
        }
        for entry in entries
    ]
    return {"unattached_count": len(compact), "entries": compact}


def download_now() -> dict:
    """Scrape Strava's recent Activities right now and record each into
    Kiln's inbox. Returns the sync summary — ``scraped_count``, counts of
    ``recorded``/``failed``, plus a per-item list (or a top-level
    ``scrape_error`` if Strava itself couldn't be reached/scraped). Call this
    only when asked to download, sync, or fill the inbox, not just to check
    status (use ``list_unattached`` for that).
    """
    return sync.run_sync()


_turn_logger = TurnLogger(model=MODEL.as_record(), log_path=Path(".stengents/strava_downloader/turns.jsonl"))


root_agent = LlmAgent(
    name="strava_downloader",
    description="Reports on and fills Kiln's downloaded-from-Strava inbox.",
    model=MODEL.llm,
    instruction=(
        "You are Strava Downloader, an assistant for a single athlete's Strava-to-Kiln import "
        "queue. Call list_unattached to see which downloaded Activities haven't been linked to a "
        "logged Session yet — report the unattached_count and a plain summary of each entry's "
        "type/name/date. Call download_now only when asked to download, sync, or fill the inbox, "
        "and report its recorded/failed counts (or scrape_error) afterward. Reply in at most two "
        "short sentences unless asked for detail. Base every claim on the data and say so plainly "
        "when it does not contain an answer."
    ),
    tools=[list_unattached, download_now],
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    before_agent_callback=_turn_logger.before_agent,
    after_agent_callback=_turn_logger.after_agent,
    before_tool_callback=_turn_logger.before_tool,
    after_tool_callback=_turn_logger.after_tool,
    on_tool_error_callback=_turn_logger.on_tool_error,
    after_model_callback=_turn_logger.after_model,
    on_model_error_callback=_turn_logger.on_model_error,
)
