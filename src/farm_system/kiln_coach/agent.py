"""Kiln Coach: an ADK agent that reads your Kiln training history.

The first capability is pulling your finished workout Sessions from the Kiln
instance on the home-gym box. The agent is structured to grow into planning and
Session summaries, which map onto Kiln's existing plan and strategy surfaces.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .kiln_client import compact_sessions, fetch_sessions
from .turn_log import TurnLogger

MODEL_BASE_URL = os.environ.get(
    "FARM_SYSTEM_MODEL_BASE_URL",
    os.environ.get("STENGENTS_MODEL_BASE_URL", "http://127.0.0.1:11434"),
)
MODEL_NAME = os.environ.get(
    "FARM_SYSTEM_MODEL_NAME",
    os.environ.get("STENGENTS_MODEL_NAME", "qwen2.5:7b-8k"),
)
MODEL_API_KEY = os.environ.get(
    "FARM_SYSTEM_MODEL_API_KEY",
    os.environ.get("STENGENTS_MODEL_API_KEY", "local"),
)


def get_workouts() -> dict:
    """Return all of the athlete's completed workouts with a ready-made count.

    A workout here is one training day the athlete finished. Returns
    ``workout_count`` (the exact total, already counted for you — use it directly
    for "how many" questions, never recount the list) and ``workouts`` (newest
    first). Each workout has its date, name, type, minutes, overall feel, and the
    exercises performed — each with a compact set summary (e.g. "4×10@3plate")
    and any note or feel. Ground every answer in this data and do not invent
    workouts, exercises, or numbers.
    """
    workouts = compact_sessions(fetch_sessions(limit=100))
    return {"workout_count": len(workouts), "workouts": workouts}


_turn_logger = TurnLogger(model={"provider": "openai-compatible", "name": MODEL_NAME})


root_agent = LlmAgent(
    name="kiln_coach",
    description="Reads your Kiln training history to answer questions about past workouts.",
    model=LiteLlm(
        model=f"openai/{MODEL_NAME}",
        api_base=f"{MODEL_BASE_URL.rstrip('/')}/v1",
        api_key=MODEL_API_KEY,
    ),
    instruction=(
        "You are Kiln Coach, an assistant for a single athlete's strength training. "
        "Call get_workouts to read their completed workouts. A workout is one completed "
        "training day — one entry in the workouts list. For 'how many' questions, answer "
        "with workout_count verbatim; never count workout names or list entries yourself "
        "(the same workout name repeats across days). Reply in at most two short sentences "
        "unless asked for detail — plain and conversational, the way a coach talks. Base "
        "every claim on the data and say so plainly when it does not contain an answer."
    ),
    tools=[get_workouts],
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    before_agent_callback=_turn_logger.before_agent,
    after_agent_callback=_turn_logger.after_agent,
    before_tool_callback=_turn_logger.before_tool,
    after_tool_callback=_turn_logger.after_tool,
    on_tool_error_callback=_turn_logger.on_tool_error,
    after_model_callback=_turn_logger.after_model,
    on_model_error_callback=_turn_logger.on_model_error,
)
