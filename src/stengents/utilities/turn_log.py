"""Turn logging: one operational trace per request/response exchange.

A **Turn** is one request/response exchange with an agent — a user query, the
tool calls it triggered, and the agent's answer. Each Turn is appended as one
JSON line to a Turn log, giving a queryable record of what happened and its
**operational outcome** (``completed`` | ``degraded`` | ``errored`` |
``no_answer``). This is distinct from the harness's deterministic,
verification-based outcome: a Turn records whether the exchange *ran cleanly*,
not whether an answer was *correct*.

``build_turn_record`` is pure and unit-tested; the ``TurnLogger`` methods are the
thin ADK-callback shell that fills a per-Turn buffer and appends the line. Each
tool call's own bounded record comes from ``build_call_record`` in
:mod:`stengents.utilities.observation` (shared with the harness's tool-event
capture); the payload reducer beneath both lives there too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .observation import build_call_record

DEFAULT_LOG_PATH = Path(".stengents/turns.jsonl")


def build_turn_record(
    *,
    turn_id: str,
    session_id: str | None,
    started_at: str,
    duration_ms: int,
    model: dict[str, str],
    query: str | None,
    tool_calls: list[dict[str, object]],
    answer: str | None,
    model_error: str | None,
) -> dict[str, object]:
    """Assemble one Turn record and derive its operational outcome (pure).

    Outcome: an error with a recovered answer is ``degraded``; an error with no
    answer is ``errored``; no error and no answer is ``no_answer``; otherwise
    ``completed``.
    """
    had_error = model_error is not None or any(call.get("outcome") == "error" for call in tool_calls)
    if had_error:
        outcome = "degraded" if answer else "errored"
    elif not answer:
        outcome = "no_answer"
    else:
        outcome = "completed"
    record: dict[str, object] = {
        "schema_version": 1,
        "turn_id": turn_id,
        "session_id": session_id,
        "started_at": started_at,
        "duration_ms": duration_ms,
        "model": model,
        "query": query,
        "tool_calls": tool_calls,
        "answer": answer,
        "outcome": outcome,
    }
    if had_error:
        record["error"] = model_error or "tool_error"
    return record


@dataclass
class _TurnBuffer:
    session_id: str | None
    query: str | None
    started_at: str
    started_monotonic: float
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    tool_starts: dict[str, float] = field(default_factory=dict)
    answer: str | None = None
    model_error: str | None = None


def _content_text(content: object) -> str | None:
    parts = getattr(content, "parts", None) or []
    texts = [part.text for part in parts if getattr(part, "text", None)]
    return " ".join(texts).strip() or None if texts else None


class TurnLogger:
    """Capture one Turn record per invocation via ADK agent-level callbacks.

    Wire the methods as the agent's before/after-agent, before/after-tool,
    on-tool-error, after-model, and on-model-error callbacks. Every method
    returns ``None`` so it never alters agent behavior, and all capture is
    wrapped so a logging failure can never break a Turn.
    """

    def __init__(self, *, model: dict[str, str], log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.model = model
        self.log_path = Path(log_path)
        self._turns: dict[str, _TurnBuffer] = {}

    # --- Turn boundaries -------------------------------------------------
    def before_agent(self, *, callback_context: object) -> None:
        try:
            self._turns[str(getattr(callback_context, "invocation_id", ""))] = _TurnBuffer(
                session_id=getattr(getattr(callback_context, "session", None), "id", None),
                query=_content_text(getattr(callback_context, "user_content", None)),
                started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                started_monotonic=time.monotonic(),
            )
        except Exception:
            pass

    def after_agent(self, *, callback_context: object) -> None:
        buffer = self._turns.pop(str(getattr(callback_context, "invocation_id", "")), None)
        if buffer is None:
            return
        try:
            record = build_turn_record(
                turn_id=str(getattr(callback_context, "invocation_id", "")),
                session_id=buffer.session_id,
                started_at=buffer.started_at,
                duration_ms=round((time.monotonic() - buffer.started_monotonic) * 1000),
                model=self.model,
                query=buffer.query,
                tool_calls=buffer.tool_calls,
                answer=buffer.answer,
                model_error=buffer.model_error,
            )
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
        except Exception:
            pass

    # --- Tool calls ------------------------------------------------------
    def before_tool(self, *, tool: object, args: dict, tool_context: object) -> None:
        try:
            self._turns[str(getattr(tool_context, "invocation_id", ""))].tool_starts[
                str(getattr(tool_context, "function_call_id", ""))
            ] = time.monotonic()
        except Exception:
            pass

    def _tool_duration_ms(self, buffer: _TurnBuffer | None, tool_context: object) -> int:
        start = None
        if buffer is not None:
            start = buffer.tool_starts.pop(str(getattr(tool_context, "function_call_id", "")), None)
        return round((time.monotonic() - start) * 1000) if start is not None else 0

    def after_tool(self, *, tool: object, args: dict, tool_context: object, tool_response: object) -> None:
        buffer = self._turns.get(str(getattr(tool_context, "invocation_id", "")))
        if buffer is None:
            return
        try:
            buffer.tool_calls.append(build_call_record(
                str(getattr(tool, "name", "unknown")),
                args,
                self._tool_duration_ms(buffer, tool_context),
                result=tool_response,
            ))
        except Exception:
            pass

    def on_tool_error(self, *, tool: object, args: dict, tool_context: object, error: Exception) -> None:
        buffer = self._turns.get(str(getattr(tool_context, "invocation_id", "")))
        if buffer is None:
            return
        try:
            buffer.tool_calls.append(build_call_record(
                str(getattr(tool, "name", "unknown")),
                args,
                self._tool_duration_ms(buffer, tool_context),
                error=error,
            ))
        except Exception:
            pass

    # --- Model responses / errors ---------------------------------------
    def after_model(self, *, callback_context: object, llm_response: object) -> None:
        buffer = self._turns.get(str(getattr(callback_context, "invocation_id", "")))
        if buffer is None:
            return
        try:
            text = _content_text(getattr(llm_response, "content", None))
            if text:
                buffer.answer = text
        except Exception:
            pass

    def on_model_error(self, *, callback_context: object, llm_request: object, error: Exception) -> None:
        buffer = self._turns.get(str(getattr(callback_context, "invocation_id", "")))
        if buffer is None:
            return
        buffer.model_error = f"{type(error).__name__}: {error}"
