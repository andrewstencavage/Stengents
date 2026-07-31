"""Stdio MCP client for Kiln's planning boundary: `get_planning_context`,
`get_latest_strategy`, `list_plan_templates`, and `list_sessions` (with
`status`).

Kiln's MCP server is stdio-only — no LAN listener, no HTTP round trip
(ADR-0005, superseding ADR-0001's HTTP choice now that MCP's `list_sessions`
carries `status`). ``connect()`` spawns Kiln's combined browser+MCP process
(`npm run --silent kiln`, from a checkout named by ``KILN_MCP_CWD``) on
demand for one JSON-RPC session, the same way any MCP client — Claude Code
included — talks to a local stdio server: opened, used, and torn down.

This module is used only by the coach server (``stengents serve-coach``, see
`cli.py`). The `kiln_coach` chat LlmAgent keeps its own separate HTTP access
via `kiln_client`, untouched by this module's existence — the two transports
coexist rather than one replacing the other.

`cli.py` today wires only the three read seams `review_workout` needs
(`fetch`/`fetch_history`/`fetch_plans`, below) into the one live production
call site. `get_latest_strategy` and `list_plan_templates` are complete,
tested MCP reads with no caller yet — laid down for Auto-replan (#63/#66),
which needs them, rather than exercised by Workout Review today.

Each read below opens its own subprocess, uses it for one tool call, and
closes it — no session pooling across calls. `review_workout`'s three
independent seams (fetch/fetch_history/fetch_plans) mean one `/review/<id>`
request currently spawns up to three separate Kiln MCP processes rather than
sharing one; simple and correct over one call each stayed the priority for
this read-only migration. Revisit if latency matters more than it does today.

Three layers, innermost first:

* ``_JsonRpcStdio`` frames/parses newline-delimited JSON-RPC 2.0 over any
  stdin/stdout pair — a real subprocess's pipes in production, a canned
  in-memory fake in tests. No process-spawning logic lives here, so the
  request/response wire format is testable without ever starting Node.
* ``KilnMcpConnection`` adds the MCP ``initialize`` handshake and
  ``tools/call`` framing on top of one ``_JsonRpcStdio``.
* ``connect()`` spawns the real subprocess and yields a live
  ``KilnMcpConnection``, torn down on exit. Every public fetch function below
  takes ``connect`` as an injectable seam, so a test never spawns Kiln.
"""

from __future__ import annotations

import itertools
import json
import os
import select
import shlex
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Callable

from farm_system.kiln_coach import kiln_client

DEFAULT_COMMAND: tuple[str, ...] = ("npm", "run", "--silent", "kiln")
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "stengents-coach-server", "version": "0.1.0"}
DEFAULT_TIMEOUT = 15.0


class KilnMcpError(RuntimeError):
    """Raised when Kiln's MCP process can't be reached, times out, or a tool call fails."""


def _command() -> tuple[str, ...]:
    """The subprocess argv to spawn, overridable via ``KILN_MCP_COMMAND``."""
    raw = os.environ.get("KILN_MCP_COMMAND")
    return tuple(shlex.split(raw)) if raw else DEFAULT_COMMAND


def _cwd() -> str:
    """The Kiln checkout to run the MCP command from — required, no default:
    unlike `kiln_client`'s LAN IP, there's no reasonable guess for a local
    checkout path."""
    cwd = os.environ.get("KILN_MCP_CWD")
    if not cwd:
        raise KilnMcpError(
            "KILN_MCP_CWD is not set — point it at a Kiln checkout so "
            "`npm run --silent kiln` can be spawned for MCP reads."
        )
    return cwd


# --- JSON-RPC framing (transport-agnostic, testable without a process) ---


class _JsonRpcStdio:
    """Frames/parses newline-delimited JSON-RPC 2.0 messages over a
    stdin/stdout pair. Takes anything exposing ``.write``/``.flush`` (stdin)
    and ``.readline`` (stdout)."""

    def __init__(self, stdin, stdout) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._ids = itertools.count(1)

    def notify(self, method: str, params: dict | None = None) -> None:
        """Send a one-way JSON-RPC notification — no id, no reply expected."""
        self._write({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def request(self, method: str, params: dict | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """Send a JSON-RPC request and return its ``result``, raising
        ``KilnMcpError`` on a JSON-RPC error, a mismatched id, a closed
        pipe, or a read timeout."""
        request_id = next(self._ids)
        self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, **({"params": params} if params is not None else {})}
        )
        message = self._read(timeout=timeout)
        if message is None:
            raise KilnMcpError(f"no response from Kiln's MCP server for {method!r}")
        if message.get("id") != request_id:
            raise KilnMcpError(f"unexpected response id from Kiln's MCP server for {method!r}: {message!r}")
        if "error" in message:
            error = message.get("error") or {}
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise KilnMcpError(f"Kiln MCP error calling {method!r}: {detail}")
        return message.get("result") or {}

    def _write(self, message: dict) -> None:
        try:
            self._stdin.write(json.dumps(message) + "\n")
            self._stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            raise KilnMcpError(f"failed writing to Kiln's MCP server: {error}") from error

    def _read(self, *, timeout: float) -> dict | None:
        line = _readline(self._stdout, timeout)
        if not line or not line.strip():
            return None
        try:
            return json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise KilnMcpError(f"malformed JSON-RPC line from Kiln's MCP server: {line!r}") from error


def _readline(stream, timeout: float) -> str:
    """``stream.readline()``, bounded by ``timeout`` when ``stream`` is a
    real, selectable pipe. Falls back to a plain (unbounded) readline for a
    test fake with no file descriptor — those always return immediately."""
    try:
        fd = stream.fileno()
    except Exception:  # noqa: BLE001 — no real fd (a test fake); read directly.
        return stream.readline()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return ""
    return stream.readline()


# --- MCP session: handshake + tool calls ---------------------------------


class KilnMcpConnection:
    """One live MCP session with Kiln, from the ``initialize`` handshake
    through ``close()``."""

    def __init__(self, rpc: _JsonRpcStdio, *, on_close: Callable[[], None] | None = None) -> None:
        self._rpc = rpc
        self._on_close = on_close

    @classmethod
    def start(cls, rpc: _JsonRpcStdio, *, timeout: float = DEFAULT_TIMEOUT, on_close: Callable[[], None] | None = None) -> "KilnMcpConnection":
        """Complete the MCP handshake (``initialize`` then the
        ``notifications/initialized`` ack) and return a ready connection."""
        rpc.request(
            "initialize",
            {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": dict(CLIENT_INFO)},
            timeout=timeout,
        )
        rpc.notify("notifications/initialized")
        return cls(rpc, on_close=on_close)

    def call_tool(self, name: str, arguments: dict | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """Call one MCP tool and return its ``structuredContent`` (every tool
        this client uses always sets it, whether or not it declares a formal
        ``outputSchema`` — see kiln's `mcp/server.js`). Falls back to
        parsing the first text content block's JSON if a future tool ever
        omits it."""
        result = self._rpc.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        if result.get("isError"):
            raise KilnMcpError(f"Kiln MCP tool {name!r} failed: {_content_text(result)}")
        structured = result.get("structuredContent")
        if structured is not None:
            return structured
        return _parse_text_content(result)

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()


def _content_text(result: dict) -> str:
    for item in result.get("content") or []:
        if item.get("type") == "text":
            return item.get("text", "")
    return "unknown error"


def _parse_text_content(result: dict) -> dict:
    text = _content_text(result)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# --- connect(): spawn the real subprocess ---------------------------------

Connect = Callable[..., "AbstractContextManager[KilnMcpConnection]"]


@contextmanager
def connect(*, timeout: float = DEFAULT_TIMEOUT) -> Iterator[KilnMcpConnection]:
    """Spawn Kiln's stdio MCP server, complete the handshake, and yield a
    connected session — terminated and reaped on exit no matter how the
    ``with`` block ends.

    Raises ``KilnMcpError`` (never a bare ``OSError``/``subprocess`` one) on
    a missing ``KILN_MCP_CWD``, a failed spawn, or a failed handshake.
    """
    cwd = _cwd()  # raises before spawning if unset — nothing to clean up yet.
    try:
        process = subprocess.Popen(
            list(_command()),
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as error:
        raise KilnMcpError(f"failed to spawn Kiln's MCP server ({_command()!r} in {cwd!r}): {error}") from error

    def _terminate() -> None:
        try:
            if process.stdin:
                process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    try:
        connection = KilnMcpConnection.start(_JsonRpcStdio(process.stdin, process.stdout), timeout=timeout, on_close=_terminate)
    except KilnMcpError as error:
        stderr = process.stderr.read() if process.stderr else ""
        _terminate()
        detail = stderr.strip() if stderr else str(error)
        raise KilnMcpError(f"failed to initialize Kiln's MCP server: {detail}") from error

    try:
        yield connection
    finally:
        connection.close()


# --- public reads: the four tools this ticket covers ----------------------


def _call(name: str, arguments: dict, *, connect: Connect, timeout: float) -> dict:
    """Open one connection, call one tool, close it — the shared shape every
    public read below follows."""
    with connect(timeout=timeout) as session:
        return session.call_tool(name, arguments, timeout=timeout)


def _normalize_session(session: dict) -> dict:
    """MCP's `list_sessions`/`get_planning_context` key a Session's id as
    ``sessionId``; every other field matches what `kiln_client`'s HTTP fetch
    already returned (including each Activity's ``performedSets`` — Kiln's
    `toMcpSession` forwards Activities verbatim). Renaming back to ``id``
    keeps `review.py` (and its `kiln_client.finished_sessions`/`performed_sets`
    reuse) unchanged regardless of which transport is behind the seam."""
    session = dict(session)
    session["id"] = session.pop("sessionId", session.get("id"))
    return session


def list_sessions(
    *,
    limit: int = 100,
    plan_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    connect: Connect = connect,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Every Session MCP's `list_sessions` returns (any ``status``), newest
    ``limit`` first, normalized onto `kiln_client`'s dict shape."""
    arguments: dict[str, object] = {"limit": limit}
    if plan_id is not None:
        arguments["planId"] = plan_id
    if since is not None:
        arguments["from"] = since
    if until is not None:
        arguments["to"] = until
    result = _call("list_sessions", arguments, connect=connect, timeout=timeout)
    return [_normalize_session(entry) for entry in result.get("sessions") or []]


def get_planning_context(*, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """The latest Strategy, every Plan, recent Sessions (normalized), and the
    exercise catalog, via MCP's `get_planning_context`."""
    context = dict(_call("get_planning_context", {}, connect=connect, timeout=timeout))
    context["sessions"] = [_normalize_session(entry) for entry in context.get("sessions") or []]
    return context


def get_latest_strategy(*, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Kiln's latest Strategy via MCP's `get_latest_strategy`, or ``None``
    before a captain has ever authored one."""
    result = _call("get_latest_strategy", {}, connect=connect, timeout=timeout)
    return result.get("strategy")


def list_plan_templates(*, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> list[dict]:
    """Every named Plan Template via MCP's `list_plan_templates`."""
    result = _call("list_plan_templates", {}, connect=connect, timeout=timeout)
    return result.get("templates") or []


# --- public writes: the planning boundary's three write tools (#66) -------
#
# Laid down for Auto-replan's adapter (issue #66), following the exact same
# stdio pattern as every read above — one `connect()`, one `tools/call`, one
# teardown. Each takes already-serialized Kiln JSON (typically produced by
# `stengents.auto_replan.contract.to_kiln_json`) and passes it through
# verbatim as the tool's `arguments`, since Kiln's own zod schemas (not this
# client) are the single source of truth for what's valid.


def create_plan_template(content: dict, *, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Upsert one Plan Template via MCP's `create_plan_template` (keyed by
    ``content["name"]`` — calling again with an existing name replaces its
    content rather than duplicating it). Returns the stored template (Kiln's
    display shape, not the input shape) or ``{}`` if the tool responded with
    no ``template``."""
    result = _call("create_plan_template", content, connect=connect, timeout=timeout)
    return result.get("template") or {}


def create_plan_draft(content: dict, *, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Create one draft Plan via MCP's `create_plan_draft`. Draft only — this
    never activates or replaces the current Plan; call `select_current_plan`
    separately (with the returned Plan's ``planId``) to do that."""
    result = _call("create_plan_draft", content, connect=connect, timeout=timeout)
    return result.get("plan") or {}


def select_current_plan(plan_id: str, *, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Activate one existing Plan (typically a just-created draft) via MCP's
    `select_current_plan`, by its ``planId``."""
    result = _call("select_current_plan", {"planId": plan_id}, connect=connect, timeout=timeout)
    return result.get("plan") or {}


# --- kiln_client-compatible seams for review_workout -----------------------
#
# review_workout's fetch/fetch_history/fetch_plans seams were written against
# kiln_client's HTTP contract (dict shape, finished-only filtering, "every
# Plan" semantics). These mirror that contract exactly over MCP instead, so
# wiring the coach server to MCP (cli.py's `_serve_coach_command`) is a
# drop-in swap with no change to review.py itself.


def fetch_sessions(limit: int = 100, *, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> list[dict]:
    """Mirrors `kiln_client.fetch_sessions`: the most recent finished Kiln
    Sessions, via MCP's `list_sessions` instead of the HTTP `/api/sessions`."""
    sessions = list_sessions(limit=limit, connect=connect, timeout=timeout)
    return kiln_client.finished_sessions(sessions)


def fetch_workout(
    workout_id: str, *, limit: int = 100, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT
) -> dict | None:
    """Mirrors `kiln_client.fetch_workout`: one finished Session by its
    stable ``id``, or ``None`` if not found among the most recent ``limit``."""
    for session in fetch_sessions(limit=limit, connect=connect, timeout=timeout):
        if session.get("id") == workout_id:
            return session
    return None


def fetch_plans(*, connect: Connect = connect, timeout: float = DEFAULT_TIMEOUT) -> list[dict]:
    """Mirrors `kiln_client.fetch_plans`: every Plan (active, historical, and
    draft). MCP has no standalone "every Plan" tool call among the four this
    ticket covers, so this reads `get_planning_context`'s ``plans`` field,
    which already carries the full unfiltered list."""
    context = get_planning_context(connect=connect, timeout=timeout)
    return context.get("plans") or []
