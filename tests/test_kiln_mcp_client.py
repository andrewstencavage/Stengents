"""Isolated request/response tests for `kiln_mcp_client` (#65).

No real subprocess is ever spawned here: `_JsonRpcStdio` is exercised against
an in-memory fake stdin/stdout pair, and the higher-level fetch functions are
exercised against a fake `connect` context manager — mirroring how
`review.py`'s own tests never touch the network. A separate manual smoke test
(not part of this suite) confirmed the wire format against a live `npm run
--silent kiln` process.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from stengents.workout_review import kiln_mcp_client
from stengents.workout_review.kiln_mcp_client import (
    KilnMcpConnection,
    KilnMcpError,
    _JsonRpcStdio,
)


# --- _JsonRpcStdio: pure request/response framing ------------------------


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[dict] = []

    def write(self, data: str) -> None:
        self.written.append(json.loads(data))

    def flush(self) -> None:
        pass


class _FakeStdout:
    def __init__(self, lines: list[dict]) -> None:
        self._lines = [json.dumps(line) + "\n" for line in lines]

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


def test_request_writes_a_well_formed_json_rpc_request_and_returns_the_result() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout([{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}])
    rpc = _JsonRpcStdio(stdin, stdout)

    result = rpc.request("tools/call", {"name": "list_sessions", "arguments": {"limit": 5}})

    assert result == {"ok": True}
    assert stdin.written == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "list_sessions", "arguments": {"limit": 5}},
        }
    ]


def test_notify_sends_no_id_and_expects_no_response() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout([])
    rpc = _JsonRpcStdio(stdin, stdout)

    rpc.notify("notifications/initialized")

    assert stdin.written == [{"jsonrpc": "2.0", "method": "notifications/initialized"}]


def test_request_raises_on_a_json_rpc_error_response() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout([{"jsonrpc": "2.0", "id": 1, "error": {"message": "boom"}}])
    rpc = _JsonRpcStdio(stdin, stdout)

    with pytest.raises(KilnMcpError, match="boom"):
        rpc.request("tools/call", {"name": "list_sessions"})


def test_request_raises_when_the_process_closes_without_replying() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout([])
    rpc = _JsonRpcStdio(stdin, stdout)

    with pytest.raises(KilnMcpError):
        rpc.request("tools/call", {"name": "list_sessions"})


def test_request_raises_on_malformed_json() -> None:
    stdin = _FakeStdin()

    class _BadStdout:
        def readline(self) -> str:
            return "not json\n"

    rpc = _JsonRpcStdio(stdin, _BadStdout())

    with pytest.raises(KilnMcpError):
        rpc.request("tools/call", {"name": "list_sessions"})


# --- KilnMcpConnection: handshake + tool call ----------------------------


def test_start_performs_the_initialize_handshake_then_notifies_initialized() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "kiln-mcp"}},
            }
        ]
    )
    rpc = _JsonRpcStdio(stdin, stdout)

    KilnMcpConnection.start(rpc)

    assert stdin.written[0]["method"] == "initialize"
    assert stdin.written[0]["params"]["protocolVersion"] == "2025-06-18"
    assert stdin.written[1] == {"jsonrpc": "2.0", "method": "notifications/initialized"}


def test_call_tool_returns_structured_content() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18"}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "content": [{"type": "text", "text": '{"sessions": []}'}],
                    "structuredContent": {"sessions": []},
                },
            },
        ]
    )
    connection = KilnMcpConnection.start(_JsonRpcStdio(stdin, stdout))

    result = connection.call_tool("list_sessions", {"limit": 5})

    assert result == {"sessions": []}
    assert stdin.written[-1]["params"] == {"name": "list_sessions", "arguments": {"limit": 5}}


def test_call_tool_raises_when_the_tool_reports_an_error() -> None:
    stdin = _FakeStdin()
    stdout = _FakeStdout(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"isError": True, "content": [{"type": "text", "text": "Plan not found"}]},
            },
        ]
    )
    connection = KilnMcpConnection.start(_JsonRpcStdio(stdin, stdout))

    with pytest.raises(KilnMcpError, match="Plan not found"):
        connection.call_tool("get_plan_template", {"name": "missing"})


# --- higher-level fetch functions: injected fake connect -----------------


class _FakeConnection:
    def __init__(self, responses: dict[str, dict]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict | None = None, **_: object) -> dict:
        self.calls.append((name, arguments or {}))
        return self.responses[name]


def _fake_connect(fake: _FakeConnection):
    @contextmanager
    def connect(**_: object):
        yield fake

    return connect


def test_list_sessions_normalizes_session_id_to_id() -> None:
    fake = _FakeConnection(
        {
            "list_sessions": {
                "sessions": [
                    {"sessionId": "s1", "status": "finished", "activities": []},
                    {"sessionId": "s2", "status": "abandoned", "activities": []},
                ]
            }
        }
    )

    sessions = kiln_mcp_client.list_sessions(limit=7, connect=_fake_connect(fake))

    assert [s["id"] for s in sessions] == ["s1", "s2"]
    assert fake.calls == [("list_sessions", {"limit": 7})]


def test_fetch_sessions_keeps_only_finished_sessions() -> None:
    fake = _FakeConnection(
        {
            "list_sessions": {
                "sessions": [
                    {"sessionId": "s1", "status": "finished", "activities": []},
                    {"sessionId": "s2", "status": "abandoned", "activities": []},
                    {"sessionId": "s3", "status": "finished", "activities": []},
                ]
            }
        }
    )

    sessions = kiln_mcp_client.fetch_sessions(limit=100, connect=_fake_connect(fake))

    assert [s["id"] for s in sessions] == ["s1", "s3"]


def test_fetch_workout_finds_the_matching_finished_session_by_id() -> None:
    fake = _FakeConnection(
        {
            "list_sessions": {
                "sessions": [
                    {"sessionId": "w1", "status": "finished", "activities": []},
                    {"sessionId": "w2", "status": "finished", "activities": []},
                ]
            }
        }
    )

    assert kiln_mcp_client.fetch_workout("w2", connect=_fake_connect(fake))["id"] == "w2"


def test_fetch_workout_returns_none_for_an_unknown_id() -> None:
    fake = _FakeConnection({"list_sessions": {"sessions": []}})

    assert kiln_mcp_client.fetch_workout("missing", connect=_fake_connect(fake)) is None


def test_fetch_plans_reads_every_plan_from_planning_context() -> None:
    fake = _FakeConnection(
        {
            "get_planning_context": {
                "strategy": None,
                "plans": [{"id": "p1"}, {"id": "p2"}],
                "sessions": [],
                "exercises": [],
            }
        }
    )

    assert kiln_mcp_client.fetch_plans(connect=_fake_connect(fake)) == [{"id": "p1"}, {"id": "p2"}]


def test_get_planning_context_normalizes_its_embedded_sessions() -> None:
    fake = _FakeConnection(
        {
            "get_planning_context": {
                "strategy": None,
                "plans": [],
                "sessions": [{"sessionId": "s1", "status": "finished", "activities": []}],
                "exercises": [],
            }
        }
    )

    context = kiln_mcp_client.get_planning_context(connect=_fake_connect(fake))

    assert context["sessions"][0]["id"] == "s1"


def test_get_latest_strategy_returns_the_strategy_or_none() -> None:
    fake = _FakeConnection({"get_latest_strategy": {"strategy": {"version": 3, "markdown": "..."}}})

    assert kiln_mcp_client.get_latest_strategy(connect=_fake_connect(fake)) == {"version": 3, "markdown": "..."}


def test_get_latest_strategy_is_none_when_kiln_has_no_strategy_yet() -> None:
    fake = _FakeConnection({"get_latest_strategy": {"strategy": None}})

    assert kiln_mcp_client.get_latest_strategy(connect=_fake_connect(fake)) is None


def test_list_plan_templates_returns_the_template_list() -> None:
    fake = _FakeConnection({"list_plan_templates": {"templates": [{"name": "strength"}]}})

    assert kiln_mcp_client.list_plan_templates(connect=_fake_connect(fake)) == [{"name": "strength"}]


def test_list_plan_templates_is_empty_list_not_none_when_kiln_has_none() -> None:
    fake = _FakeConnection({"list_plan_templates": {"templates": []}})

    assert kiln_mcp_client.list_plan_templates(connect=_fake_connect(fake)) == []


# --- connect(): requires KILN_MCP_CWD ------------------------------------


def test_connect_raises_a_clear_error_when_kiln_mcp_cwd_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("KILN_MCP_CWD", raising=False)

    with pytest.raises(KilnMcpError, match="KILN_MCP_CWD"):
        with kiln_mcp_client.connect():
            pass
