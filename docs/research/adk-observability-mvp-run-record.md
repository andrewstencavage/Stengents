# ADK observability for the MVP run record: findings for issue #4

## Answer

Use one ADK `BasePlugin` registered on the `Runner` as the capture boundary,
and let the fixture harness supply the values ADK cannot know.  The plugin's
`before_run_callback`/`after_run_callback` delimit the ADK invocation;
`before_tool_callback`, `after_tool_callback`, and `on_tool_error_callback`
capture every named harness action; and `on_event_callback` preserves the ADK
`invocation_id` and event sequence.  ADK describes plugins as global hooks for
logging/monitoring and specifies these callback signatures and their ordering.
[`BasePlugin` source](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/base_plugin.py)

The harness—not ADK—must create and own the complete JSON record.  It should
record its version/commit, fixture identity, immutable verification command,
allowed-source surface, wall-clock timestamps, elapsed monotonic durations,
and the verifier's exit/result.  This is necessary because ADK's invocation
context identifies an invocation and current agent/session, but has no concept
of this repository's harness or deterministic coding fixture.
[`InvocationContext` source](https://github.com/google/adk-python/blob/main/src/google/adk/agents/invocation_context.py)

## What each required field comes from

| Requirement | ADK source/hook | MVP decision |
| --- | --- | --- |
| Harness and fixture identity | None | Harness injects `harness` and `fixture` metadata before starting the runner. Include version/commit and a fixture content/version id. |
| ADK invocation/agent identity | `before_run_callback(invocation_context)`; event `invocation_id`, `author`, `timestamp`, and `id` | Store `invocation_id`, root-agent name, and optional event ids. `Event` explicitly supplies invocation id, author, generated id, and timestamp. [`Event` source](https://github.com/google/adk-python/blob/main/src/google/adk/events/event.py) |
| Model and provider | Root `LlmAgent.model` plus adapter configuration captured by the harness | Store the configured model identifier and an explicit provider/adapter label (for example `litellm/openai-compatible`). ADK allows `model` to be a string or `BaseLlm`, so provider identity is configuration, not a portable event field. [`LlmAgent` source](https://github.com/google/adk-python/blob/main/src/google/adk/agents/llm_agent.py) |
| Named tool events | Plugin `before_tool_callback(tool, tool_args, tool_context)`, `after_tool_callback(..., result)`, `on_tool_error_callback(..., error)` | Append one event per invocation with tool name, sanitized arguments/result summary, outcome, and per-action monotonic duration. ADK exposes exactly these hooks globally. [`BasePlugin` source](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/base_plugin.py) |
| Timing | Plugin start/end hooks plus `time.monotonic_ns()` in the harness/plugin | Record a UTC `started_at`/`finished_at` for correlation and `duration_ms` derived from monotonic time for measurements. ADK event timestamps are useful correlation metadata, but the MVP should not calculate elapsed time by subtracting wall-clock timestamps. [`Event.timestamp` source](https://github.com/google/adk-python/blob/main/src/google/adk/events/event.py) |
| Artifacts | Harness hashes the fixture workspace after the run; optionally observe ADK event `actions.artifact_delta` | The required coding-fixture artifacts are changed allowed source files and verifier output, which the harness can hash deterministically. ADK artifact deltas only name an artifact and its version; the artifact service stores payloads separately. [`EventActions` source](https://github.com/google/adk-python/blob/main/src/google/adk/events/event_actions.py) · [`BaseArtifactService` source](https://github.com/google/adk-python/blob/main/src/google/adk/artifacts/base_artifact_service.py) |
| Deterministic verification outcome | Harness verifier; optionally ADK evaluation result | Run the fixture-defined verification command after ADK ends (or when the named verification action is invoked); capture exit code, timeout/budget failure, and artifact validation. Do not infer pass/fail from final model text or ADK `is_final_response()`. ADK evaluation can report metric status, but it is not the fixture's required deterministic oracle. [ADK custom metrics](https://adk.dev/evaluate/custom_metrics/) |

## Smallest proposed JSON schema

```json
{
  "schema_version": 1,
  "run_id": "uuid",
  "started_at": "2026-07-22T18:40:00Z",
  "duration_ms": 8421,
  "harness": {"id": "stengents", "revision": "git-sha"},
  "fixture": {"id": "fix-null-parser", "revision": "content-sha"},
  "adk": {"version": "pinned-package-version", "invocation_id": "...", "agent": "coding_agent"},
  "model": {"provider": "openai-compatible", "name": "local-model"},
  "tool_events": [
    {"name": "read_file", "started_offset_ms": 34, "duration_ms": 2, "outcome": "ok"}
  ],
  "artifacts": [{"path": "src/parser.py", "sha256": "..."}],
  "verification": {"command": "pytest -q", "exit_code": 0, "passed": true}
}
```

`run_id`, the harness and fixture metadata, and `verification` are mandatory.
`adk.invocation_id` can be absent only if runner setup fails before an
invocation context exists.  Avoid raw prompts, model responses, tool arguments,
and tool results in this first schema: they are not required for the stated run
record and may contain source or secrets. Record bounded, redacted summaries
only if later debugging proves them necessary.

## Implementation boundary

The first implementation should create the record before invoking `Runner`,
register the plugin, then finalize the artifact digest and verifier result in a
`finally` path. This preserves a usable record for model/tool errors; ADK also
offers `on_model_error_callback`, `on_tool_error_callback`, and
`on_run_error_callback` for recording the error category without suppressing
the original failure. [`BasePlugin` source](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/base_plugin.py)
