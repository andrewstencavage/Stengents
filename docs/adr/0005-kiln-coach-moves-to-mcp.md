# kiln_coach's server moves to Kiln's MCP boundary, superseding ADR-0001

ADR-0001 chose Kiln's HTTP API over its stdio MCP boundary because MCP's `list_sessions` dropped Session `status`, which finished-only filtering needed. That gap is now closed on Kiln's side — MCP's `list_sessions` exposes `status` directly — so the coach server drops its HTTP client entirely and becomes a pure MCP client, for both reads and the new Plan Template / Auto-replan writes this requires (see kiln `docs/adr/0001-reintroduce-plan-templates.md` and `docs/adr/0002-auto-replan-activates-without-review.md` for the paired Kiln-side decisions). This aligns it with Kiln's own stated boundary ("planning is exclusively the local stdio MCP boundary") instead of being the one exception to it.

Status: supersedes 0001-kiln-access-via-http.md

## Consequences

- Auto-replan and Plan Template maintenance are implemented as new responsibilities on the existing `stengents serve-coach` process (the same one that already serves Workout Review), not on the separate `kiln_coach` chat LlmAgent, which stays interactive-only and unchanged.
- The coach server now depends on spawning or connecting to Kiln's stdio MCP server on demand, rather than Kiln's always-running HTTP server.
