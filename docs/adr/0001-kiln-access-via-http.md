# Kiln Coach reads Kiln over its LAN HTTP API, not MCP

The `kiln_coach` farm-system agent reads training history from the Kiln
instance on the home-gym box. We access Kiln's browser HTTP API directly
(`GET http://<gym>:4173/api/sessions`) rather than Kiln's stdio MCP boundary,
even though Kiln's own `AGENTS.md` names MCP the sanctioned agent surface. HTTP
is chosen because it is reachable on the LAN with no SSH or MCP handshake, and —
decisively — its `/api/sessions` payload carries each Session's `status`, which
Kiln's `toMcpSession` drops, so finished-only filtering is possible today
without changing Kiln.

## Consequences

- The agent depends on Kiln's HTTP server running on gym; the MCP path would
  have spawned its own process on demand. Both require LAN reachability, so SSH
  bought no extra access here.
- Kiln's future planning and summary surfaces (`/api/plans`, `/api/plans/drafts`,
  `/api/strategies`, `/api/exercises`) are also on this HTTP API, so the choice
  is not a dead end.
- If we later want finished/abandoned distinction *and* MCP alignment, the fix
  is to expose `status` from Kiln's MCP `list_sessions`; revisit this ADR then.
