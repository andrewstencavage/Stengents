# A run is a session of many turns, and turn capture is shared

The domain names three nested things, not two parallel ones:

```
Turn  ⊂  Session  ⊂  Run   ( Run = Session + a verified outcome )
```

A **Turn** is one request/response exchange with a single agent. A **Session**
is the set of Turns that share context. A **Run** is a Session that has been put
on trial — it carries the harness's deterministic verification verdict on top of
whatever Turns happened. A Run *is-a* Session; it does not sit beside the Turn as
a second, unrelated kind of trace. The only thing a Run adds over generic Session
logging is the verdict, and that verdict is the one part that is not reusable
across agents.

We decide two things here.

**1. A Run contains many Turns.** Today a fixture attempt is a single ADK
invocation, so `1 Run = 1 Session = 1 Turn`. That 1:1 coincidence is convenient
but it is not the model we build to. A future planning agent will decompose a
task and delegate to subagents in a graph pattern: the planner's Turn spawns
child Turns, each attributed to the agent that ran it and linked to the Turn that
delegated it. A Session is therefore a **tree of Turns** — the delegating Turn is
the parent, each subagent Turn is a child — carrying a causal order rather than a
flat linear one. (Shared subagent *results* are data references between Turns, not
extra parent edges; the capture tree stays single-parent.) A Run wraps that whole
tree plus one verdict.

**2. Turn capture is a shared utility; the Run is a harness specialization that
consumes it.** The generic substrate — bounding logged payloads (`reduce_value`),
turning ADK tool/model callbacks into a structured Turn record, deriving a Turn's
*operational outcome* (`completed` / `degraded` / `errored` / `no_answer`), and
grouping Turns into a Session — belongs in an agent-utilities library any agent
can use. The **Run record** is not part of that library: it adds fixture and
harness identity, artifacts, budgets, and the *verification outcome*
(`passed` / `failed` / `harness_failed`, per ADR-0002) by embedding a captured
Session, not by re-implementing capture.

The two outcomes stay distinct and are computed at different levels. Operational
outcome is per-Turn and asks "did the exchange run cleanly?" The Run outcome is
per-Run and asks "did the verifier accept the result?" A Run's verdict is **not**
a reduction of its Turns' operational outcomes — a Session of cleanly-`completed`
Turns can still be a `failed` Run.

## Consequences

- A Turn gains identity and attribution: a `turn_id`, the `agent` that ran it,
  and a `parent_turn_id` (absent for the root Turn) naming the Turn that delegated
  to it. `kiln_coach`'s `TurnLogger` already brackets one Turn per invocation on
  `invocation_id`; extending it to record parent/agent is additive.
- The reusable seam to extract is the **Turn-capture + `reduce_value`** pair
  (review candidate #3), *not* `build_turn_record` and not a merged record
  builder. `build_turn_record` and `build_run_record` stay separate: the records
  are deliberately different concepts (see `CONTEXT.md`), and merging them is the
  forced shape we are choosing to avoid.
- The Run record's shape will change to embed the Session's Turns, replacing the
  three thin parallel traces the harness records today — `tool_events` (an
  Actions-level tool log with no args/results), `RunCapturePlugin`'s lifecycle
  phases, and `adk.tool_lifecycle_events`. That is a `run_record` schema bump when
  it lands; it is **not** done yet, and nothing here forces it before the planning
  agent needs it. Until then the current flat record stands.
- `kiln_coach` stays a single-Turn Session; the graph is a capability the model
  allows, not an obligation every agent takes on.
- Reconstructing "what happened, in order" from a Session is a topological walk of
  the Turn tree by `parent_turn_id` and start time, not an array index. Any
  viewer or tally must not assume a linear list.
