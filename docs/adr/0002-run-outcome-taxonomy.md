# A run fails as the harness only when the harness itself failed

A coding-fixture run terminates in exactly one of three outcomes, and the
boundary between them is a deliberate design rule, not an accident of where
exceptions happen to land:

- **`0` — passed.** The agent was given a fair attempt and the fixture's
  deterministic verifier accepted the result.
- **`1` — failed.** The agent was given a fair attempt and the verifier
  rejected the result. Every way the *model under test* can fall short lives
  here: a wrong patch, no tool calls, a rejected out-of-fixture path, or an
  exhausted run budget. These are verdicts *about the model*.
- **`2` — harness failed.** The harness could not render a fair verdict at all,
  because something on *our* side broke: the model endpoint dropped mid-run, an
  ADK internal error, a bug in harness code, or an I/O failure writing the
  record. This is a verdict *about the harness*.

The rule, stated as an invariant:

> A run is recorded as `harness_failed` **if and only if** the harness itself
> failed to do its job — never because the agent under test did something the
> harness is designed to handle.

Anything the agent can do — including hostile or incompetent things — must be
converted by the named actions into a recorded verdict (`passed`, `failed`, or
budget exhaustion), never allowed to crash the run. The reproducibility
boundary polices agent behaviour; it does not treat that behaviour as an
exception. Concretely, a named action that the agent misuses (an out-of-fixture
`read_file`, a write outside the source surface) returns a `rejected: …` string
to the agent as a normal tool result, so the misuse becomes both feedback the
agent can act on and a `failed` verdict — not a traceback and a false
`harness_failed`.

`harness_failed` is kept as a real, load-bearing signal rather than suppressed.
Swallowing it — catching every exception and calling the run `failed` — would
be a lie: a flaky endpoint would masquerade as "these models are bad," and an
operations problem would be chased as a model-quality problem. When
`harness_failed` fires it should be rare, alarming, and actionable: fix the
harness or the environment, do not read the run as a score of `2`.

## Consequences

- The named actions in `harness.py` are the enforcement point. Any action that
  accepts agent-supplied input must convert misuse into a returned rejection,
  never a raised exception; `read_file` and `write_source_file` both do this.
  A new action that lets an exception escape reintroduces the false
  `harness_failed` this ADR exists to prevent.
- The catch-all `except Exception -> harness_failed` in `run_fixture` is a
  backstop for genuinely unhandled harness faults, not the primary mechanism.
  If it ever fires for an agent behaviour, that behaviour belongs in a named
  action's rejection path instead.
- The backstop is still slightly too broad in one direction: it does not
  distinguish an *environment* failure (a mid-run endpoint drop, e.g.
  `ConnectionError`) from a *harness-code* bug. Both are correctly kept out of
  the model's score, but they have different fixes. If run-level triage later
  needs that split, add a distinct outcome rather than widening `failed`.
- Because a rejected action returns normally, it records `outcome: "ok"` in the
  tool events, so a rejected `read_file` currently satisfies the
  `list_files -> read_file` discovery gate. If the gate should require a
  *successful* read, introduce a third tool-event outcome (`rejected`) distinct
  from both `ok` and `error`, and count only `ok` toward discovery.
