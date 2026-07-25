# Kiln coach

An ADK agent that reads your [Kiln](https://github.com/andrewstencavage/kiln)
training history. Its first capability pulls your finished workout Sessions; it
is meant to grow toward planning and Session summaries.

## Data source

Kiln runs on the home-gym box and serves its browser API over the LAN on
`0.0.0.0:4173`. This agent reads finished Sessions directly from that HTTP API
(`GET /api/sessions`) — no SSH and no MCP handshake. See
[ADR 0001](../../../docs/adr/0001-kiln-access-via-http.md) for why HTTP rather
than Kiln's MCP boundary.

`kiln_client.fetch_sessions` is a plain, model-free function: call it to dump
your history as data, fast and free. The agent wraps it as the `get_sessions`
tool.

## Terms

This agent uses Kiln's own vocabulary. In Kiln a **Workout** is an *assigned*
item in a Plan, while a **Session** is a *performed* run-through — finished or
abandoned. "Previous workouts" means finished **Sessions**; abandoned ones are
dropped as noise. Kiln retains abandoned Sessions in its store, but they carry
`status: "abandoned"`. See `kiln/CONTEXT.md` for the full glossary.

## Turn logging

Each request/response exchange (a **Turn**) is appended as one JSON line to
`.stengents/kiln_coach/turns.jsonl` (gitignored). A Turn record holds the query,
the tool calls it triggered (args verbatim, results summarized to scalars +
shapes), timing, the answer, and an **operational outcome** — `completed`,
`degraded` (an error the agent recovered from with an answer), `errored`, or
`no_answer` — capturing whether the exchange *ran cleanly*, not whether the
answer was correct (answer quality is a later, separate concern).

Capture is wired through the agent's callbacks in `agent.py` via the shared
[`stengents.utilities.turn_log.TurnLogger`](../../stengents/utilities/turn_log.py),
so it works under `adk run` and `adk web`.
Record-building and the result reducer are pure and unit-tested. To review
outcomes:

```bash
# tally operational outcomes
python -c "import json,collections,sys; print(collections.Counter(json.loads(l)['outcome'] for l in open('.stengents/kiln_coach/turns.jsonl')))"
# eyeball query -> answer for quality
jq -r '"\(.query)\n-> \(.answer)\n"' .stengents/kiln_coach/turns.jsonl
```

## Resolved: workout miscount + rambling (context window)

The agent used to answer "how many workouts" wrongly (reporting 7 distinct
workout *names*, or 1) and ramble at length. Root cause: Ollama's default
context window (`num_ctx` 4096) truncated the oldest tokens — the system
instruction — once the full workouts payload was in context, so the model lost
the "use workout_count / be brief" guidance.

Fix: run a model with a larger context. On gym, `qwen2.5:7b-8k` (a
`num_ctx 8192` variant of `qwen2.5:7b`, created via a Modelfile
`PARAMETER num_ctx 8192`) answers correctly and concisely — e.g. "You have
logged 16 workouts. Your most recent one was … Upper B — Volume." Set
`STENGENTS_MODEL_NAME=qwen2.5:7b-8k` to use it (or leave it; it is the agent's
default model name).

## Configuration

- `KILN_BASE_URL` — Kiln's base URL. Defaults to `http://192.168.40.161:4173`.
- Model connection resolves through `stengents.utilities.model_source`: `STENGENTS_MODEL_*`,
  falling back to the local `gym` defaults. The model still runs loopback-only on
  `gym`, so its tunnel to `11434` must be up.

## Run

Activate the project venv once (`source .venv/bin/activate`); its editable
install already puts `src/` on the path, so no `PYTHONPATH` is needed.

With the model tunnel running and Kiln reachable on the LAN:

```bash
adk run src/farm_system/kiln_coach
```

`adk run` takes the agent's folder path and adds its parent to `sys.path`, so
the real path works directly.

To pull sessions with no model at all:

```bash
python -c 'from farm_system.kiln_coach.kiln_client import fetch_sessions; print(len(fetch_sessions()))'
```
