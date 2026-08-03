# Farm system

Use this directory for exploratory agents and small experiments that are not
part of the reproducible coding-agent harness.

Keep evaluated fixture logic in the harness and fixture directories. Promote an
experiment here only after it has a defined interface, tests, and a clear role
in the product.

Each agent owns a directory under `src/farm_system/`. Put its executable code,
prompts, assets, and agent-specific notes there rather than sharing a flat
module namespace.

## Hello world

With the project venv active and the `gym` tunnel running, invoke the
standalone agent with:

```bash
adk run src/farm_system/hello_world
```

`hello_world/agent.py` exports the standard ADK `root_agent`. It resolves the
development-time model source through `stengents.utilities.model_source`, reading
`STENGENTS_MODEL_BASE_URL`, `STENGENTS_MODEL_NAME`, and
`STENGENTS_MODEL_API_KEY` when present, or the local `gym` defaults.

## Kiln coach

An agent that reads your Kiln training history over the LAN HTTP API and will
grow toward planning and Session summaries. See
[`kiln_coach/README.md`](kiln_coach/README.md).

## Strava uploader

Drains Kiln's Strava upload outbox — a FIT file queued per finished Session —
by driving a real browser through Strava's upload form via Playwright, so no
paid Strava API dependency is needed. Meant to run periodically
(`stengents strava-sync`), not just conversationally; validated today against
a local mock Strava server, since real credentials aren't available in this
environment. See [`strava_uploader/README.md`](strava_uploader/README.md).

## Strava downloader

The reverse of the uploader above: scrapes the athlete's recent Activities
from Strava's training log via Playwright and records each one into Kiln's
Strava inbox (Kiln stores them standalone, never as a Session). Meant to run
periodically (`stengents strava-download`), not just conversationally;
validated today against a local mock, since — like the uploader before its
own real-site confirmation — nothing here has run against real strava.com
yet. See [`strava_downloader/README.md`](strava_downloader/README.md).
