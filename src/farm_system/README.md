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

With the `gym` tunnel running, invoke the standalone agent with:

```bash
PYTHONPATH=src adk run farm_system/hello_world
```

`hello_world/agent.py` exports the standard ADK `root_agent`. It uses
`STENGENTS_MODEL_BASE_URL`, `STENGENTS_MODEL_NAME`, and
`STENGENTS_MODEL_API_KEY` when present, or the local `gym` defaults. Prefix the
same settings with `FARM_SYSTEM_` to override only farm-system agents.
