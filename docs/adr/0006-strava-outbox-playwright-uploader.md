# Strava uploader lives in farm_system, reads the outbox over HTTP, and is validated against a local mock, not real Strava

Kiln issue #172 splits into two halves: Kiln owns FIT generation and a local
outbox (`strava-outbox/`, exposed only over HTTP — see kiln's
`docs/adr/0003-strava-outbox-via-http.md`); this repo owns draining that
outbox by uploading each queued FIT file to Strava. This ADR is the Stengents
half's shape.

## Decision

- **Placement: `src/farm_system/strava_uploader/`, the same pattern as
  `kiln_coach`.** `agent.py` exports `root_agent = LlmAgent(...)` for the
  conversational case ("what's pending", "sync now"); the deterministic work
  is plain functions in sibling modules the agent only wraps, mirroring how
  `auto_replan`'s pure decision logic lives outside `kiln_coach`'s chat agent
  entirely. Unlike `auto_replan` (which sits under `src/stengents/` since it
  has no standalone chat surface), the uploader keeps a conversational agent
  too, so it stays in `farm_system` alongside it — one directory, one owner,
  matching this repo's "each agent owns a directory under `src/farm_system/`"
  convention (`farm_system/README.md`).
- **`kiln_outbox_client.py` reads Kiln over plain HTTP (`urllib.request`),
  not MCP** — the same choice ADR-0001 made for `kiln_coach.kiln_client`, and
  explicitly *not* superseded by ADR-0005 (which only moved the coach
  *server*'s planning writes to MCP; the chat agent's own HTTP read path was
  left untouched by that decision). Kiln's own
  `docs/adr/0003-strava-outbox-via-http.md` independently settles the same
  question from Kiln's side: the outbox is exposed over HTTP specifically so
  a possibly-remote, independently-swappable uploader never needs a shared
  filesystem or Kiln's stdio MCP process running alongside it. Both repos'
  ADRs agree; this one is the Stengents-side record of the same boundary.
- **The periodic drain (`stengents strava-sync`) is deterministic, no model
  in the loop** — the same shape `auto_replan`/`_serve_coach_command`
  established for triggered, non-chat logic. `sync.run_sync` never resolves a
  model and the CLI command never preflights one, unlike every other
  subcommand in `cli.py`.
- **No real Strava access in this environment.** Real credentials aren't
  available here, so `strava_playwright.upload_fit`'s selectors
  (`strava.com/upload/select`'s file input, activity-name field, save button,
  and the assumption that a successful save redirects to
  `/activities/<id>`) are written from current best knowledge of Strava's
  upload flow, not confirmed against the real page, and are marked
  `UNVERIFIED` in code. Validation instead runs end to end — a real
  Playwright/Chromium browser, no mocking of Playwright itself — against
  `mock_strava/`, a small stdlib-only local server built specifically to
  exercise both the success and (deliberately forced) failure paths. See
  `farm_system/strava_uploader/README.md`'s "What's validated" section for
  the exact gap and the steps a maintainer needs to run once, with real
  credentials, to confirm or fix the selectors — including the still-open
  question of how a login session gets established for the real site at all
  (nothing here automates login; `upload_fit` accepts an optional Playwright
  `storage_state` for a pre-captured session, but producing and refreshing
  that file is not yet wired up).
- **A failed upload leaves a screenshot behind** (`screenshot_dir`, default
  `.stengents/strava_uploader/screenshots/`) — the diagnostics acceptance
  criterion issue #172 calls out explicitly, and the reason
  `test_strava_playwright.py` exercises a forced-failure path (a too-small
  "garbage FIT" upload) rather than only the happy path.
- **Kiln's `attempts`/`lastError` is the give-up signal, and `sync.py` owns
  the give-up policy** (`max_attempts`, default 5) — matching Kiln's own
  stated boundary ("Kiln never decides when to give up retrying") in both its
  ADR and `CLAUDE.md`. An entry at `max_attempts` is skipped untouched, not
  deleted or hidden: its FIT file stays in Kiln's outbox exactly as it was,
  so `attempts` remains a maintainer-visible "this one needs a look," not
  something silently swallowed.
- **One bad upload never aborts a batch.** `sync.run_sync` catches any
  exception per outbox entry (a fetch failure, an upload failure, even a
  failed report back to Kiln) and folds it into that entry's own `failed`
  result — mirroring the "best-effort, never blocks" philosophy Kiln's own
  coach-review and Strava-outbox integrations already use for their
  post-Session hooks (per kiln's `AGENTS.md`/`CLAUDE.md`). The periodic CLI
  loop extends the same philosophy one level up: a failed *pass* (e.g. Kiln
  itself unreachable) is reported and the loop continues to the next
  interval, rather than crashing the whole periodic process.

## Consequences

- The uploader depends on Kiln's HTTP server being reachable
  (`KILN_BASE_URL`, same variable and default as `kiln_coach`) and on a
  Playwright Chromium binary being installed in whatever environment runs
  `stengents strava-sync` — `python -m playwright install chromium` is a
  one-time setup step, not handled by `pip install -e .` alone.
  `playwright` is now a `pyproject.toml` dependency.
- **Update, 2026-08-01**: real-site selectors are now confirmed, not just
  believed — the maintainer uploaded a real FIT file (from a real finished
  Kiln Session) through an already-logged-in browser and it landed correctly
  at `strava.com/activities/19558762653`, auto-titled and auto-categorized as
  a strength workout by Strava itself, with no selector changes needed. What
  remains unverified is automated login/session establishment for *unattended*
  `stengents strava-sync` runs — that confirmation used a human-authenticated
  browser, not a Playwright `storage_state`. See
  `farm_system/strava_uploader/README.md`'s "What's validated" section.
  Kiln's `fit-encoder.js` also picked up a real fix from this: FIT timestamps
  now encode local wall-clock time rather than true UTC, since Strava (like
  most consumer devices) doesn't convert on display — the 2026-08-01 test
  activity's start time was off by the UTC offset until that fix landed.
- `mock_strava/`'s route shape (`/upload/select`, `/activities/<id>`) is a
  best-effort mirror of Strava's believed flow, not a spec pulled from
  Strava's own documentation (Strava publishes no such spec for its web
  upload form). If the real selectors turn out to need a materially
  different flow (e.g. a JS-driven single-page upload with no full-page
  navigation, rather than a plain form POST that redirects), both
  `mock_strava/` and `strava_playwright.upload_fit` need revisiting together,
  not just the selectors.
