# Strava downloader lives in farm_system, writes Kiln's inbox over HTTP, and is validated against a local mock, not real Strava

Kiln's "download the runs from strava" ask splits into two halves the same
way issue #172 did, just reversed: Kiln owns the Strava inbox — a standalone
`strava_activities` table, exposed only over HTTP (see kiln's
`docs/adr/0004-strava-inbox-as-standalone-activities.md`); this repo owns
filling that inbox by scraping Activities from Strava's training log. This
ADR is the Stengents half's shape, and deliberately mirrors ADR-0006 (the
uploader's own ADR) wherever the same reasoning applies.

## Decision

- **Placement: `src/farm_system/strava_downloader/`, the same pattern as
  `strava_uploader` and `kiln_coach`.** `agent.py` exports
  `root_agent = LlmAgent(...)` for the conversational case ("what's
  unattached", "download now"); the deterministic work is plain functions in
  sibling modules the agent only wraps — `sync.py`, `strava_playwright.py`,
  `kiln_inbox_client.py` — matching `farm_system`'s "each agent owns a
  directory" convention.
- **`kiln_inbox_client.py` reads/writes Kiln over plain HTTP
  (`urllib.request`), not MCP** — the same choice ADR-0001 and ADR-0006 both
  made for their own Kiln clients, for the same reason: kiln's own
  `docs/adr/0004-strava-inbox-as-standalone-activities.md` independently
  settles this from Kiln's side (the inbox is HTTP-only so a
  possibly-remote, independently-swappable downloader never needs a shared
  filesystem or Kiln's stdio MCP process running alongside it).
- **The periodic drain (`stengents strava-download`) is deterministic, no
  model in the loop** — the same shape `strava-sync` established. `sync.run_sync`
  never resolves a model and the CLI command never preflights one.
- **No real Strava access in this environment**, same constraint ADR-0006
  hit. `strava_playwright.list_recent_activities`'s scrape target
  (`/athlete/training`, a `[data-testid="activity-entry"]` block per
  Activity with nested `data-stat="..."` spans for the numeric fields) is
  written from a guess at Strava's likely training-log shape, not confirmed
  against the real page, and is marked `UNVERIFIED` in code — unlike the
  uploader, which has since been confirmed (2026-08-01) against real
  strava.com, **nothing in this module has run against the real site at
  all**. Validation instead runs end to end — a real Playwright/Chromium
  browser, no mocking of Playwright itself — against `mock_strava/`, a small
  stdlib-only local server mirroring the same guessed shape. See
  `farm_system/strava_downloader/README.md`'s "What's validated" section for
  the exact gap and the single likeliest thing to be wrong (real Strava
  almost certainly renders distance/time/elevation as localized,
  unit-formatted text rather than raw meters/seconds in a data attribute).
- **No per-item failure report goes back to Kiln, unlike the uploader.** The
  uploader's outbox tracks retry state Kiln owns (`attempts`/`lastError`),
  so a failed upload has somewhere to report to. A failed scrape or a failed
  `record_activity` call has nothing to report *to* — Kiln was never told
  about that Activity — so it just stays unrecorded until the next periodic
  pass tries again. `record_activity` is an idempotent upsert keyed by
  Strava's own activity id, so re-scraping and re-posting the same recent
  window every pass needs no dedup logic on either side.
- **One bad item never aborts a batch**, and **a bad pass never aborts the
  periodic loop** — the same two-level "best-effort, never blocks"
  philosophy ADR-0006 established, extended here: `sync.run_sync` catches
  any exception per Activity (a malformed record, a failed POST) and folds
  it into that item's own `failed` outcome; a scrape failure (Strava/network
  trouble) is reported as a top-level `scrape_error` rather than raised; and
  `cli.py`'s periodic loop catches any exception from a whole pass, reports
  it, and continues to the next interval.

## Update, 2026-08-03: confirmed against real strava.com, not just a mock

The maintainer checked the guessed shape above against real, authenticated
`strava.com` (via an already-logged-in Chrome browser, not this module's own
Playwright — see the package README's "What's validated" section for the
full detail) and it was wrong on every count:

- The training log is a plain server-rendered `<table>`
  (`tr.training-activity-row` rows), not the JS-driven
  `[data-testid="activity-entry"]` div structure first guessed.
- Distance/elevation are unit-formatted display text (`"3.53 mi"`), as
  suspected — now parsed and converted to meters.
- **The title link's `href` is absolute**
  (`https://www.strava.com/activities/<id>`), not the relative
  `/activities/<id>` assumed. A selector anchored on that prefix would have
  silently matched zero rows on the real page — caught only by checking the
  real site, since the mock had (also wrongly) rendered a relative href and
  so could not have caught its own mismatch. Both the scraper and the mock
  were fixed together.
- Two things a guess couldn't have known either way: there is no
  average-heart-rate column on this page at all (always `None` from this
  source), and the date cell carries no time-of-day (every recorded
  Activity's `date` is midnight UTC on that calendar day).

Three real Activities (the same `Morning Weight Training` Activity the
uploader's own 2026-08-01 confirmation produced, plus two real runs) were
scraped, parsed, and recorded end to end into a running Kiln instance via the
real `POST /api/strava-inbox` call — including confirming the idempotent
upsert (re-posting the same id left the inbox at the same three entries).
That instance was `kiln-dev` (the scratch Docker container this repo's
sibling Kiln project defines specifically for this kind of check), not
production — production Kiln hadn't been redeployed with the
`/api/strava-inbox` route yet at check time.

## Consequences

- The downloader depends on Kiln's HTTP server being reachable
  (`KILN_BASE_URL`, same variable and default as every other farm_system
  Kiln client) and on a Playwright Chromium binary being installed —
  identical setup requirement to `strava_uploader`, no new dependency
  (`playwright` is already a `pyproject.toml` dependency).
- Attaching a downloaded Activity to a Kiln Session
  (`kiln_inbox_client.attach_activity`) exists as a client function but is
  not called by anything in this module yet — `agent.py`'s tools only list
  and download. Automatic Workout-match suggestion and any UI for confirming
  an attachment are both out of scope here, matching kiln's own map issue
  #74's explicit "Kiln may suggest, user must confirm" boundary and its
  "not yet specified" list.
- `mock_strava/`'s route and DOM shape (`/athlete/training`,
  `tr.training-activity-row`) is confirmed against the real page as of
  2026-08-03 (see the update above), not merely a guess anymore — though
  metric-unit (km/m) accounts and anything past the first page remain
  unconfirmed, and Strava's own markup can of course still change later
  without notice. `mock_strava/`'s route shape has no source-of-truth
  guarantee beyond that one checked account/session.
- Login/session handling for unattended runs is unsolved here too, same as
  the uploader — `list_recent_activities` accepts an optional Playwright
  `storage_state` but nothing produces or refreshes one. The 2026-08-03
  confirmation used a human-authenticated Chrome session (via the browser
  extension), not a captured Playwright `storage_state`, so unattended
  `stengents strava-download` against production remains exactly as
  unproven as it was before this update.
