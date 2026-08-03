# Strava downloader

The download half of Kiln's "download the runs from strava" ask — the
reverse of [`strava_uploader`](../strava_uploader/README.md) (Kiln issue
#172). Kiln stores downloaded Strava Activities standalone, never as a
Session (see kiln's
`docs/adr/0004-strava-inbox-as-standalone-activities.md`); this agent scrapes
the athlete's recent Activities from Strava's training log via
[Playwright](https://playwright.dev/python/) and records each one into
Kiln's inbox, so no paid Strava API dependency or OAuth token is needed. It
is meant to run periodically (`stengents strava-download`) rather than only
conversationally.

## Motivation

Kiln never calls Strava's API or holds an OAuth token — same reasoning
`strava_uploader` established, which keeps this agent clear of Strava's
AI-use/data-retention policy restrictions (see kiln's
`docs/adr/0004-strava-inbox-as-standalone-activities.md` and this repo's
`docs/adr/0007-strava-inbox-playwright-downloader.md`). Scraping and
recording are deliberately separable from what happens to an Activity next
(attaching it to a Session, generate-plan interpretation, ...) — none of
that is built yet; this agent's only job is getting Activities into Kiln.

## Data destination

Kiln exposes the inbox only over its LAN HTTP API — this agent never touches
Kiln's SQLite file directly, same as `strava_uploader` never touches
`strava-outbox/` on disk. `kiln_inbox_client.py` is the plain, model-free
client for it, mirroring `strava_uploader.kiln_outbox_client`'s style and its
`KILN_BASE_URL` convention (same env var, same default, same instance):

- `list_activities(status=...)` — `GET /api/strava-inbox?status=unattached|attached`
- `record_activity(activity)` — `POST /api/strava-inbox` (idempotent upsert by Strava's own activity id)
- `attach_activity(id, session_id)` — `POST /api/strava-inbox/:id/attach`

Recording is the only thing this agent does automatically; attaching an
Activity to a Session is a separate, explicit, user-confirmed step this
agent's tools don't call — Kiln itself never infers that association either.

## Terms

Kiln's own vocabulary applies: a scraped Strava run/ride/etc. is an
**Activity**, addressed by Strava's own activity id. "Unattached" means no
Kiln Session has been linked to it yet; "attached" means it has, via a
separate call this agent doesn't make on its own.

## What's validated today, and what is not

Built first against a local mock (no real Strava credentials were available
in this environment), then **confirmed against real production strava.com on
2026-08-03** — via a Chrome browser already authenticated by the maintainer
(the Chrome DevTools MCP extension, not this module's own Playwright; no
automated login exists here or in the sibling uploader). Real DOM was read
directly, this module's selectors were corrected to match, and three real
Activities (including the same `Morning Weight Training` Activity
`strava_uploader`'s own 2026-08-01 confirmation produced) were scraped,
parsed, and recorded into a running Kiln instance (`kiln-dev`, the scratch
Docker container — not production, since production hadn't been redeployed
with this route yet) via the real `POST /api/strava-inbox` HTTP call,
confirming the full pipeline end to end, including idempotent re-posting.

**What the real confirmation corrected** (the guessed shape from before
2026-08-03 was wrong on all three counts):

- The training log is a plain, server-rendered `<table class="... activities
  ...">`, not the `[data-testid="activity-entry"]` div structure originally
  guessed. Real selectors: `tr.training-activity-row` with `.col-type` /
  `.col-date` / `.col-title` / `.col-time` / `.col-dist` / `.col-elev` cells.
- Distance/elevation render as unit-formatted display text (e.g. `"3.53
  mi"`, `"101 ft"`) exactly as suspected — `strava_playwright.py` now parses
  and converts mi/km/ft/m to meters.
- **The title link's `href` is absolute** (`https://www.strava.com/activities/<id>`),
  not the relative `/activities/<id>` first assumed. A selector anchored on
  that prefix (`a[href^="/activities/"]`) would have silently matched
  nothing on the real page — this was caught only because of the real-site
  check, not by the mock (which had rendered a relative href, hiding the
  same bug). Fixed by selecting `.col-title a` directly and reading the
  href's last path segment regardless of absolute/relative form; the mock
  now renders an absolute href too, so this class of bug can't hide again.

**Two real gaps this confirmation turned up, not previously knowable**:

- **There is no average-heart-rate column on the training log at all.**
  `averageHeartRate` is always `None` from this scrape — getting it would
  need a separate page load per Activity (its detail page), not built here.
- **The date cell carries no time-of-day** (`"Sat, 8/1/2026"` only) — every
  recorded Activity's `date` is midnight UTC on that calendar day, coarser
  than a real FIT-derived Kiln Session date.

**Validated** (real Playwright, real Chromium, no mocking of Playwright
itself — see `tests/test_strava_playwright_downloader.py`):

- `mock_strava/` is a small stdlib-only (`http.server`) stand-in for
  Strava's training log, its default fixture now mirroring the **confirmed**
  real DOM shape (module docstring above): `GET /athlete/training` serves a
  `<table class="... activities ...">` of `tr.training-activity-row` rows,
  each with `.col-type` / `.col-date` / `.col-title` (an absolute-href link)
  / `.col-time` / `.col-dist` / `.col-elev` cells, no heart-rate column.
- `strava_playwright.list_recent_activities` drives a real headless Chromium
  through that page end to end and extracts each row into the dict shape
  `kiln_inbox_client.record_activity` expects, including parsing mi/km/ft/m
  units and both mm:ss/h:mm:ss duration formats.
- `sync.run_sync`'s orchestration (scrape -> record each Activity, one bad
  item never aborting the batch, a scrape failure reported as
  `scrape_error` rather than crashing) is proven against fakes for both
  `strava_playwright` and `kiln_inbox_client` in `tests/test_sync_downloader.py`.

**Confirmed against real `strava.com`** (2026-08-03, see the update above):
the route, table/row/cell selectors, the absolute-href id extraction, the
mi/ft unit parsing, and the mm:ss/h:mm:ss duration parsing were all checked
directly against real DOM and a real recorded round-trip into a running Kiln
instance — see the corrections list above for exactly what changed as a
result.

**Still NOT validated**:

- **Automated login for unattended runs.** The 2026-08-03 confirmation used
  a Chrome browser a human had already logged into by hand (via the browser
  extension, not this module's Playwright); nothing here can establish that
  session on its own yet for a real, unattended `stengents strava-download`
  run. `list_recent_activities` accepts an optional Playwright
  `storage_state` for a pre-authenticated *Playwright* session, but
  producing and refreshing that file is not automated — see
  `strava_uploader/README.md`'s "Steps to capture a reusable login session"
  section, which applies here unchanged (one shared login-session file could
  plausibly serve both agents, but that's an open question, not a decision
  made here).
- **Whether pagination/"load more" is needed for anything beyond the first
  20-row page** is confirmed to exist (the real page shows "1-20 of N" with
  next/prev controls) but not handled — out of scope for this version,
  mirroring kiln's own map issue #74, which punted a dedicated backfill
  experience the same way.
- **Metric-unit accounts (km/m display) are unconfirmed** — the account used
  for the 2026-08-03 check displays imperial (mi/ft); `_UNIT_TO_METERS`
  already has km/m entries ready, but no real km/m row has been seen.
- **Recording has only been proven against `kiln-dev`** (the scratch Docker
  container), not the production Kiln instance — production hadn't been
  redeployed with the `/api/strava-inbox` route at the time of this check.

## Configuration

- `KILN_BASE_URL` — Kiln's base URL, same variable and default
  (`http://192.168.40.161:4173`) as every other farm_system Kiln client.
- `STRAVA_DOWNLOAD_BASE_URL` — the scrape target's origin for `stengents
  strava-download`. Unset (default) targets real `https://www.strava.com`;
  point it at a local `mock_strava.serve()` instance to validate the flow
  with no real credentials at all.
- Model connection (`STENGENTS_MODEL_*`) is only used by the conversational
  `adk run` agent below — `stengents strava-download` never resolves a
  model.

## Run

### Conversationally

```bash
adk run src/farm_system/strava_downloader
```

Ask "what's unattached?" (calls `list_unattached`) or "download now" (calls
`download_now`, which runs the real `sync.run_sync()` — a real scrape
attempt, not a preview).

### Periodically (the intended production path)

```bash
stengents strava-download --interval 300
```

`--once` runs a single pass and exits (exit code `1` if that pass errored,
e.g. Kiln or Strava unreachable — a transient failure otherwise never kills
the loop, just gets reported and retried next interval). `--dry-run`
previews a full pass — skips both the real Playwright scrape and recording
anything into Kiln, so nothing is touched, mirroring
`strava_playwright.upload_fit`'s own `dry_run` shape (proves the loop's
machinery, not a preview of real content). `--limit` overrides how many
recent Activities to scrape per pass (default 10). This path is plain
Python + Playwright and never resolves or preflights a model.

### With no browser at all

```bash
python -c 'from farm_system.strava_downloader.kiln_inbox_client import list_activities; print(len(list_activities(status="unattached")))'
```
