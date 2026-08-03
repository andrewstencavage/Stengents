# Strava uploader

The uploader half of [Kiln](https://github.com/andrewstencavage/kiln) issue
#172: Kiln generates a FIT Activity file for every finished Session and queues
it in a local outbox; this agent drains that outbox by driving a real browser
through Strava's upload form via [Playwright](https://playwright.dev/python/),
so no paid Strava API dependency is needed. It is meant to run periodically
(`stengents strava-sync`) rather than only conversationally.

## Motivation

Kiln owns FIT generation and the outbox as its own local state; uploading is a
deliberately separable, independently swappable concern (see kiln's
`docs/adr/0003-strava-outbox-via-http.md`) — today a Playwright uploader,
later possibly a real Strava API client, with FIT generation untouched either
way.

## Data source

Kiln exposes the outbox only over its LAN HTTP API — the uploader never
touches `strava-outbox/` on disk directly, same as `kiln_coach` never touches
`kiln.sqlite`. `kiln_outbox_client.py` is the plain, model-free client for it,
mirroring `kiln_coach/kiln_client.py`'s style and its `KILN_BASE_URL`
convention (same env var, same default, same instance):

- `list_pending()` — `GET /api/strava-outbox?status=pending`
- `fetch_fit(session_id)` — `GET /api/strava-outbox/:sessionId/fit` (raw FIT bytes)
- `mark_uploaded(session_id, strava_url=...)` — `POST .../uploaded`
- `mark_failed(session_id, error)` — `POST .../failed`

Kiln never decides when to give up retrying a failed upload — `mark_failed`
only increments `attempts`/`lastError` and leaves the entry `pending`, FIT
file untouched. That give-up policy lives here, in `sync.py`'s `max_attempts`.

## Terms

Kiln's own vocabulary applies: a queued FIT file is an **outbox entry**,
addressed by the Kiln **Session** `id` it was generated from. "Uploaded" means
Kiln has archived the FIT file and recorded a `stravaUrl`; "pending" (still
below `max_attempts`) or "skipped at max attempts" are the only other states —
Kiln's outbox never deletes an entry on failure.

## What's validated today, and what is not

Real Strava credentials weren't available while this was first built, so it
was validated against a local mock. **Update, 2026-08-01: the maintainer then
logged into Strava and confirmed the real thing directly** — a FIT file
generated from a real finished Kiln Session (`Upper B — Volume`, 46 minutes)
was uploaded through an already-authenticated browser to
`strava.com/upload/select`. Result:
[strava.com/activities/19558762653](https://www.strava.com/activities/19558762653) —
Strava correctly read the FIT file's sport/sub_sport as a strength workout,
auto-titled it "Morning Weight Training", and showed the right elapsed time
(46:00) and date. The selectors in `strava_playwright.py` (file input, title
field, save button, `/activities/<id>` redirect) are confirmed correct
as-written — no changes were needed. The `UNVERIFIED` markers in that file
have been replaced with confirmation notes.

One real bug turned up and was fixed the same day: the uploaded activity's
displayed start time was off by the local UTC offset (Strava treats a FIT
`timestamp` field as already-local, not true UTC, same as most consumer
devices) — Kiln's `fit-encoder.js` now encodes local wall-clock time instead
of a true UTC instant; see kiln's `docs/adr/0003-strava-outbox-via-http.md`
and that commit's `toFitTimestamp` for the fix.

**Still open**: that confirmation used a browser a human had already logged
into — nothing here can establish a Strava session on its own yet. See "Exact
steps" below, which is now about capturing a reusable session for *unattended*
runs, not about verifying selectors (that part's done).

**Validated** (real Playwright, real Chromium, no mocking of Playwright
itself — see `tests/test_strava_playwright.py` — plus the real-strava.com
confirmation above):

- `mock_strava/` is a small stdlib-only (`http.server`) stand-in for Strava's
  upload flow: `GET /upload/select` serves an upload form, `POST` to the same
  path (multipart file + activity name) either redirects to a confirmation
  page at `/activities/<id>` (success) or re-renders the form with an error
  banner for a too-small/garbage upload (the deliberate failure path).
- `strava_playwright.upload_fit` drives a real headless Chromium through that
  flow end to end: file input, activity-name field, save button, and detects
  success by waiting for navigation to `/activities/<id>`.
- The forced-failure path (`test_upload_fit_captures_a_screenshot_on_a_forced_failure`)
  confirms a failed upload leaves a screenshot behind under `screenshot_dir`
  for troubleshooting — the diagnostics acceptance criterion issue #172 calls
  for.
- `sync.run_sync`'s orchestration (pending -> uploaded, a failure incrementing
  `attempts` and staying pending, and the `max_attempts` skip) is proven
  against fakes for both `kiln_outbox_client` and `upload_fit` in
  `tests/test_sync.py` — the orchestration logic, independent of what's on
  the other end of `base_url`.

**Confirmed against real `strava.com`** (2026-08-01, see the update above for
the full evidence): the file input, `input[name="name"]` title field,
`button[type="submit"]` save button, and the `/activities/<id>` post-save
redirect all match the real page exactly as `strava_playwright.py` already
had them written. Strava also renders the Title/Save form asynchronously
after the file finishes processing (no page navigation) — Playwright's
auto-waiting `.fill()`/`.click()` already handled that correctly with no
extra wait needed.

**NOT validated** — automated login:

- **Login/session handling for unattended runs is entirely unsolved.** The
  2026-08-01 confirmation used a browser a human had already logged into by
  hand; nothing here can establish that session on its own. `upload_fit`
  accepts an optional Playwright `storage_state` (a saved cookies/local-storage
  JSON) for a pre-authenticated session, but producing that file — a one-time
  manual login, saved via `context.storage_state(path=...)` — is not automated
  and needs a decision from the maintainer (a fixed path? refreshed how often?
  stored where?) before `stengents strava-sync` can run unattended against
  production.

### Steps to capture a reusable login session for unattended runs

1. **Capture a login session once, by hand**: run a small script that
   launches a non-headless `playwright.chromium.launch(headless=False)`,
   navigates to `https://www.strava.com/login`, log in manually in the opened
   window, then call `context.storage_state(path="strava-storage-state.json")`
   and close the browser. Keep that file out of git (it is a live session
   credential).
2. Point the uploader at the real site (the default — `STRAVA_UPLOAD_BASE_URL`
   unset) and pass that `storage_state` path through to `upload_fit` (today
   this requires a one-line call, not a CLI flag — see the "still open" note
   below).
3. **Still open**: `sync.py`/`cli.py`'s `strava-sync` command has no flag yet
   to pass a `storage_state` path through to `upload_fit` — that plumbing
   (plus a decision on where the file lives and how it's refreshed once
   Strava's session eventually expires) is the concrete next step before
   `stengents strava-sync` can run unattended against production.

## Configuration

- `KILN_BASE_URL` — Kiln's base URL, same variable and default
  (`http://192.168.40.161:4173`) as `kiln_coach`.
- `STRAVA_UPLOAD_BASE_URL` — the upload target's origin for `stengents
  strava-sync`. Unset (default) targets real `https://www.strava.com`; point
  it at a local `mock_strava.serve()` instance to validate the flow with no
  real credentials at all.
- Model connection (`STENGENTS_MODEL_*`) is only used by the conversational
  `adk run` agent below — `stengents strava-sync` never resolves a model.

## Run

### Conversationally

```bash
adk run src/farm_system/strava_uploader
```

Ask "what's pending?" (calls `list_pending`) or "sync now" (calls `sync_now`,
which runs the real `sync.run_sync()` — a real upload attempt, not a preview).

### Periodically (the intended production path)

```bash
stengents strava-sync --interval 300
```

`--once` runs a single pass and exits (exit code `1` if that pass errored,
e.g. Kiln unreachable — a transient failure otherwise never kills the loop,
just gets reported and retried next interval). `--dry-run` previews a full
pass — still lists Kiln's pending outbox, but skips both the real Playwright
upload and reporting anything back to Kiln, so nothing is touched. `--max-attempts`
overrides the default give-up threshold (`5`). This path is plain Python +
Playwright and never resolves or preflights a model.

### With no browser at all

```bash
python -c 'from farm_system.strava_uploader.kiln_outbox_client import list_pending; print(len(list_pending()))'
```
