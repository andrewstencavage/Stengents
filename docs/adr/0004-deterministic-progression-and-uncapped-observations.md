# Progression and selection are computed, not model-inferred; the observation cap and `summary` are dropped

The original #17 contract asked one model call (`synthesize`) to do three jobs
at once: find which of the extracted findings were worth keeping, pick the
best three, and write a free-text `summary` of the whole session. In practice,
real reviews generated against live Kiln history showed this doesn't produce
what the review is actually for. Across three real sessions it produced nine
observations, all nine `category: performance` — plain restatement of numbers
already visible in the Train console — and zero `progression` or `adherence`
observations, despite comparison history existing for some of the exercises
involved. The one time the model did attempt a cross-session comparison, it
confused units (compared a load in plates against a duration in seconds as if
they were the same quantity) — an invented-comparison failure of exactly the
kind the contract's other machinery (evidence grounding, the
`unsupported_claim_rate == 0.0` safety floor) exists to prevent, just in a
place nothing was checking.

## Decision

- **Progression is computed, not inferred.** A per-exercise **top-set delta**
  compares the subject Session's top set (highest `load`, ties broken by more
  `reps`) against the single most recent prior Session's top set for that
  exact exercise: report the first of `load`, then set count, then `reps`
  that differs; an exact match is itself reported as "held steady." No
  multi-session trend/plateau detection yet — one prior comparison point,
  deliberately. The model's only role here is phrasing a computed fact into
  one sentence, never the comparison itself.
- **A new deterministic signal, the Plan streak**, feeds an `adherence`
  observation: consecutive fully-elapsed calendar weeks where every non-rest
  scheduled Workout in that week's Plan had at least one finished Session
  (same semantics as Kiln's existing `doneWorkoutNames` — no day-level
  deadline, abandoned Sessions don't count). A week with no active Plan
  breaks the streak.
- **Selection is deterministic and uncapped.** Every exercise touched gets a
  progression line (including "held steady" ties); the streak line is added
  when active. Nothing is trimmed for length or variety.
- **The `observations` cap (`max_length=3`) is removed from the contract.**
  `WorkoutReview.observations` is no longer schema-capped.
- **`summary: str` is removed from the contract entirely** — it was pure
  filler once the model-selected-and-narrated design went away, and it was
  never displayed by Kiln's UI to begin with.
- **Extraction is untouched.** Data-quality and note-based findings
  (skipped/malformed sets, etc.) still come from the existing per-exercise
  `extract()` model calls; this decision only changes how progression is
  found and how the final observation set is assembled, not the qualitative
  side of the pipeline.
- Negative or flat progression is reported as found, never softened — the
  accountability signal comes from the separate, genuinely evidence-backed
  streak observation, not from reframing bad news.

## Consequences

- This is a breaking change to the #17 contract. `evaluator.py`'s always-on
  assertion of `≤3 observations` and any benchmark `expectations.json` that
  assumes the old cap or a populated `summary` need updating; the corpus
  should be re-run and re-baselined, not compared against the old run
  directly, once this lands (see the `tune-capability`/`benchmark-review`
  skills).
- `review.py`'s `synthesize` step shrinks: it no longer selects among
  candidates or writes a summary, only phrases the deterministically chosen
  progression/streak facts. Selection logic (ranking, capping) moves out of
  the model prompt and into plain code, testable without a model call.
- The Plan streak computation needs Kiln Plan/Workout data, not just Session
  history — `kiln_client` may need a new fetch (Plan-per-week, scheduled
  Workout names) if it doesn't already expose what `doneWorkoutNames`-equivalent
  logic requires on the Stengents side.
- `summary` is easy to reintroduce later (it's a schema addition, not a
  removal of anything else depended on) if a future consumer wants a one-line
  recap; not restoring it preemptively is deliberate given nothing reads it
  today.
