"""The evidence-backed Workout Review capability over Kiln training history.

Public surface: the ``review_workout`` entry point and the locked #17 output
contract types.
"""

# The version of the review capability itself — the prompt/generation/decode
# behaviour, distinct from the model and from the output contract. Stamped onto
# every benchmark artifact so a recorded score is tied to the capability that
# produced it. Bump on a meaningful capability change (e.g. a prompt revision or
# the revise-once loop landing).
#
# 0.2.0 — offer the model only groundable evidence candidates: drop the derived
#   ``sets_completed`` candidate, render fact values via the shared ``as_text``,
#   and guard decode against ungroundable citations. Changes the candidate set,
#   so its benchmark run is a distinct baseline from 0.1.0's.
# 0.3.0 — nudge the generator to name data-quality limitations: the prompt now
#   directs the model to inspect each exercise for skipped/missing/conflicting/
#   malformed data and emit a matching ``Limitation`` naming that exercise (#39).
#   A prompt-only change, so a distinct baseline from 0.2.0's. On a clean
#   qwen2.5:7b-8k A/B it lifts ``required_limitation_recall`` 0.333 -> 0.667
#   (c06, c10) and ``required_detail_recall`` 0.528 -> 0.625 with the safety
#   floors held, taking the live gate FAIL -> PASS. c05 (skipped Tricep Pushdown)
#   and c11 (malformed sets invisible to the model, #41) remain unrecalled.
# 0.4.0 — surface the malformed-set signal to the generator (#41): sets that
#   fail to build a valid ``SetEvidence`` are silently dropped by ``_set_rows``,
#   so a prompt-only nudge could never name them. ``_build_prompt`` now emits a
#   per-exercise DATA QUALITY NOTES section counting the dropped sets, plus an
#   inline instruction to raise ``malformed_data`` naming each listed exercise.
#   Both are injected ONLY when a session has dropped sets, so every clean-case
#   prompt stays byte-identical to 0.3.0 (no collateral perturbation). On a warm
#   qwen2.5:7b-8k A/B (corpus 7e40ed6e) c11 (malformed Chest Fly) now recalls its
#   limitation, lifting ``required_limitation_recall`` 0.667 -> 0.833 with both
#   safety floors held and the gate PASS. c05 (visible skipped signal) still
#   misses — a nudge-strength gap, not a visibility gap.
# 0.5.0 — cap total history evidence (no ticket filed yet). HISTORY_CAP bounds
#   comparison depth per exercise but never the total across a session; a real
#   session with several exercises at full history depth reached ~95 candidate
#   rows and the model degraded to echoing the last row instead of the
#   summary/observations/limitations schema — confirmed NOT a context-length
#   failure (prompt_tokens=4098 of an 8192 window) but a lost-in-the-middle
#   failure from sheer homogeneous-row volume; a larger-context model
#   (deepseek-64k) reproduced the same failure at 433s, ruling out "bigger
#   model" as a fix. MAX_HISTORY_EVIDENCE round-robins one whole prior instance
#   at a time per exercise so no exercise is starved and no prior's sets are
#   split, capping the total rather than any single exercise's depth. Changes
#   the candidate set, so a distinct baseline from 0.4.0's. Live-verified first
#   (95 -> 69 candidates, real session recovers a well-formed review), then a
#   warm qwen2.5:7b-8k A/B (corpus 7e40ed6e) holds both safety floors and
#   schema_valid_rate at 1.0, and *lifts* required_detail_recall 0.618 -> 0.660
#   and cases_passing 4 -> 5, with required_limitation_recall unchanged at
#   0.833 and the gate PASS -> PASS. Expected, not surprising: the 12-case
#   corpus has no session large enough to hit MAX_HISTORY_EVIDENCE at all, so
#   this A/B cannot exercise the actual fix — it only confirms no regression on
#   corpus-scale sessions. The corpus having no case resembling the real
#   failure (many exercises, deep real history) is exactly why the original
#   regression shipped undetected, and remains true after this change; adding
#   such a case is unstarted follow-up work.
# A prompt nudge for within-session load/rep changes (an explicit instruction to
# compare an exercise's first performed set against its last) was tried and
# abandoned before 0.6.0 landed: it regressed required_detail_recall 0.660 ->
# 0.569 (below the 0.60 floor, gate PASS -> FAIL) on a warm qwen2.5:7b-8k A/B —
# competing with the model's limited 3-observation budget and attention for the
# things the corpus actually grades (progression, data-quality limitations), not
# a free addition the way 0.3.0/0.4.0's nudges were. Never committed; the
# within-session-progression gap it targeted remains open.
#
# 0.6.0 — group history evidence into whole-prior-instance slots instead of one
#   row per set (no ticket filed yet). 0.5.0 capped *how much* history evidence
#   reaches the prompt but not *how many visible rows* it costs — the same real
#   data (e.g. 8 prior instances x 3 sets) still cost 24 separate near-identical
#   JSON lines, which is the exact row-volume mechanism 0.5.0's own diagnosis
#   pinned as the failure cause. _history_candidate_groups groups each prior
#   instance's SetEvidence rows into one compact citable slot
#   (_render_candidate); citing it expands to all of that instance's real,
#   individually-grounded rows via _resolve_evidence, so contract.py and the
#   "every cited row resolves verbatim" guarantee are completely untouched —
#   only the number of chunks the model has to read shrinks, not the underlying
#   evidence or what gets output. MAX_HISTORY_EVIDENCE keeps the same real-row
#   budget as 0.5.0; this only changes how it's chunked. Changes the candidate
#   representation, so a distinct baseline from 0.5.0's.
#
#   Warm qwen2.5:7b-8k A/B (corpus 7e40ed6e): both safety floors and
#   schema_valid_rate hold at 1.0/1.0/0.0, required_limitation_recall is
#   unchanged at 0.833, required_detail_recall lifts 0.660 -> 0.722 (the
#   12-case corpus is small enough that grouping rarely changes anything for
#   it, so this is likely model-sampling variance rather than a real effect of
#   the change), cases_passing lifts 5 -> 6, and the gate holds PASS.
#   Live-verified on the same real rich-history session 0.5.0 was tested
#   against: visible candidate
#   rows drop 69 -> 53, and the model spontaneously reports a real
#   within-session load change ("Seated Row 4.0 -> 6.0 plates") as a
#   'performance' observation — the exact gap an explicit prompt nudge tried
#   and failed to close moments earlier (see the note above), recovered here as
#   a side effect of less visual clutter, not a targeted fix.
#
#   Two real gaps surfaced landing this, neither fixed here, both pre-existing
#   in kind even though the second is worse in degree because of this change:
#   (1) that same "4.0 -> 6.0" claim is itself an unsupported number — the only
#   evidence it cited was a note ("moved up to 6, feels good there"); "4.0"
#   appears in no cited row at all (real data: 5/6/6 plates). The deterministic
#   evaluator's unsupported_claim_rate does not, and by design cannot, catch
#   this: it only checks that cited rows resolve and that a 'progression' claim
#   has a prior-Session row, never whether the claim's prose matches what's
#   cited — CONTEXT.md documents that as deferred to a not-yet-built "later LLM
#   rubric". This run still scored unsupported_claim_rate 0.0. Not introduced by
#   0.6.0, but 0.6.0 does nothing to catch it either. (2) grouping means citing
#   one id can pull in a whole prior instance's rows at once; a review from the
#   same live session had one observation cite 26 evidence rows, mostly
#   irrelevant history from unrelated exercises/sessions, once the model cited
#   several group ids for one claim. Still "grounded" (every row resolves) so
#   no floor catches it, but a real citation-hygiene regression this change
#   introduces — nothing here caps evidence-per-observation. Shipped anyway per
#   product decision: the row-count win is real and measured, and both gaps are
#   about claim/citation quality, not what this pass was scoped to fix.
#
# 0.7.0 — decompose generation into per-exercise extraction + one synthesis
#   call (Wayfinder map #56, decision #59, implementation #60). Two independent
#   one-shot fixes for coverage (a written nudge, then a deterministic hint,
#   both tried against this same failure) each regressed the corpus by
#   displacing something the model used to get right elsewhere in the same
#   completion — evidence the one-shot completion itself, not the wording, was
#   the constraint. ``partition_session``/``extract``/``synthesize`` replace
#   the single ``_build_prompt``/one call/``_decode_observations`` path:
#   ``extract`` runs once per exercise partition (full attention, no
#   competition from other exercises) via ``_needs_extraction``, which skips a
#   call entirely for an exercise with nothing to report (no note, not skipped,
#   not malformed, unchanged within the session and vs. its most recent prior)
#   — the latency lever, since the local endpoint is compute-bound (measured:
#   8 concurrent calls barely beat 8 sequential ones), so fewer tokens
#   processed is the only thing that meaningfully helps. ``synthesize`` then
#   selects the best <=3 findings and writes the summary. Grounding is
#   completely unchanged: extraction and synthesis both cite into the same
#   shared, numbered candidate pool ``_candidate_evidence`` already built, so
#   every citation still resolves through the same, unmodified
#   ``_resolve_evidence``/``Grounding`` — no new grounding code, verified live
#   and by the unchanged evidence-validity floor. Changes the candidate
#   representation and the generation shape, so a distinct baseline from
#   0.6.0's.
#
#   Two real bugs found and fixed while landing this, both via the corpus A/B:
#   (1) an exercise ``_needs_extraction`` skipped got no finding at all in the
#   first cut — dropped from 0.4375 detail recall (vs 0.6.0's 0.722) because
#   "nothing notable happened" isn't "nothing worth reporting"; even a boring
#   exercise's basic performance facts are exactly what the corpus's required
#   evidence checks for, and the one-shot design always had them. Fixed by
#   ``_baseline_finding``: a skipped exercise gets a deterministic "performance"
#   finding (no model call) citing its own real sets, instead of silence. (2) a
#   live extraction call for a real malformed-data corpus case put its
#   limitation in a stray top-level ``"limitation"`` key instead of inside the
#   schema's array, and the parser silently dropped it — traced by calling the
#   model directly on the exact failing prompt, not guessed. Root cause: a
#   unified ``{"findings": [...]}`` schema was a novel shape for the model
#   relative to the proven, separate ``observations``/``limitations`` arrays
#   every version through 0.6.0 used. Reverted extraction to that exact proven
#   two-array shape rather than patch the parser to tolerate the stray key.
#
#   After both fixes, the corpus still failed the gate: required_detail_recall
#   0.583 (floor 0.60), cases_passing 3/12 vs 0.6.0's 6/12, safety floors and
#   schema_valid_rate still holding at 1.0/1.0/0.0. This wasn't chased with
#   more prompt tuning — the corpus's 12 cases are mostly 1-2 exercises, and
#   splitting a simple case into independent extraction+synthesis calls gives
#   the model more independent rolls of the dice to be vague on citations than
#   one focused one-shot call would (observed directly: c03 flipped between a
#   full pass and a partial miss across otherwise-identical re-runs) — exactly
#   the corpus inadequacy the Wayfinder decision (#59) anticipated going in.
#
#   A third real issue surfaced live instead, on the actual session motivating
#   this whole map: repeated runs consistently produced 4 limitations, every
#   time, shuffling between 'conflicting_data' and 'missing_data' on different
#   exercises with nothing actually wrong — not a wording problem in one
#   category, but the per-exercise extraction call treating "inspect for a
#   data-quality problem" as an invitation to always find one. A prompt-only
#   reword of the conflicting_data trigger (tried first) did not fix this — 3
#   live re-runs still produced 4 findings each time, just reshuffled. Fixed by
#   removing the model's discretion entirely: 'conflicting_data' (the one check
#   needing subjective note-vs-numbers judgment) is no longer solicited at all,
#   and 'missing_data' (skipped) is now a deterministic DATA QUALITY NOTE
#   (``_skipped_exercises``) injected only when ``activity.get("skipped")`` is
#   actually true — the same pattern 0.4.0 already used for malformed sets.
#   Live-verified: 0 limitations across repeated runs on a session with zero
#   real data-quality issues, down from 4 every time.
#
#   That fix also happened to clear the corpus gate: required_detail_recall
#   0.75 (floor 0.60), cases_passing 4/12, required_limitation_recall 0.5
#   (floor 0.33, down from 0.6.0's 0.833 — an accepted, deliberate cost: c06
#   specifically requires a conflicting_data catch this version no longer
#   attempts), gate PASS. The actual target failure mode (many exercises, real
#   accumulated history) is live-verified separately: every partition found its
#   own real signal (all 3 within-session bumps, real cross-session
#   progression), with clean 3-4 row citations and zero fabrication —
#   something the one-shot design could not do reliably.
#
#   Latency, measured (#60's actual requirement, not just architected):
#   repeated runs against the real session on its worst case for the
#   pre-filter — every one of its 7 exercises has a note, so ``_needs_extraction``
#   skips none of them and every exercise still costs a real call — land at
#   43.7-59.6s, down from the undecomposed prototype's 67.5s (same session,
#   same 8 total calls, before ``_needs_extraction``/``_baseline_finding``
#   existed). Not yet "closer to 10-25s" on this adversarial case, but the
#   mechanism's actual saving is session-dependent by design: any exercise
#   with nothing notable (the corpus's typical case, and presumably most real
#   sessions) costs zero model time via ``_baseline_finding`` instead of a full
#   extraction call. A session with fewer notable exercises than this one was
#   not separately measured here.
CAPABILITY_VERSION = "0.7.0"

from .contract import (
    Category,
    Confidence,
    Evidence,
    FactEvidence,
    Limitation,
    LimitationKind,
    Observation,
    ObservationKind,
    SetEvidence,
    WorkoutReview,
)
from .review import review_workout

__all__ = [
    "review_workout",
    "CAPABILITY_VERSION",
    "WorkoutReview",
    "Observation",
    "Evidence",
    "SetEvidence",
    "FactEvidence",
    "Limitation",
    "Category",
    "Confidence",
    "ObservationKind",
    "LimitationKind",
]
