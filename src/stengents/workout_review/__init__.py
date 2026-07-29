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
CAPABILITY_VERSION = "0.5.0"

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
