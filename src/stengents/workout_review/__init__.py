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
CAPABILITY_VERSION = "0.4.0"

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
