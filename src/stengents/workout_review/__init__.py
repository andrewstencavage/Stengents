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
CAPABILITY_VERSION = "0.3.0"

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
