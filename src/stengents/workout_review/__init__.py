"""The evidence-backed Workout Review capability over Kiln training history.

Public surface: the ``review_workout`` entry point and the locked #17 output
contract types.
"""

# The version of the review capability itself — the prompt/generation/decode
# behaviour, distinct from the model and from the output contract. Stamped onto
# every benchmark artifact so a recorded score is tied to the capability that
# produced it. Bump on a meaningful capability change (e.g. a prompt revision or
# the revise-once loop landing).
CAPABILITY_VERSION = "0.1.0"

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
