"""The locked Workout Review output contract (issue #17).

Pure content models: a ``WorkoutReview`` carries a factual summary, at most
three evidence-backed ``Observation``s, and any acknowledged ``Limitation``s.
Nothing operational lives here — model, latency, cost, and version belong to the
run record, never inside a review. Every closed vocabulary below is passed
through verbatim from Kiln (pinned in #18) or fixed by the contract (#17).

The naming collision with ``stengents.utilities.observation`` (a logging
primitive) is deliberate and kept apart: that module's ``Observation`` is
unrelated to this domain type.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Kiln's ``measurement.unit``, passed through verbatim; extended as new units
# appear (a contract change), so v1 pins the two seen so far.
LoadType = Literal["plate", "sec"]

# FactEvidence.field — session-level and activity-level keys, closed per #18.
FactField = Literal[
    "feel", "minutes", "date", "workout", "type",  # session-level
    "note", "skipped", "spec", "sets_completed", "planned_sets",  # activity-level
]

ObservationKind = Literal["fact", "inference"]
Confidence = Literal["firm", "tentative"]
Category = Literal["performance", "progression", "adherence", "data_quality"]
LimitationKind = Literal["insufficient_history", "missing_data", "malformed_data", "conflicting_data"]


class _Content(BaseModel):
    """A frozen, strict content model — no extra fields, no mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SetEvidence(_Content):
    """One performed set — exactly one cited row.

    ``load`` is ``None`` for a bodyweight movement; ``reps`` is ``0`` for a timed
    hold, where the work lives in ``load``/``loadType`` (a 45-second plank is
    ``reps=0, load=45, loadType="sec"``).
    """

    kind: Literal["set"] = "set"
    workout_id: str
    exercise: str
    set_index: int
    reps: int
    load: float | None
    loadType: LoadType


class FactEvidence(_Content):
    """A single non-set datum — session-level (``exercise`` is ``None``) or
    activity-level. ``value`` is the datum passed through as text."""

    kind: Literal["fact"] = "fact"
    workout_id: str
    exercise: str | None
    field: FactField
    value: str


# A cited datum is exactly one of the two, discriminated on ``kind``.
Evidence = Annotated[Union[SetEvidence, FactEvidence], Field(discriminator="kind")]


class Observation(_Content):
    """One evidence-backed statement. Both ``fact`` and ``inference`` kinds carry
    at least one Evidence row, so no claim is ever evidence-free. The exercise an
    Observation is about is derived from its Evidence, never restated here."""

    kind: ObservationKind
    confidence: Confidence
    category: Category
    claim: str
    evidence: list[Evidence] = Field(min_length=1)


class Limitation(_Content):
    """A structured acknowledgement of a gap rather than a guess, so the
    deterministic evaluator can assert the correct gap was named."""

    kind: LimitationKind
    detail: str


class WorkoutReview(_Content):
    """The review of one finished Kiln Session: a factual summary, 0–3
    Observations (hard-capped), and any Limitations. A pure content object."""

    workout_id: str
    summary: str
    observations: list[Observation] = Field(default_factory=list, max_length=3)
    limitations: list[Limitation] = Field(default_factory=list)
