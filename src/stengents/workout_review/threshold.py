"""The passing threshold for the Workout Review capability (ticket #33).

A benchmark run **passes** when every floor here holds. The gate is defined over
the *aggregate* metrics (#33 chose per-metric floors over a cases-passing count),
in three tiers:

* **safety** (zero-tolerance) — ``evidence_validity_rate == 1.0`` and
  ``unsupported_claim_rate == 0.0``. These carry the headline promise
  ("evidence-backed, no invented facts"). A failure here is *retested* by the
  runner before it fails the capability — see :mod:`benchmark_runner`'s
  confirm-before-failing retest — so a one-off stochastic miss doesn't sink a
  release, but a *reproducible* violation still does. That retest is the only
  thing that can flip a failed safety floor to passing; a quality floor never
  gets one.
* **structural** — ``schema_valid_rate == 1.0`` (a contract guarantee, free to
  hold).
* **quality** (revisable) — ``required_detail_recall >= 0.60`` and
  ``required_limitation_recall >= 0.33``. Set below the 0.2.0 spread so ordinary
  generation jitter doesn't flag a false regression while a real backslide trips;
  expected to ratchet up as tune-loop experiments land. The detail floor is
  anchored to the *central* recall of 0.2.0 (~0.66 over repeated runs; the
  first-recorded 0.708 was a high-tail single run) with margin below the observed
  low (0.632), since a quality floor gets no confirm-before-failing retest.

``cases_passing`` is deliberately **not** a floor — it stays in the artifact as
an informational number only.

The ``== 1.0`` / ``== 0.0`` safety floors are expressed as ``>= 1.0`` / ``<= 0.0``
so a rate that cannot exceed its bound needs no fragile float-equality check.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluator import AggregateMetrics

SAFETY = "safety"
STRUCTURAL = "structural"
QUALITY = "quality"


@dataclass(frozen=True)
class Threshold:
    """The floor values a run must clear. Defaults are the #33 decision."""

    evidence_validity_min: float = 1.0
    unsupported_claim_max: float = 0.0
    schema_valid_min: float = 1.0
    required_detail_recall_min: float = 0.60
    required_limitation_recall_min: float = 0.33


@dataclass(frozen=True)
class FloorResult:
    """One floor evaluated against a run's aggregate value.

    ``comparator`` is ``">="`` (value must be at least ``bound``) or ``"<="``
    (value must be at most ``bound``). ``retest_cleared`` is set by the runner's
    confirm-before-failing retest: when a failed *safety* floor's offending cases
    all turn out to be flakes, the floor passes on retest even though ``raw_passed``
    is ``False``.
    """

    metric: str
    tier: str
    comparator: str
    value: float
    bound: float
    retest_cleared: bool = False

    @property
    def raw_passed(self) -> bool:
        """Whether the floor held on the metric alone, before any retest."""
        if self.comparator == ">=":
            return self.value >= self.bound
        return self.value <= self.bound

    @property
    def passed(self) -> bool:
        return self.raw_passed or self.retest_cleared

    def cleared_by_retest(self) -> "FloorResult":
        """This floor, marked as cleared by the confirm-before-failing retest."""
        return FloorResult(
            metric=self.metric,
            tier=self.tier,
            comparator=self.comparator,
            value=self.value,
            bound=self.bound,
            retest_cleared=True,
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "tier": self.tier,
            "comparator": self.comparator,
            "value": self.value,
            "bound": self.bound,
            "raw_passed": self.raw_passed,
            "retest_cleared": self.retest_cleared,
            "passed": self.passed,
        }


def evaluate_floors(
    aggregate: AggregateMetrics, threshold: Threshold = Threshold()
) -> list[FloorResult]:
    """The five floors evaluated against ``aggregate`` — no retest applied yet."""
    return [
        FloorResult("evidence_validity_rate", SAFETY, ">=", aggregate.evidence_validity_rate, threshold.evidence_validity_min),
        FloorResult("unsupported_claim_rate", SAFETY, "<=", aggregate.unsupported_claim_rate, threshold.unsupported_claim_max),
        FloorResult("schema_valid_rate", STRUCTURAL, ">=", aggregate.schema_valid_rate, threshold.schema_valid_min),
        FloorResult("required_detail_recall", QUALITY, ">=", aggregate.required_detail_recall, threshold.required_detail_recall_min),
        FloorResult("required_limitation_recall", QUALITY, ">=", aggregate.required_limitation_recall, threshold.required_limitation_recall_min),
    ]


@dataclass(frozen=True)
class RetestReport:
    """What the confirm-before-failing retest did, for the artifact.

    ``performed`` is ``False`` when no safety floor failed the initial run (so no
    retest was needed). ``n`` is the confirmation-run budget per offending case.
    ``cases`` maps each retested safety metric to ``{case_id: "reproduced" |
    "cleared"}`` — "reproduced" means the violation recurred in at least one
    confirmation run (a confirmed defect); "cleared" means all ``n`` confirmation
    runs were clean (a one-off flake).
    """

    performed: bool
    n: int
    cases: dict[str, dict[str, str]]

    def to_dict(self) -> dict:
        return {"performed": self.performed, "n": self.n, "cases": self.cases}


@dataclass(frozen=True)
class GateReport:
    """The overall pass/fail verdict plus its per-floor breakdown."""

    floors: tuple[FloorResult, ...]
    retest: RetestReport

    @property
    def passed(self) -> bool:
        return all(floor.passed for floor in self.floors)

    @property
    def safety_floors_failed(self) -> tuple[FloorResult, ...]:
        return tuple(f for f in self.floors if f.tier == SAFETY and not f.raw_passed)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "floors": [floor.to_dict() for floor in self.floors],
            "retest": self.retest.to_dict(),
        }
