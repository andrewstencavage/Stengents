"""PROTOTYPE — throwaway, answers Wayfinder ticket #58 (map #56). Not shipped.

Question this prototype answers: does splitting review_workout's single
one-shot completion into a per-partition EXTRACTION phase (full attention per
exercise, no competition) and a single SYNTHESIS phase (select <=3, write the
final review) actually work as a data shape — specifically, does the existing
"every cited row resolves verbatim" grounding guarantee (contract.py's locked
#17 types) survive the split unchanged, or does it need new machinery?

Design under test: extraction and synthesis both operate over the SAME shared,
already-built, globally-numbered candidate pool that review.py's
_candidate_evidence already produces today. A partition is just a subset of
those global ids — extraction never invents its own local numbering, so a
finding's evidence_ids are already valid ids into the shared pool. Synthesis
picks among findings and hands their evidence_ids straight to the *existing*
_resolve_evidence/Grounding machinery, completely unchanged. If that holds,
decomposition costs zero new grounding code — only prompt/orchestration change.

Pure logic here: no I/O except through the injectable `complete` seam (same
pattern review.py itself already uses for testability). The TUI shell
(prototype_decompose_tui.py) is the only impure part, and is not meant to
survive past this ticket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .contract import (
    Category,
    Confidence,
    FactEvidence,
    Limitation,
    LimitationKind,
    Observation,
    ObservationKind,
    SetEvidence,
    WorkoutReview,
)
from .grounding import Grounding
from .review import Candidate, _resolve_evidence, _factual_summary

Complete = Callable[[str], str]


@dataclass(frozen=True)
class Partition:
    """One extraction unit: a subject exercise, or the session-level facts.

    ``candidate_ids`` are 1-based positions into the SAME shared candidate list
    review.py's _candidate_evidence already builds — not a private numbering.
    """

    key: str  # exercise name, or "__session__"
    candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExtractionFinding:
    """One partition's proposed observation or limitation, citing shared-pool ids.

    Deliberately looser than the real Observation/Limitation contract types —
    extraction is allowed to propose more than 3 and can be wrong; synthesis
    (via the existing _resolve_evidence/Grounding) is where invalid citations
    get dropped and the real, validated contract objects get built.
    """

    partition_key: str
    kind: str  # "observation" | "limitation"
    category: str | None = None
    claim: str | None = None
    evidence_ids: tuple[int, ...] = ()
    limitation_kind: str | None = None
    limitation_detail: str | None = None


def partition_session(
    session: dict, selected: dict[str, list[dict]], candidates: list[Candidate]
) -> list[Partition]:
    """One partition per subject exercise (its own sets/facts + its own history
    group), in the session's own exercise order, plus one session-level
    partition (date/type/minutes/feel — where an 'adherence over the whole
    session' claim would ground) first.
    """
    by_exercise: dict[str, list[int]] = {}
    session_ids: list[int] = []
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, list):  # a grouped history slot
            by_exercise.setdefault(candidate[0].exercise, []).append(index)
            continue
        exercise = getattr(candidate, "exercise", None)
        if exercise is None:  # a session-level FactEvidence (date/type/minutes/feel)
            session_ids.append(index)
        else:
            by_exercise.setdefault(exercise, []).append(index)

    partitions = [Partition(key="__session__", candidate_ids=tuple(session_ids))]
    for name in selected:  # selected preserves the subject's own exercise order
        partitions.append(Partition(key=name, candidate_ids=tuple(by_exercise.get(name, ()))))
    return partitions


def _render_id(index: int, candidate: Candidate) -> str:
    if isinstance(candidate, list):
        first = candidate[0]
        return json.dumps(
            {
                "kind": "history_sets",
                "exercise": first.exercise,
                "sets": [{"reps": r.reps, "load": r.load, "loadType": r.loadType} for r in candidate],
            }
        )
    return json.dumps(candidate.model_dump())


_EXTRACT_INSTRUCTIONS = (
    "You are reviewing ONE exercise (or session-level facts) from a workout, in isolation. "
    "List every fact worth noting about ONLY this partition's evidence — do not worry about "
    "picking the 'best' ones or staying under any limit; a later step selects among your findings. "
    "Cite evidence ONLY by the integer ids listed below. Output STRICT JSON: "
    '{"findings": [{"kind": "observation", "category": "performance"|"progression"|"adherence"|"data_quality", '
    '"claim": str, "evidence_ids": [int, ...]}, '
    '{"kind": "limitation", "limitation_kind": "insufficient_history"|"missing_data"|"malformed_data"|"conflicting_data", '
    '"detail": str}]}'
)


def build_extraction_prompt(partition: Partition, candidates: list[Candidate]) -> str:
    lines = [_EXTRACT_INSTRUCTIONS, "", f"PARTITION: {partition.key}", "", "EVIDENCE (id: row):"]
    for index in partition.candidate_ids:
        lines.append(f"{index}: {_render_id(index, candidates[index - 1])}")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def extract(partition: Partition, candidates: list[Candidate], complete: Complete) -> list[ExtractionFinding]:
    """One small model call, scoped to just this partition's candidate ids."""
    if not partition.candidate_ids:
        return []
    prompt = build_extraction_prompt(partition, candidates)
    try:
        parsed = json.loads(complete(prompt))
    except Exception:  # noqa: BLE001 — a broken partition call yields no findings, not a crash.
        return []
    findings: list[ExtractionFinding] = []
    for item in parsed.get("findings", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        findings.append(
            ExtractionFinding(
                partition_key=partition.key,
                kind=item.get("kind", ""),
                category=item.get("category"),
                claim=item.get("claim"),
                evidence_ids=tuple(i for i in item.get("evidence_ids", []) if isinstance(i, int)),
                limitation_kind=item.get("limitation_kind"),
                limitation_detail=item.get("detail"),
            )
        )
    return findings


_SYNTHESIZE_INSTRUCTIONS = (
    "Below are candidate findings extracted independently per exercise/session-facts, each already "
    "citing real evidence ids. Select the best at most 3 'observation' findings (prefer variety across "
    "categories and exercises over redundancy) and write a short factual summary of the whole session. "
    "Every 'limitation' finding should be kept as-is (they are not competing for a slot). Output STRICT JSON: "
    '{"summary": str, "selected_observation_indexes": [int, ...]}\n'
    "selected_observation_indexes are 1-based positions into the OBSERVATION FINDINGS list below."
)


def build_synthesis_prompt(findings: list[ExtractionFinding]) -> str:
    observations = [f for f in findings if f.kind == "observation"]
    lines = [_SYNTHESIZE_INSTRUCTIONS, "", "OBSERVATION FINDINGS:"]
    for index, finding in enumerate(observations, start=1):
        lines.append(
            f"{index}: [{finding.partition_key}] {finding.category} | {finding.claim} "
            f"(evidence ids {list(finding.evidence_ids)})"
        )
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def synthesize(
    session: dict,
    findings: list[ExtractionFinding],
    candidates: list[Candidate],
    grounding: Grounding,
    complete: Complete,
) -> WorkoutReview:
    """The one synthesis call: select among findings, write the summary.

    Selected observations' evidence_ids are resolved through the EXISTING,
    unchanged _resolve_evidence/Grounding machinery — proving the grounding
    guarantee needs no new code for the decomposed shape.
    """
    observations_found = [f for f in findings if f.kind == "observation"]
    limitations_found = [f for f in findings if f.kind == "limitation"]

    summary = ""
    selected_indexes: list[int] = []
    if observations_found:
        try:
            parsed = json.loads(complete(build_synthesis_prompt(observations_found)))
            if isinstance(parsed, dict):
                summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
                selected_indexes = [i for i in parsed.get("selected_observation_indexes", []) if isinstance(i, int)]
        except Exception:  # noqa: BLE001 — degrade cleanly, same as review.py today.
            pass

    observations: list[Observation] = []
    for index in selected_indexes:
        if not (1 <= index <= len(observations_found)):
            continue
        finding = observations_found[index - 1]
        evidence = _resolve_evidence(list(finding.evidence_ids), candidates, grounding)
        if not evidence:
            continue
        try:
            observations.append(
                Observation(
                    kind="fact",
                    confidence="firm",
                    category=finding.category,  # type: ignore[arg-type]
                    claim=finding.claim or "",
                    evidence=evidence,
                )
            )
        except Exception:  # noqa: BLE001 — drop a finding the contract rejects.
            continue
        if len(observations) == 3:
            break

    limitations: list[Limitation] = []
    seen: set[tuple[str, str]] = set()
    for finding in limitations_found:
        key = (finding.limitation_kind or "", finding.limitation_detail or "")
        if key in seen or not finding.limitation_kind or not finding.limitation_detail:
            continue
        try:
            limitations.append(Limitation(kind=finding.limitation_kind, detail=finding.limitation_detail))  # type: ignore[arg-type]
            seen.add(key)
        except Exception:  # noqa: BLE001
            continue

    if not summary:
        summary = _factual_summary(session)

    return WorkoutReview(
        workout_id=session["id"],
        summary=summary,
        observations=observations,
        limitations=limitations,
    )
