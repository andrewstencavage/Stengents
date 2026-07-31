"""The stable Workout Review entry point: ``review_workout(workout_id)``.

Independent of the ``kiln_coach`` chat agent. It reads one finished Session by
its stable id — via the injected ``fetch``/``fetch_history``/``fetch_plans``
seams below, defaulting to ``kiln_client``'s HTTP fetchers — selects each
performed exercise's comparison history (#19), resolves the development-time
model from ``model_source``, and assembles an evidence-backed review — decoded
strictly into the #17 contract.

ADR-0005: the coach server (``stengents serve-coach``) wires these seams to
``kiln_mcp_client``'s MCP-backed fetchers instead (see ``cli.py``), so this
module's HTTP defaults are only reached by a caller that doesn't override
them. Either transport returns the same dict shape, so nothing else here
changes with the transport.

As of ADR 0004, generation has two independent tracks, both feeding the same
uncapped ``observations`` list with no model call selecting among them:

* **Deterministic** (:mod:`progression`, no model call): a ``progression``
  Observation (Top-set delta) for every exercise with comparison history, a
  plain ``performance`` recap for every exercise with none, and an
  ``adherence`` Observation when a Plan streak is active. All built directly
  from source data — no candidate-index indirection, nothing for a model to
  miscalculate.
* **Extraction** (per-exercise model call, unchanged in shape from #60): finds
  qualitative findings — notes, skipped/malformed-data limitations — that
  numbers alone can't surface. ``_needs_extraction`` now triggers purely on
  those qualitative signals (a note, an explicit skip, malformed data); the
  old numeric-change heuristic moved entirely to the deterministic track.
  Extraction is barred from proposing ``progression`` (comparison is no
  longer its job) both by prompt instruction and a decode-time filter — but
  it still *sees* comparison history as context: a first attempt at dropping
  it entirely regressed c11's malformed-data recall (4/4 reproduced, not a
  flake), so ``build_extraction_prompt`` keeps showing it, just captioned
  "context only."

Grounding is structural, not trusted, for both tracks: every deterministic
Observation's evidence and every resolved extraction citation is checked
against one shared ``Grounding`` built over the subject, its comparison
history, and any Plan-streak milestone Sessions before entering the review.

Three seams keep the whole pipeline testable offline: ``fetch`` (subject by
id), ``fetch_history`` (the finished-Session pool), ``fetch_plans`` (every
Plan, for the streak), ``model``, and ``complete`` (one model call, reused for
every extraction). A test injects a fake ``complete`` returning canned
structured replies and never touches the endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Literal

from farm_system.kiln_coach import kiln_client
from farm_system.kiln_coach.kiln_client import performed_sets
from stengents.utilities.model_source import ModelConnection, resolve_model

from .contract import (
    Evidence,
    FactEvidence,
    Limitation,
    Observation,
    SetEvidence,
    WorkoutReview,
)
from .grounding import Grounding, as_text
from .progression import plan_streak, streak_observation, top_set_delta

# The review capability wants the same larger tool-calling model kiln_coach uses.
DEFAULT_MODEL_NAME = "qwen2.5:7b-8k"

# Comparison history is capped at the 10 most recent prior instances (#19).
# Only the newest entry is used by the deterministic Top-set delta (ADR 0004);
# the fuller list still backs _history_limitations' insufficient-history check.
HISTORY_CAP = 10

# The seam onto Kiln: by-id fetch of one finished Session, injectable for tests.
Fetch = Callable[[str], "dict | None"]
# The seam onto the finished-Session pool comparison history is selected from.
FetchHistory = Callable[..., "list[dict]"]
# The seam onto every Plan (active/historical/draft), for the Plan streak.
FetchPlans = Callable[..., "list[dict]"]
# The single model call: a resolved connection + a prompt -> raw model text.
Complete = Callable[[ModelConnection, str], str]

_CATEGORIES = frozenset({"performance", "progression", "adherence", "data_quality"})
_KINDS = frozenset({"fact", "inference"})
_CONFIDENCES = frozenset({"firm", "tentative"})
_LIMITATION_KINDS = frozenset(
    {"insufficient_history", "missing_data", "malformed_data", "conflicting_data"}
)


def review_workout(
    workout_id: str,
    *,
    fetch: Fetch = kiln_client.fetch_workout,
    fetch_history: FetchHistory = kiln_client.fetch_sessions,
    fetch_plans: FetchPlans = kiln_client.fetch_plans,
    model: ModelConnection | None = None,
    complete: Complete | None = None,
) -> WorkoutReview:
    """Review one finished Kiln Session, addressed by its stable ``workout_id``.

    Returns a schema-valid ``WorkoutReview`` in every case: an unknown id yields
    a review whose only content is a ``missing_data`` Limitation, never an
    exception. ``fetch``, ``fetch_history``, ``fetch_plans``, ``model``, and
    ``complete`` are all injectable so a caller (or a test) can supply frozen
    data and skip the endpoint entirely.
    """
    session = fetch(workout_id)
    if session is None:
        return WorkoutReview(
            workout_id=workout_id,
            limitations=[
                Limitation(
                    kind="missing_data",
                    detail=f"No finished Session with id {workout_id!r}.",
                )
            ],
        )
    pool = kiln_client.finished_sessions(fetch_history())
    connection = model or resolve_model(DEFAULT_MODEL_NAME)
    return _generate_review(session, pool, fetch_plans(), connection, complete or _complete_with_model)


def _generate_review(
    session: dict,
    pool: list[dict],
    plans: list[dict],
    model: ModelConnection,
    complete: Complete,
) -> WorkoutReview:
    """Produce the review for a fetched Session against its comparison history
    and Plan history.

    Builds the deterministic progression/baseline/streak Observations, runs
    extraction for whichever exercises still need a qualitative look, grounds
    everything against one shared pool, and assembles the final review with no
    selection step — every Observation that grounds is kept (ADR 0004). Any
    failure to reach or parse the model degrades to the deterministic
    Observations alone plus acknowledged limitations, never raises.
    """
    workout_id = session["id"]
    selected = select_comparison_history(session, pool)
    history_limits = _history_limitations(selected)
    malformed = _malformed_set_notes(session)
    skipped = _skipped_exercises(session)
    activities_by_name = {a.get("name"): a for a in session.get("activities") or [] if a.get("name")}

    deterministic_observations: list[Observation] = []
    for name, priors in selected.items():
        activity = activities_by_name.get(name, {})
        if priors:
            prior_session = priors[0]["session"]
            prior_activity = priors[0]["activity"]
            subject_rows = _set_rows(workout_id, name, activity)
            prior_rows = _set_rows(prior_session.get("id"), name, prior_activity)
            delta = top_set_delta(name, subject_rows, prior_rows)
            if delta is not None:
                deterministic_observations.append(delta)
        else:
            baseline = _baseline_performance_observation(workout_id, name, activity)
            if baseline is not None:
                deterministic_observations.append(baseline)

    streak_result = plan_streak(plans, pool, session.get("date") or "")
    streak = streak_observation(streak_result)
    if streak is not None:
        deterministic_observations.append(streak)

    grounding = review_grounding(session, selected, streak_result.milestones)
    # Defense in depth: every deterministic Observation is built directly from
    # source data and should always ground, but never trust that unconditionally.
    deterministic_observations = [
        observation
        for observation in deterministic_observations
        if all(grounding.resolves(row) for row in observation.evidence)
    ]

    candidates = _candidate_evidence(session)
    partitions = partition_session(session, selected, candidates, malformed)
    findings: list[ExtractionFinding] = []
    for partition in partitions:
        findings.extend(extract(partition, candidates, selected, malformed, skipped, model, complete))

    extraction_observations = _decode_extraction_observations(findings, candidates, grounding)
    model_limits = _decode_limitations(
        [{"kind": f.limitation_kind, "detail": f.limitation_detail} for f in findings if f.finding_kind == "limitation"]
    )

    observations = [*deterministic_observations, *extraction_observations]
    limitations = _dedupe_limitations(history_limits + model_limits)
    return WorkoutReview(workout_id=workout_id, observations=observations, limitations=limitations)


# --- comparison history (#19) -------------------------------------------


def select_comparison_history(session: dict, pool: list[dict]) -> dict[str, list[dict]]:
    """Select each performed exercise's comparison history, deterministically.

    For every activity in ``session`` carrying non-empty ``performedSets``,
    return the finished Sessions strictly earlier by ``date`` that performed an
    activity of the **exact same name** *with* non-empty ``performedSets``,
    newest-first and capped at :data:`HISTORY_CAP`. The mapping's keys are the
    subject's performed exercise names; an exercise with no priors maps to ``[]``
    (its *insufficient history*), never omitted.

    Each prior is the subject-shaped ``{"session": <Session>, "activity":
    <Activity>}`` so callers can cite the prior's own ``id`` and sets.
    """
    subject_date = session.get("date") or ""
    subject_id = session.get("id")
    earlier = sorted(
        (
            other
            for other in pool
            if other.get("id") != subject_id
            and other.get("status", "finished") == "finished"
            and (other.get("date") or "") < subject_date
        ),
        key=lambda other: other.get("date") or "",
        reverse=True,
    )

    history: dict[str, list[dict]] = {}
    for activity in session.get("activities") or []:
        if not activity.get("performedSets"):
            continue
        name = activity.get("name")
        if name in history:
            continue
        priors: list[dict] = []
        for other in earlier:
            for other_activity in other.get("activities") or []:
                if other_activity.get("name") == name and other_activity.get("performedSets"):
                    priors.append({"session": other, "activity": other_activity})
                    break
            if len(priors) >= HISTORY_CAP:
                break
        history[name] = priors
    return history


def review_grounding(
    session: dict, selected: dict[str, list[dict]], streak_milestones: Iterable[dict] = ()
) -> Grounding:
    """Grounding over the pool a review's evidence must resolve against: the
    subject Session, every prior Session its comparison history was drawn
    from, and any Plan-streak milestone Sessions (ADR 0004). Shared with the
    corpus test so production and test can't drift on what that pool is.
    """
    pool = [session, *(prior["session"] for priors in selected.values() for prior in priors), *streak_milestones]
    return Grounding(pool)


def _history_limitations(selected: dict[str, list[dict]]) -> list[Limitation]:
    """A ``Limitation(insufficient_history)`` for each exercise with zero priors."""
    return [
        Limitation(
            kind="insufficient_history",
            detail=(
                f"No prior finished Session performed {name!r} with logged sets; "
                "no progression comparison was made."
            ),
        )
        for name, priors in selected.items()
        if not priors
    ]


# --- deterministic Observations (ADR 0004) -------------------------------


def _baseline_performance_observation(workout_id: str, name: str, activity: dict) -> Observation | None:
    """A plain factual recap for an exercise with no comparison history — no
    progression story is possible yet, but what was performed is still worth
    knowing. ``None`` when the exercise offers no citable sets at all."""
    rows = _set_rows(workout_id, name, activity)
    if not rows:
        return None
    collapsed = kiln_client._collapse_sets(activity.get("performedSets") or [])
    return Observation(
        kind="fact",
        confidence="firm",
        category="performance",
        claim=f"{name}: {collapsed}.",
        evidence=rows,
    )


def _set_rows(workout_id: str, exercise: str | None, activity: dict) -> list[SetEvidence]:
    rows: list[SetEvidence] = []
    for row in performed_sets(activity):
        try:
            rows.append(SetEvidence(workout_id=workout_id, exercise=exercise, **row))
        except Exception:  # noqa: BLE001 — skip a malformed/out-of-vocabulary set.
            continue
    return rows


def _malformed_set_notes(session: dict) -> dict[str, int]:
    """Per-exercise count of the subject's performed sets that fail to parse.

    ``_set_rows`` silently drops any ``performedSet`` that can't build a valid
    ``SetEvidence`` — a set missing its load unit, a null measurement, an
    out-of-vocabulary ``loadType`` — so those rows never reach the candidate
    evidence or the prompt, leaving a prompt-only nudge nothing to name. This
    counts, per subject exercise, how many of its logged sets were unparseable so
    ``build_extraction_prompt`` can surface the signal as a DATA QUALITY NOTE
    line. It is deliberately *not* a citable candidate: a synthetic 'malformed' fact could
    never ground verbatim against real Kiln fields and would be dropped by the
    resolvability guard — but a ``Limitation`` needs no evidence citation.
    """
    workout_id = session.get("id", "")
    notes: dict[str, int] = {}
    for activity in session.get("activities") or []:
        name = activity.get("name")
        if not name:
            continue
        total = len(activity.get("performedSets") or [])
        parsed = len(_set_rows(workout_id, name, activity))
        dropped = total - parsed
        if dropped > 0:
            notes[name] = dropped
    return notes


def _skipped_exercises(session: dict) -> set[str]:
    """Subject exercise names explicitly marked skipped.

    Deterministic, same pattern as ``_malformed_set_notes``: live testing found
    the model prone to inventing a ``missing_data``/``conflicting_data``
    limitation on exercises with nothing actually wrong when left to judge for
    itself, so ``build_extraction_prompt`` surfaces this as an explicit
    DATA QUALITY NOTE the model must react to, rather than something it goes
    looking for.
    """
    return {
        activity.get("name")
        for activity in session.get("activities") or []
        if activity.get("name") and activity.get("skipped")
    }


def _needs_extraction(name: str, activity: dict, malformed: dict[str, int]) -> bool:
    """Whether this exercise needs a real *qualitative* extraction call.

    Numeric change detection moved entirely to the deterministic Top-set delta
    (ADR 0004), which runs unconditionally for every exercise with comparison
    history — this now only decides whether there's something qualitative a
    model might find: a note, an explicit skip, or malformed/dropped sets.
    """
    return bool(activity.get("note")) or bool(activity.get("skipped")) or name in malformed


def _session_facts(workout_id: str, session: dict) -> list[Evidence]:
    pairs = {
        "date": session.get("date"),
        "workout": session.get("workoutName"),
        "type": session.get("type"),
        "minutes": session.get("minutes"),
        "feel": session.get("feel"),
    }
    return _facts(workout_id, None, pairs)


def _activity_facts(workout_id: str, exercise: str | None, activity: dict) -> list[Evidence]:
    prescribed = ((activity.get("plannedActivity") or {}).get("prescribedSets")) or []
    pairs: dict[str, object] = {
        "note": activity.get("note"),
        "spec": activity.get("spec"),
    }
    if activity.get("skipped"):
        pairs["skipped"] = True
    # Only a *literal* ``sets_completed`` field grounds — that's what grounding
    # (and the evaluator) resolve it against. A structured activity's count is
    # derived from its ``performedSets``, which are already citable one-per-row as
    # SetEvidence; synthesising ``sets_completed = len(performedSets)`` here
    # manufactured a candidate that could never resolve, and the model citing it
    # was the sole source of the baseline's unsupported claims. So offer it only
    # when Kiln actually carries the field (freeform activities).
    if activity.get("sets_completed") is not None:
        pairs["sets_completed"] = activity.get("sets_completed")
    if prescribed:
        pairs["planned_sets"] = len(prescribed)
    return _facts(workout_id, exercise, pairs)


def _facts(workout_id: str, exercise: str | None, pairs: dict) -> list[Evidence]:
    facts: list[Evidence] = []
    for field, value in pairs.items():
        if value is None:
            continue
        try:
            facts.append(
                FactEvidence(
                    workout_id=workout_id,
                    exercise=exercise,
                    field=field,
                    # Render through the shared ``as_text`` so a value grounds
                    # against what the evaluator resolves it to — a Python bool is
                    # JSON-cased ("true"), never str()'s "True".
                    value=as_text(value),
                )
            )
        except Exception:  # noqa: BLE001 — skip an out-of-vocabulary field.
            continue
    return facts


def _candidate_evidence(session: dict) -> list[Evidence]:
    """Build every citable prompt slot from real subject data, for extraction.

    Comparison-history evidence is no longer offered here (ADR 0004):
    extraction no longer proposes ``progression``, so it has no need to cite a
    prior's sets — only the deterministic Top-set delta does, and it builds
    its own evidence directly from source data, bypassing this candidate list
    entirely. The model never emits raw evidence values — it selects from this
    list by its 1-based position, so every cited row is real by construction.
    """
    workout_id = session["id"]
    candidates: list[Evidence] = []
    candidates.extend(_session_facts(workout_id, session))
    for activity in session.get("activities") or []:
        name = activity.get("name")
        candidates.extend(_set_rows(workout_id, name, activity))
        candidates.extend(_activity_facts(workout_id, name, activity))
    return candidates


# --- partitioning (#60) --------------------------------------------------


@dataclass(frozen=True)
class Partition:
    """One extraction unit: a subject exercise that needs a real qualitative
    look.

    ``candidate_ids`` are 1-based positions into the same shared candidate list
    ``_candidate_evidence`` builds — not a private numbering, so a citation from
    any partition resolves through the unmodified ``_resolve_evidence``.
    """

    key: str  # exercise name
    candidate_ids: tuple[int, ...]


def partition_session(
    session: dict,
    selected: dict[str, list[dict]],
    candidates: list[Evidence],
    malformed: dict[str, int],
) -> list[Partition]:
    """One partition per subject exercise that needs a real qualitative look.

    Progression (the deterministic Top-set delta) and the no-history baseline
    recap are both fully deterministic (ADR 0004) and never need a partition —
    this only decides where a model call earns its cost: a note, an explicit
    skip, or malformed data (``_needs_extraction``). No session-level
    partition: session facts never appear in any corpus case's required
    evidence.
    """
    by_exercise: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates, start=1):
        exercise = getattr(candidate, "exercise", None)
        if exercise is not None:
            by_exercise.setdefault(exercise, []).append(index)

    activities_by_name = {a.get("name"): a for a in session.get("activities") or [] if a.get("name")}

    partitions: list[Partition] = []
    for name in selected:  # selected preserves the subject's own exercise order
        activity = activities_by_name.get(name, {})
        if _needs_extraction(name, activity, malformed):
            partitions.append(Partition(key=name, candidate_ids=tuple(by_exercise.get(name, ()))))
    return partitions


# --- extraction -----------------------------------------------------------


_SYSTEM = (
    "You are a strength-training review generator. Output STRICT JSON only — "
    "no prose outside the JSON."
)


@dataclass(frozen=True)
class ExtractionFinding:
    """One partition's proposed observation or limitation, citing shared-pool ids.

    Looser than the real Observation/Limitation contract types — extraction can
    propose as many findings as it finds and can be wrong; decoding (via
    ``_resolve_evidence``/``Grounding``/vocabulary checks) is where invalid
    citations and out-of-vocabulary values get dropped and the real, validated
    contract objects get built.
    """

    partition_key: str
    finding_kind: Literal["observation", "limitation"]
    category: str | None = None
    observation_kind: str | None = None  # "fact" | "inference"
    confidence: str | None = None  # "firm" | "tentative"
    claim: str | None = None
    evidence_ids: tuple[int, ...] = ()
    limitation_kind: str | None = None
    limitation_detail: str | None = None


_EXTRACTION_INSTRUCTIONS = (
    "You are reviewing ONE exercise from a workout session in isolation. List every "
    "qualitative fact worth noting about ONLY the evidence below; do not worry about "
    "picking the 'best' ones or staying under any limit.\n"
    "Rules:\n"
    "- Cite evidence ONLY by the integer ids listed under EVIDENCE. Never invent sets, "
    "numbers, or facts.\n"
    "- NEVER use category 'progression' — cross-session comparison is computed "
    "separately from real data, not your job here.\n"
    "- Only add a 'limitation' when this prompt EXPLICITLY tells you to below (a DATA "
    "QUALITY NOTE naming a required kind) — never infer a data-quality problem on your "
    "own judgment (e.g. from a note's tone or wording). Most exercises have no "
    "limitation at all; that is the normal, expected case, not a gap to fill.\n"
    "Output JSON shape:\n"
    '{"observations": [{"kind": "fact"|"inference", "confidence": "firm"|"tentative",'
    ' "category": "performance"|"adherence"|"data_quality",'
    ' "claim": str, "evidence_ids": [int, ...]}],'
    ' "limitations": [{"kind": "missing_data"|"malformed_data",'
    ' "detail": str}]}'
)


def build_extraction_prompt(
    partition: Partition,
    candidates: list[Evidence],
    selected: dict[str, list[dict]],
    malformed: dict[str, int],
    skipped: set[str],
) -> str:
    """One partition's whole prompt: its comparison history (context only —
    extraction is barred from proposing 'progression' from it, ADR 0004), any
    deterministic DATA QUALITY NOTE (malformed/skipped) the model must react
    to, plus its own slice of the shared candidate pool.

    Comparison history was cut from this prompt once progression moved to the
    deterministic Top-set delta, on the assumption extraction had no more use
    for it — reverted after a live A/B showed a real, reproducible regression:
    without it, the model started calling malformed data 'missing_data'
    instead of 'malformed_data' on c11 (4/4 repeated calls, not a flake). The
    history evidently still helps the model's general judgment, not just
    progression-finding.
    """
    lines: list[str] = [_EXTRACTION_INSTRUCTIONS, "", f"PARTITION: {partition.key}", ""]
    priors = selected.get(partition.key) or []
    if priors:
        summary = [
            {
                "date": (prior["session"].get("date") or "")[:10],
                "sets": kiln_client._collapse_sets(prior["activity"].get("performedSets") or []),
            }
            for prior in priors
        ]
        lines.append(f"COMPARISON HISTORY (newest first, context only — do not use category 'progression'): {json.dumps(summary)}")
    dropped = malformed.get(partition.key)
    if dropped:
        plural = "s" if dropped != 1 else ""
        lines.append(
            f"DATA QUALITY NOTE: {dropped} performed set{plural} were unreadable/malformed "
            f"and dropped — not shown as evidence. You MUST add a 'malformed_data' "
            f"limitation naming {partition.key!r}."
        )
    if partition.key in skipped:
        lines.append(
            f"DATA QUALITY NOTE: {partition.key!r} was marked skipped. You MUST add a "
            f"'missing_data' limitation naming {partition.key!r}."
        )
    lines.append("")
    lines.append("EVIDENCE (id: row):")
    for index in partition.candidate_ids:
        lines.append(f"{index}: {json.dumps(candidates[index - 1].model_dump())}")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def extract(
    partition: Partition,
    candidates: list[Evidence],
    selected: dict[str, list[dict]],
    malformed: dict[str, int],
    skipped: set[str],
    model: ModelConnection,
    complete: Complete,
) -> list[ExtractionFinding]:
    """One small model call, scoped to just this partition's candidate ids."""
    if not partition.candidate_ids:
        return []
    prompt = build_extraction_prompt(partition, candidates, selected, malformed, skipped)
    try:
        parsed = _extract_json(complete(model, prompt))
    except Exception:  # noqa: BLE001 — a broken partition call yields no findings, not a crash.
        return []
    if not isinstance(parsed, dict):
        return []

    def observation(item: dict) -> ExtractionFinding:
        return ExtractionFinding(
            partition_key=partition.key,
            finding_kind="observation",
            category=item.get("category"),
            observation_kind=item.get("kind"),
            confidence=item.get("confidence"),
            claim=item.get("claim"),
            evidence_ids=tuple(
                i for i in item.get("evidence_ids", []) if isinstance(i, int) and not isinstance(i, bool)
            ),
        )

    def limitation(item: dict) -> ExtractionFinding:
        return ExtractionFinding(
            partition_key=partition.key,
            finding_kind="limitation",
            limitation_kind=item.get("kind"),
            limitation_detail=item.get("detail"),
        )

    return _parsed_findings(parsed.get("observations", []), observation) + _parsed_findings(
        parsed.get("limitations", []), limitation
    )


def _parsed_findings(raw_items: object, build: Callable[[dict], ExtractionFinding]) -> list[ExtractionFinding]:
    if not isinstance(raw_items, list):
        return []
    return [build(item) for item in raw_items if isinstance(item, dict)]


# --- decode extraction findings into the contract (ADR 0004) --------------


def _decode_extraction_observations(
    findings: list[ExtractionFinding], candidates: list[Evidence], grounding: Grounding
) -> list[Observation]:
    """Validate and include every extraction-proposed observation — no
    selection, no cap (ADR 0004): a model-found qualitative finding is kept
    whenever it's schema-valid and every cited row grounds. ``progression`` is
    rejected here too, as a decode-time backstop to the prompt instruction —
    that comparison is deterministic-only now, never trust the model not to
    try anyway.
    """
    observations: list[Observation] = []
    for finding in findings:
        if finding.finding_kind != "observation":
            continue
        if (
            finding.category not in _CATEGORIES
            or finding.category == "progression"
            or finding.observation_kind not in _KINDS
            or finding.confidence not in _CONFIDENCES
            or not isinstance(finding.claim, str)
            or not finding.claim.strip()
        ):
            continue
        evidence = _resolve_evidence(list(finding.evidence_ids), candidates, grounding)
        if not evidence:
            continue
        try:
            observations.append(
                Observation(
                    kind=finding.observation_kind,  # type: ignore[arg-type]
                    confidence=finding.confidence,  # type: ignore[arg-type]
                    category=finding.category,  # type: ignore[arg-type]
                    claim=finding.claim,
                    evidence=evidence,
                )
            )
        except Exception:  # noqa: BLE001 — drop a finding the contract rejects.
            continue
    return observations


def _extract_json(raw: str) -> dict | None:
    """Parse a model reply into a dict, tolerating prose around the JSON body."""
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_evidence(raw_ids: object, candidates: list[Evidence], grounding: Grounding) -> list[Evidence]:
    if not isinstance(raw_ids, list):
        return []
    resolved: list[Evidence] = []
    seen: set[int] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        index = raw_id - 1  # ids are 1-based in the prompt.
        if 0 <= index < len(candidates) and index not in seen:
            candidate = candidates[index]
            if not grounding.resolves(candidate):
                continue  # guard: never surface a row that doesn't ground verbatim.
            seen.add(index)
            resolved.append(candidate)
    return resolved


def _decode_limitations(raw_limitations: object) -> list[Limitation]:
    if not isinstance(raw_limitations, list):
        return []
    decoded: list[Limitation] = []
    for item in raw_limitations:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        detail = item.get("detail")
        if kind not in _LIMITATION_KINDS or not isinstance(detail, str) or not detail.strip():
            continue
        decoded.append(Limitation(kind=kind, detail=detail))  # type: ignore[arg-type]
    return decoded


def _dedupe_limitations(limitations: list[Limitation]) -> list[Limitation]:
    seen: set[tuple[str, str]] = set()
    unique: list[Limitation] = []
    for limitation in limitations:
        key = (limitation.kind, limitation.detail)
        if key not in seen:
            seen.add(key)
            unique.append(limitation)
    return unique


# --- default model seam -------------------------------------------------


def _complete_with_model(model: ModelConnection, prompt: str) -> str:
    """The default model call: one JSON-mode completion over the endpoint.

    Kept behind the ``complete`` seam so tests never reach the network.
    Provider branching, transient-error retry, and the litellm call itself all
    live on ``ModelConnection.complete``, so this is just a fixed-``_SYSTEM``
    binding.
    """
    return model.complete(_SYSTEM, prompt)
