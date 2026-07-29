"""The stable Workout Review entry point: ``review_workout(workout_id)``.

Independent of the ``kiln_coach`` chat agent. It reads one finished Session by
its stable id from ``kiln_client``, selects each performed exercise's comparison
history (#19), resolves the development-time model from ``model_source``, and
asks it for an evidence-backed review — decoded strictly into the #17 contract.

Generation is decomposed (#60, Wayfinder map #56), not one-shot: a per-exercise
EXTRACTION call (``extract``, full attention on one exercise, no competition
from the others) proposes candidate findings, and one SYNTHESIS call
(``synthesize``) selects the best <=3 and writes the final review. Two
independent one-shot fixes for this — a prompt nudge, then a deterministic
hint — each regressed the corpus by displacing something the model used to get
right elsewhere in the same completion; decomposition was chosen because it
gives each exercise its own uncontested attention instead of asking one
completion to do everything at once. ``_needs_extraction`` skips a call
entirely for exercises with nothing to report — the latency lever, since the
local endpoint is compute-bound (concurrent calls barely beat sequential ones),
so fewer tokens processed is the only thing that meaningfully helps.

Grounding is structural, not trusted, and unchanged by the decomposition: every
candidate ``Evidence`` row is built here from real subject/history data and
handed to the model *numbered*; extraction and synthesis both cite into that
same shared, numbered pool, so a citation from either phase resolves through
the same ``_resolve_evidence``/``Grounding`` with no new code. Any row cited
resolves back to real Kiln data by construction.

Four seams keep the whole pipeline testable offline, analogous to the existing
``fetch``: ``fetch`` (subject by id), ``fetch_history`` (the finished-Session
pool), ``model`` (the resolved connection), and ``complete`` (one model call,
reused for every extraction and the synthesis call). A test injects a fake
``complete`` returning canned structured replies and never touches the
endpoint.
"""

from __future__ import annotations

import json
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

# The review capability wants the same larger tool-calling model kiln_coach uses.
DEFAULT_MODEL_NAME = "qwen2.5:7b-8k"

# Comparison history is capped at the 10 most recent prior instances (#19).
HISTORY_CAP = 10

# Total prior-Session SetEvidence rows across the whole prompt, regardless of how
# many exercises carry history. HISTORY_CAP alone bounds depth per exercise, not
# the sum across exercises — a session with several exercises each at full depth
# can still put ~100 near-identical evidence rows in front of the model. That
# volume causes it to lose track of the schema it was asked for earlier in the
# prompt and just pattern-complete "more of the same" (a lost-in-the-middle
# failure, not a context-length one: the failing prompt used under half the
# model's context window). Subject evidence is never trimmed by this cap; only
# history rows are, round-robined whole-prior-instance-at-a-time across
# exercises so no single exercise's cap starves the others and no prior's sets
# are cut mid-instance.
MAX_HISTORY_EVIDENCE = 24

# The seam onto Kiln: by-id fetch of one finished Session, injectable for tests.
Fetch = Callable[[str], "dict | None"]
# The seam onto the finished-Session pool comparison history is selected from.
FetchHistory = Callable[..., "list[dict]"]
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
    model: ModelConnection | None = None,
    complete: Complete | None = None,
) -> WorkoutReview:
    """Review one finished Kiln Session, addressed by its stable ``workout_id``.

    Returns a schema-valid ``WorkoutReview`` in every case: an unknown id yields
    a review whose only content is a ``missing_data`` Limitation, never an
    exception. ``fetch``, ``fetch_history``, ``model``, and ``complete`` are all
    injectable so a caller (or a test) can supply frozen data and skip the
    endpoint entirely.
    """
    session = fetch(workout_id)
    if session is None:
        return WorkoutReview(
            workout_id=workout_id,
            summary="",
            limitations=[
                Limitation(
                    kind="missing_data",
                    detail=f"No finished Session with id {workout_id!r}.",
                )
            ],
        )
    pool = kiln_client.finished_sessions(fetch_history())
    connection = model or resolve_model(DEFAULT_MODEL_NAME)
    return _generate_review(session, pool, connection, complete or _complete_with_model)


def _generate_review(
    session: dict,
    pool: list[dict],
    model: ModelConnection,
    complete: Complete,
) -> WorkoutReview:
    """Produce the review for a fetched Session against its comparison history.

    Selects per-exercise history (#19), builds grounded candidate Evidence,
    partitions it per exercise (skipping partitions with nothing to report),
    extracts each partition independently, then synthesizes the final review.
    Any failure to reach or parse the model degrades to a factual, claim-free
    review (summary + acknowledged limitations) rather than raising.
    """
    selected = select_comparison_history(session, pool)
    candidates = _candidate_evidence(session, selected)
    history_limits = _history_limitations(selected)
    grounding = review_grounding(session, selected)
    malformed = _malformed_set_notes(session)
    skipped = _skipped_exercises(session)

    partitions, findings = partition_session(session, selected, candidates, malformed)
    for partition in partitions:
        findings.extend(extract(partition, candidates, selected, malformed, skipped, model, complete))

    return synthesize(session, findings, candidates, selected, grounding, history_limits, model, complete)


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


def review_grounding(session: dict, selected: dict[str, list[dict]]) -> Grounding:
    """Grounding over the pool a review's evidence must resolve against.

    That pool is the subject Session plus every prior Session its comparison
    history was drawn from — the exact set every candidate is built from, so
    every candidate (and every cited row) grounds against it. Shared with the
    corpus test so production and test can't drift on what that pool is.
    """
    pool = [session, *(prior["session"] for priors in selected.values() for prior in priors)]
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


# --- grounded candidate evidence ----------------------------------------


# A prompt "slot": one Evidence row, or (history only) a whole prior instance's
# SetEvidence rows grouped and displayed as a single compact, single-id citable
# unit. Grouping never touches contract.py — a cited group slot still expands to
# however many real, individually-grounded SetEvidence rows it contains; only
# the number of *visible* rows the model has to read shrinks.
Candidate = Evidence | list[SetEvidence]


def _candidate_evidence(session: dict, selected: dict[str, list[dict]]) -> list[Candidate]:
    """Build every citable prompt slot from real subject and history data.

    The model never emits raw evidence values — it selects from this list by its
    1-based position, so every cited row is real by construction. Rows that do
    not satisfy the closed contract vocabularies (e.g. an unrecognised
    ``loadType``) are silently skipped rather than crashing generation.
    """
    workout_id = session["id"]
    candidates: list[Candidate] = []

    # Subject session-level facts.
    candidates.extend(_session_facts(workout_id, session))

    # Subject sets and activity-level facts. Kept ungrouped (one row per set, as
    # before) — subject evidence is typically small and the data-quality checks
    # (0.3.0/0.4.0) need to point at one specific subject set, not a group.
    for activity in session.get("activities") or []:
        name = activity.get("name")
        candidates.extend(_set_rows(workout_id, name, activity))
        candidates.extend(_activity_facts(workout_id, name, activity))

    # History sets — what progression claims compare against. Grouped whole prior
    # instance at a time: the same information a flat set of rows would carry, in
    # far fewer visible rows (#see MAX_HISTORY_EVIDENCE / 0.5.0's cap).
    candidates.extend(_history_candidate_groups(selected))

    return candidates


def _history_candidate_groups(selected: dict[str, list[dict]]) -> list[list[SetEvidence]]:
    """Prior-Session set rows, grouped whole-instance-per-slot, round-robined per
    exercise, and capped in total underlying rows (not slot count).

    Builds each exercise's priors (already newest-first, capped at HISTORY_CAP)
    into whole per-instance row groups, then takes one whole prior instance at a
    time from each exercise in turn — never splitting a prior's own sets across
    the cutoff — until MAX_HISTORY_EVIDENCE real SetEvidence rows are reached.
    Round-robining instead of exhausting one exercise before the next keeps every
    exercise represented even when the total is capped; a session under budget is
    unaffected. Each returned group becomes exactly one CANDIDATE EVIDENCE line
    in the prompt regardless of how many sets it holds — the total real evidence
    budget is unchanged from the flat form, only how many chunks it costs to show.
    """
    queues: dict[str, list[list[SetEvidence]]] = {}
    for name, priors in selected.items():
        instances: list[list[SetEvidence]] = []
        for prior in priors:
            prior_id = prior["session"].get("id")
            if prior_id is None:
                continue
            rows = _set_rows(prior_id, name, prior["activity"])
            if rows:
                instances.append(rows)
        if instances:
            queues[name] = instances

    groups: list[list[SetEvidence]] = []
    total_rows = 0
    pointers = {name: 0 for name in queues}
    active = list(queues.keys())
    while active and total_rows < MAX_HISTORY_EVIDENCE:
        for name in list(active):
            position = pointers[name]
            if position >= len(queues[name]):
                active.remove(name)
                continue
            instance = queues[name][position]
            if groups and total_rows + len(instance) > MAX_HISTORY_EVIDENCE:
                active.remove(name)
                continue
            groups.append(instance)
            total_rows += len(instance)
            pointers[name] = position + 1
    return groups


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


def _needs_extraction(name: str, activity: dict, priors: list[dict], malformed: dict[str, int]) -> bool:
    """Whether this exercise partition needs a real extraction call, or can be
    skipped entirely — the primary latency lever for decomposed generation
    (#60). A per-exercise extraction call costs real GPU time regardless of how
    many partitions run concurrently (empirically: 8 concurrent calls barely
    beat 8 sequential ones — the local endpoint is compute-bound, not
    round-trip-bound), so the only lever that meaningfully cuts latency is
    processing fewer tokens in total, not rearranging the same tokens.

    An exercise is skippable — genuinely has nothing to report — only when
    ALL of: no note, not flagged skipped, not among the malformed-set exercises,
    its own performed sets are unchanged from first to last, and (when a prior
    exists) its last set matches the most recent prior's last set. Any one of
    these being true means a real signal might be there, so the partition still
    gets a full extraction call — this errs toward calling the model on
    borderline cases rather than risking a silent miss.
    """
    if activity.get("note"):
        return True
    if activity.get("skipped"):
        return True
    if name in malformed:
        return True
    rows = performed_sets(activity)
    if len(rows) >= 2 and (rows[0]["reps"] != rows[-1]["reps"] or rows[0]["load"] != rows[-1]["load"]):
        return True
    if priors and rows:
        prior_rows = performed_sets(priors[0]["activity"])
        if prior_rows and (rows[-1]["reps"] != prior_rows[-1]["reps"] or rows[-1]["load"] != prior_rows[-1]["load"]):
            return True
    return False


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


# --- partitioning (#60) --------------------------------------------------


@dataclass(frozen=True)
class Partition:
    """One extraction unit: a subject exercise that needs a real look.

    ``candidate_ids`` are 1-based positions into the same shared candidate list
    ``_candidate_evidence`` builds — not a private numbering, so a citation from
    any partition resolves through the unmodified ``_resolve_evidence``.
    """

    key: str  # exercise name
    candidate_ids: tuple[int, ...]


def partition_session(
    session: dict,
    selected: dict[str, list[dict]],
    candidates: list[Candidate],
    malformed: dict[str, int],
) -> tuple[list[Partition], list["ExtractionFinding"]]:
    """One partition per subject exercise that needs a real look.

    No session-level partition: session facts (date/type/minutes/feel) never
    appear in any corpus case's required evidence, and live testing found
    extracting "observations" from them anyway (e.g. "the workout was on
    2026-07-25") produced generic, low-value findings that crowded out
    substantive per-exercise ones during synthesis's selection — synthesis
    still sees the session facts directly (``build_synthesis_prompt``) for
    writing the summary, so nothing is lost.

    An exercise ``_needs_extraction`` rules out gets no partition — no wasted
    extraction call — but it is NOT simply dropped: it still gets a
    deterministic baseline "performance" finding (``_baseline_finding``, no
    model call) describing what was performed, citing its own real sets. Live
    testing first tried dropping skipped exercises entirely and it broke the
    corpus badly (``required_detail_recall`` 0.722 -> 0.4375): "nothing notable
    happened" is not the same as "nothing worth reporting" — even an
    unremarkable exercise's basic performance facts are exactly what
    ``required_detail_recall`` checks for, and the one-shot design always had
    them because it saw every exercise in one prompt. The baseline finding
    restores that coverage without spending a model call on exercises that
    have nothing that needs judgment.
    """
    by_exercise: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, list):
            by_exercise.setdefault(candidate[0].exercise, []).append(index)
            continue
        exercise = getattr(candidate, "exercise", None)
        if exercise is not None:
            by_exercise.setdefault(exercise, []).append(index)

    activities_by_name = {a.get("name"): a for a in session.get("activities") or [] if a.get("name")}

    partitions: list[Partition] = []
    baseline_findings: list[ExtractionFinding] = []
    for name in selected:  # selected preserves the subject's own exercise order
        activity = activities_by_name.get(name, {})
        if _needs_extraction(name, activity, selected[name], malformed):
            partitions.append(Partition(key=name, candidate_ids=tuple(by_exercise.get(name, ()))))
        else:
            baseline = _baseline_finding(name, activity, by_exercise.get(name, ()), candidates)
            if baseline is not None:
                baseline_findings.append(baseline)
    return partitions, baseline_findings


# --- extraction -----------------------------------------------------------


_SYSTEM = (
    "You are a strength-training review generator. Output STRICT JSON only — "
    "no prose outside the JSON."
)


@dataclass(frozen=True)
class ExtractionFinding:
    """One partition's proposed observation or limitation, citing shared-pool ids.

    Looser than the real Observation/Limitation contract types — extraction can
    propose as many findings as it finds and can be wrong; synthesis (via the
    existing ``_resolve_evidence``/``Grounding``/vocabulary checks) is where
    invalid citations and out-of-vocabulary values get dropped and the real,
    validated contract objects get built.
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


def _baseline_finding(
    name: str, activity: dict, candidate_ids: list[int], candidates: list[Candidate]
) -> ExtractionFinding | None:
    """A deterministic 'performance' finding for an exercise ``_needs_extraction``
    skipped — no model call, just a plain factual recap citing every one of the
    exercise's own performed-set candidates. ``None`` when the exercise offers
    no citable sets at all (e.g. a freeform activity with no ``performedSets``).
    """
    set_ids = [i for i in candidate_ids if isinstance(candidates[i - 1], SetEvidence)]
    if not set_ids:
        return None
    collapsed = kiln_client._collapse_sets(activity.get("performedSets") or [])
    return ExtractionFinding(
        partition_key=name,
        finding_kind="observation",
        category="performance",
        observation_kind="fact",
        confidence="firm",
        claim=f"{name}: {collapsed}.",
        evidence_ids=tuple(set_ids),
    )


_EXTRACTION_INSTRUCTIONS = (
    "You are reviewing ONE exercise from a workout session in isolation. List every "
    "fact worth noting about ONLY the evidence below; do not worry about picking the "
    "'best' ones or staying under any limit — a later step selects among every "
    "exercise's findings together.\n"
    "Rules:\n"
    "- Cite evidence ONLY by the integer ids listed under EVIDENCE. Never invent sets, "
    "numbers, or facts.\n"
    "- Use category 'progression' ONLY when COMPARISON HISTORY is shown below, and cite "
    "at least one of its sets for it.\n"
    "- Only add a 'limitation' when this prompt EXPLICITLY tells you to below (a DATA "
    "QUALITY NOTE naming a required kind) — never infer a data-quality problem on your "
    "own judgment (e.g. from a note's tone or wording). Most exercises have no "
    "limitation at all; that is the normal, expected case, not a gap to fill.\n"
    "Output JSON shape:\n"
    '{"observations": [{"kind": "fact"|"inference", "confidence": "firm"|"tentative",'
    ' "category": "performance"|"progression"|"adherence"|"data_quality",'
    ' "claim": str, "evidence_ids": [int, ...]}],'
    ' "limitations": [{"kind": "missing_data"|"malformed_data",'
    ' "detail": str}]}'
)


def _render_candidate(candidate: Candidate) -> str:
    """Render one EVIDENCE line: a single row, or a whole grouped instance."""
    if isinstance(candidate, list):
        first = candidate[0]
        return json.dumps(
            {
                "kind": "history_sets",
                "workout_id": first.workout_id,
                "exercise": first.exercise,
                "sets": [{"reps": row.reps, "load": row.load, "loadType": row.loadType} for row in candidate],
            }
        )
    return json.dumps(candidate.model_dump())


def build_extraction_prompt(
    partition: Partition,
    candidates: list[Candidate],
    selected: dict[str, list[dict]],
    malformed: dict[str, int],
    skipped: set[str],
) -> str:
    """One partition's whole prompt: its comparison history, any deterministic
    DATA QUALITY NOTE (malformed/skipped) the model must react to, and its own
    slice of the shared candidate pool."""
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
        lines.append(f"COMPARISON HISTORY (newest first): {json.dumps(summary)}")
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
        lines.append(f"{index}: {_render_candidate(candidates[index - 1])}")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def extract(
    partition: Partition,
    candidates: list[Candidate],
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


# --- synthesis --------------------------------------------------------------


_SYNTHESIS_INSTRUCTIONS = (
    "Below are candidate observations extracted independently per exercise/session-facts, "
    "each already citing real evidence ids. Select the best at most 3 (prefer variety across "
    "categories and exercises over redundancy) and write a short factual summary of the whole "
    "session.\n"
    "Output JSON shape:\n"
    '{"summary": str, "selected_observation_indexes": [int, ...]}\n'
    "selected_observation_indexes are 1-based positions into the OBSERVATION FINDINGS list below."
)


def build_synthesis_prompt(session: dict, observation_findings: list[ExtractionFinding]) -> str:
    """The one synthesis prompt: session facts (for the summary line) plus every
    extracted observation finding, numbered for the model to select among."""
    lines: list[str] = [_SYNTHESIS_INSTRUCTIONS, "", "SUBJECT SESSION:"]
    lines.append(
        json.dumps(
            {
                "date": (session.get("date") or "")[:10],
                "workout": session.get("workoutName"),
                "type": session.get("type"),
                "minutes": session.get("minutes"),
                "feel": session.get("feel"),
            }
        )
    )
    lines.append("")
    lines.append("OBSERVATION FINDINGS:")
    for index, finding in enumerate(observation_findings, start=1):
        lines.append(f"{index}: [{finding.partition_key}] {finding.category} | {finding.claim}")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


def synthesize(
    session: dict,
    findings: list[ExtractionFinding],
    candidates: list[Candidate],
    selected: dict[str, list[dict]],
    grounding: Grounding,
    history_limits: list[Limitation],
    model: ModelConnection,
    complete: Complete,
) -> WorkoutReview:
    """Select among extraction findings and write the final review.

    Selected observations' ``evidence_ids`` resolve through the same,
    unmodified ``_resolve_evidence``/``Grounding`` extraction candidates ground
    against — decomposition adds no new grounding code. Any failure to reach or
    parse the model degrades to a factual, claim-free review, same as before.
    """
    workout_id = session["id"]
    prior_ids = {prior["session"].get("id") for priors in selected.values() for prior in priors}
    observation_findings = [f for f in findings if f.finding_kind == "observation"]
    limitation_findings = [f for f in findings if f.finding_kind == "limitation"]

    summary = ""
    selected_indexes: list[int] = []
    if observation_findings:
        prompt = build_synthesis_prompt(session, observation_findings)
        try:
            parsed = _extract_json(complete(model, prompt))
        except Exception:  # noqa: BLE001
            parsed = None
        if isinstance(parsed, dict):
            summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
            selected_indexes = [
                i
                for i in parsed.get("selected_observation_indexes", [])
                if isinstance(i, int) and not isinstance(i, bool)
            ]

    observations: list[Observation] = []
    seen_indexes: set[int] = set()
    for index in selected_indexes:
        if index in seen_indexes or not (1 <= index <= len(observation_findings)):
            continue
        seen_indexes.add(index)
        finding = observation_findings[index - 1]
        if (
            finding.category not in _CATEGORIES
            or finding.observation_kind not in _KINDS
            or finding.confidence not in _CONFIDENCES
            or not isinstance(finding.claim, str)
            or not finding.claim.strip()
        ):
            continue
        evidence = _resolve_evidence(list(finding.evidence_ids), candidates, grounding)
        if not evidence:
            continue
        if finding.category == "progression" and not _cites_prior_set(evidence, prior_ids):
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
        if len(observations) == 3:
            break

    model_limits = _decode_limitations(
        [{"kind": f.limitation_kind, "detail": f.limitation_detail} for f in limitation_findings]
    )

    if not summary:
        summary = _factual_summary(session)

    limitations = _dedupe_limitations(history_limits + model_limits)
    return WorkoutReview(workout_id=workout_id, summary=summary, observations=observations, limitations=limitations)


# --- decode / ground ----------------------------------------------------


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


def _resolve_evidence(
    raw_ids: object, candidates: list[Candidate], grounding: Grounding
) -> list[Evidence]:
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
            rows = candidate if isinstance(candidate, list) else [candidate]
            if not all(grounding.resolves(row) for row in rows):
                continue  # guard: never surface a group with any row that doesn't ground verbatim.
            seen.add(index)
            resolved.extend(rows)
    return resolved


def _cites_prior_set(evidence: list[Evidence], prior_ids: set) -> bool:
    """True when some cited row is a performed set from a prior (history) Session."""
    return any(isinstance(row, SetEvidence) and row.workout_id in prior_ids for row in evidence)


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


def _factual_summary(session: dict) -> str:
    """A deterministic factual recap, used when the model supplies no summary."""
    activities = session.get("activities") or []
    performed = [a for a in activities if a.get("performedSets")]
    parts = [session.get("workoutName") or "Session"]
    date = (session.get("date") or "")[:10]
    if date:
        parts.append(f"on {date}")
    tail = [f"{len(performed)} exercise{'s' if len(performed) != 1 else ''} performed"]
    if session.get("minutes") is not None:
        tail.append(f"{session['minutes']} min")
    if session.get("feel"):
        tail.append(f"felt {session['feel']}")
    return f"{' '.join(parts)}: {', '.join(tail)}."


# --- default model seam -------------------------------------------------


def _complete_with_model(model: ModelConnection, prompt: str) -> str:
    """The default model call: one JSON-mode completion over the endpoint.

    Kept behind the ``complete`` seam so tests never reach the network. Uses the
    same OpenAI-compatible connection ``model_source`` resolves for kiln_coach.
    """
    from litellm import completion

    response = completion(
        model=f"openai/{model.name}",
        api_base=f"{model.base_url.rstrip('/')}/v1",
        api_key=model.api_key,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response["choices"][0]["message"]["content"]
