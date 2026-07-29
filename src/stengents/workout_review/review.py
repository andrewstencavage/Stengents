"""The stable Workout Review entry point: ``review_workout(workout_id)``.

Independent of the ``kiln_coach`` chat agent. It reads one finished Session by
its stable id from ``kiln_client``, selects each performed exercise's comparison
history (#19), resolves the development-time model from ``model_source``, and
asks it for an evidence-backed review — decoded strictly into the #17 contract.

Grounding is structural, not trusted: every candidate ``Evidence`` row is built
here from real subject/history data and handed to the model *numbered*; the model
selects rows by id and claims over them, so it can never invent a set or a fact.
Any row it cites resolves back to real Kiln data by construction.

Three seams keep the whole pipeline testable offline, analogous to the existing
``fetch``: ``fetch`` (subject by id), ``fetch_history`` (the finished-Session
pool), and ``complete`` (the one model call). A test injects a fake ``complete``
returning a canned structured response and never touches the endpoint.
"""

from __future__ import annotations

import json
from typing import Callable

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
    prompts the model over it, and decodes the reply strictly into the contract.
    Any failure to reach or parse the model degrades to a factual, claim-free
    review (summary + acknowledged limitations) rather than raising.
    """
    workout_id = session["id"]
    selected = select_comparison_history(session, pool)
    candidates = _candidate_evidence(session, selected)
    history_limits = _history_limitations(selected)
    grounding = review_grounding(session, selected)
    malformed = _malformed_set_notes(session)

    prompt = _build_prompt(session, selected, candidates, malformed)
    try:
        raw = complete(model, prompt)
        parsed = _extract_json(raw)
    except Exception:  # noqa: BLE001 — any model/transport/parse failure degrades cleanly.
        parsed = None

    observations: list[Observation] = []
    model_limits: list[Limitation] = []
    summary = ""
    if isinstance(parsed, dict):
        summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
        observations = _decode_observations(parsed.get("observations"), candidates, selected, grounding)
        model_limits = _decode_limitations(parsed.get("limitations"))

    if not summary:
        summary = _factual_summary(session)

    limitations = _dedupe_limitations(history_limits + model_limits)
    return WorkoutReview(
        workout_id=workout_id,
        summary=summary,
        observations=observations[:3],
        limitations=limitations,
    )


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
    ``_build_prompt`` can surface the signal as a DATA QUALITY NOTES line. It is
    deliberately *not* a citable candidate: a synthetic 'malformed' fact could
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


# --- prompt -------------------------------------------------------------


_SYSTEM = (
    "You are a strength-training review generator. You review ONE finished "
    "workout session and output STRICT JSON only — no prose outside the JSON."
)

_INSTRUCTIONS = (
    "Write a concise, factual review of the subject session.\n"
    "Rules:\n"
    "- Cite evidence ONLY by the integer ids listed under CANDIDATE EVIDENCE. "
    "Never invent sets, numbers, or facts.\n"
    "- At most 3 observations; every observation must cite at least one evidence id.\n"
    "- Use category 'progression' ONLY for an exercise that has prior history "
    "shown below, and cite at least one prior set for it.\n"
    "- Put anything unsupported or uncertain into 'limitations', do not assert it.\n"
    "- Inspect EACH exercise's evidence for a data-quality problem and, when you "
    "find one, add a 'limitation' whose 'detail' NAMES that exact exercise:\n"
    "    * an exercise marked skipped (a 'skipped' fact is true) -> 'missing_data'\n"
    "    * a set whose load or reps is null, blank, or absent -> 'missing_data'\n"
    "    * an exercise 'note' that contradicts its performed numbers (e.g. the note "
    "says strong/easy/PR but the reps or load are lower than the compared sets) -> "
    "'conflicting_data'\n"
    "    * a value that is impossible or unparseable -> 'malformed_data'\n"
    "  One limitation per affected exercise; do NOT invent problems for exercises "
    "whose evidence is clean.\n"
    "- 'summary' is a short factual recap of what was performed.\n"
    "Output JSON shape:\n"
    '{"summary": str,'
    ' "observations": [{"kind": "fact"|"inference", "confidence": "firm"|"tentative",'
    ' "category": "performance"|"progression"|"adherence"|"data_quality",'
    ' "claim": str, "evidence_ids": [int, ...]}],'
    ' "limitations": [{"kind": "insufficient_history"|"missing_data"|"malformed_data"|"conflicting_data",'
    ' "detail": str}]}'
)


def _render_candidate(candidate: Candidate) -> str:
    """Render one CANDIDATE EVIDENCE line: a single row, or a whole grouped instance."""
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


def _build_prompt(
    session: dict,
    selected: dict[str, list[dict]],
    candidates: list[Candidate],
    malformed: dict[str, int] | None = None,
) -> str:
    lines: list[str] = [_INSTRUCTIONS, ""]
    lines.append("SUBJECT SESSION:")
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
    lines.append("COMPARISON HISTORY (per exercise, newest first):")
    for name, priors in selected.items():
        if priors:
            summary = [
                {
                    "date": (prior["session"].get("date") or "")[:10],
                    "sets": kiln_client._collapse_sets(prior["activity"].get("performedSets") or []),
                }
                for prior in priors
            ]
            lines.append(f"- {name}: {json.dumps(summary)}")
        else:
            lines.append(f"- {name}: no prior history (insufficient history)")
    lines.append("")
    if malformed:
        lines.append("DATA QUALITY NOTES (sets that could not be parsed, not shown as evidence):")
        for name, dropped in malformed.items():
            plural = "s" if dropped != 1 else ""
            lines.append(
                f"- {name}: {dropped} performed set{plural} were unreadable/malformed "
                "and were dropped."
            )
        lines.append(
            "For EACH exercise listed above you MUST add a 'malformed_data' limitation "
            "whose 'detail' NAMES that exact exercise — those sets could not be parsed "
            "and are absent from the evidence, so this note is the only signal of them."
        )
        lines.append("")
    lines.append("CANDIDATE EVIDENCE (id: row):")
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"{index}: {_render_candidate(candidate)}")
    lines.append("")
    lines.append("Return only the JSON object.")
    return "\n".join(lines)


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


def _decode_observations(
    raw_observations: object,
    candidates: list[Candidate],
    selected: dict[str, list[dict]],
    grounding: Grounding,
) -> list[Observation]:
    """Decode the model's observations, grounding each cited id to a real row.

    An observation survives only if it cites at least one candidate id that
    *grounds* verbatim in the real data (the self-detectable resolvability guard:
    a cited row that does not resolve is dropped, so no unresolvable row ever
    reaches the scored review — the same check the evaluator applies). A
    ``progression`` observation must additionally cite at least one *prior*
    (history) set — #19 forbids a progression claim without a comparison, so an
    exercise with zero priors (no prior sets to cite) can never carry one; its
    gap is acknowledged with a ``Limitation(insufficient_history)`` instead.
    """
    if not isinstance(raw_observations, list):
        return []
    prior_ids = {
        prior["session"].get("id") for priors in selected.values() for prior in priors
    }

    decoded: list[Observation] = []
    for item in raw_observations:
        if not isinstance(item, dict):
            continue
        evidence = _resolve_evidence(item.get("evidence_ids"), candidates, grounding)
        if not evidence:
            continue

        kind = item.get("kind")
        confidence = item.get("confidence")
        category = item.get("category")
        claim = item.get("claim")
        if kind not in _KINDS or confidence not in _CONFIDENCES or category not in _CATEGORIES:
            continue
        if not isinstance(claim, str) or not claim.strip():
            continue
        if category == "progression" and not _cites_prior_set(evidence, prior_ids):
            continue

        try:
            decoded.append(
                Observation(
                    kind=kind,  # type: ignore[arg-type]
                    confidence=confidence,  # type: ignore[arg-type]
                    category=category,  # type: ignore[arg-type]
                    claim=claim,
                    evidence=evidence,
                )
            )
        except Exception:  # noqa: BLE001 — drop an observation the contract rejects.
            continue
        if len(decoded) == 3:
            break
    return decoded


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
