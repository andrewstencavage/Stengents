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
    # The pool every candidate and cited row must ground against: the subject plus
    # each prior Session comparison history was drawn from.
    grounding = Grounding(
        [session, *(prior["session"] for priors in selected.values() for prior in priors)]
    )

    prompt = _build_prompt(session, selected, candidates)
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


def _candidate_evidence(session: dict, selected: dict[str, list[dict]]) -> list[Evidence]:
    """Build every citable Evidence row from real subject and history data.

    The model never emits raw evidence values — it selects from this list by its
    1-based position, so every cited row is real by construction. Rows that do
    not satisfy the closed contract vocabularies (e.g. an unrecognised
    ``loadType``) are silently skipped rather than crashing generation.
    """
    workout_id = session["id"]
    candidates: list[Evidence] = []

    # Subject session-level facts.
    candidates.extend(_session_facts(workout_id, session))

    # Subject sets and activity-level facts.
    for activity in session.get("activities") or []:
        name = activity.get("name")
        candidates.extend(_set_rows(workout_id, name, activity))
        candidates.extend(_activity_facts(workout_id, name, activity))

    # History sets — what progression claims compare against.
    for name, priors in selected.items():
        for prior in priors:
            prior_id = prior["session"].get("id")
            if prior_id is None:
                continue
            candidates.extend(_set_rows(prior_id, name, prior["activity"]))

    return candidates


def _set_rows(workout_id: str, exercise: str | None, activity: dict) -> list[Evidence]:
    rows: list[Evidence] = []
    for row in performed_sets(activity):
        try:
            rows.append(SetEvidence(workout_id=workout_id, exercise=exercise, **row))
        except Exception:  # noqa: BLE001 — skip a malformed/out-of-vocabulary set.
            continue
    return rows


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
    "- 'summary' is a short factual recap of what was performed.\n"
    "Output JSON shape:\n"
    '{"summary": str,'
    ' "observations": [{"kind": "fact"|"inference", "confidence": "firm"|"tentative",'
    ' "category": "performance"|"progression"|"adherence"|"data_quality",'
    ' "claim": str, "evidence_ids": [int, ...]}],'
    ' "limitations": [{"kind": "insufficient_history"|"missing_data"|"malformed_data"|"conflicting_data",'
    ' "detail": str}]}'
)


def _build_prompt(session: dict, selected: dict[str, list[dict]], candidates: list[Evidence]) -> str:
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
    lines.append("CANDIDATE EVIDENCE (id: row):")
    for index, evidence in enumerate(candidates, start=1):
        lines.append(f"{index}: {json.dumps(evidence.model_dump())}")
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
    candidates: list[Evidence],
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
    raw_ids: object, candidates: list[Evidence], grounding: Grounding
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
            if not grounding.resolves(candidate):
                continue  # guard: never surface a row that doesn't ground verbatim.
            seen.add(index)
            resolved.append(candidate)
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
