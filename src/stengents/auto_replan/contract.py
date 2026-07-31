"""The Auto-replan decision output contract (issue #64; parent issue #63).

Pure content models: a ``ReplanDecision`` carries which Plan Template applies
(``selected_template``, ``None`` when nothing matched), a fully-populated
``draft`` Plan (``None`` alongside a ``None`` template), whether to ``activate``
it, and any ``template_updates`` to persist so a template stays fresh from
recent Session history (issue #63 user story #4, "kept up to date
automatically"). Nothing operational lives here — this module carries content
only, no MCP client, no model, no I/O. See ``docs/adr/0005-kiln-coach-moves-to-
mcp.md`` for the ADR that motivates this repo's move to MCP generally; the
Auto-replan decision/contract shape itself is specified in issue #63/#64, not
that ADR.

``draft`` mirrors Kiln's ``planDraftInputSchema`` shape exactly (weekday/
workout/activity structure, ``schemaVersion`` pinned to ``1``); this is the
acceptance criterion that the draft this contract carries "validates against
Kiln's ``planDraftInputSchema`` shape" (kiln issue #170,
``mcp/schemas.js``). ``template_updates`` reuses the identical
weekday/workout/activity shape under ``planTemplateInputSchema`` — a named,
dateless candidate week instead of one tied to ``weekStart`` — since both are
documented (kiln ``docs/adr/0001-reintroduce-plan-templates.md``) as sharing
one structural schema.

**Documented interface assumption** (no live Kiln connection to verify
against, kiln issue andrewstencavage/kiln#170 not yet merged): a Plan
Template's *content* — what this module reads as input and writes as
``template_updates`` — is modeled here in the same structured shape
``create_plan_template``/``create_plan_draft`` accept (``workoutInputSchema``:
``exerciseId`` + ``prescribedSets``), not the simpler display shape
``get_plan_template``/``list_plan_templates`` return (``name``/``spec``/
``restSeconds``). The structured shape is the only one this pure function can
build a schema-valid draft or a schema-valid template upsert from without an
exercise catalog to invert a free-text ``spec`` back into ``exerciseId`` +
numeric ``prescribedSets``. Reconciling this against what a live
``list_plan_templates``/``get_plan_template`` call actually returns is left to
the not-yet-built MCP-reading adapter (issue #63's "thin adapter"), which may
need its own structured-content source. Flagged for human review.

A second documented assumption: ``exerciseId`` is treated as equal to the
exercise's display name (``Session.activities[].name``) for the purpose of
matching a performed Session activity back to a template activity, since this
pure function has no exercise catalog to resolve a real id from a name (or
vice versa).

A third documented risk — the raw Session shape this module (and its
fixtures) read may not survive the coach server's move to a pure MCP client —
is flagged in :mod:`.decision`, next to the code it affects
(``sync_template_update``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Kiln's workoutInputSchema.weekday/type enums, passed through verbatim (mcp/schemas.js).
Weekday = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WorkoutType = Literal["strength", "run", "row", "rest"]
# Kiln's setSchema.unit enum, passed through verbatim (mcp/schemas.js).
LoadUnit = Literal["lb", "kg", "plate", "sec", "m", "mi"]

WEEKDAY_ORDER: tuple[Weekday, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class _Content(BaseModel):
    """A frozen, strict content model — no extra fields, no mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PrescribedSet(_Content):
    """One prescribed set — mirrors Kiln's ``setSchema`` exactly."""

    id: str = Field(min_length=1)
    reps: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0)
    unit: LoadUnit | None = None


class DraftActivity(_Content):
    """One structured activity — mirrors Kiln's ``workoutInputSchema.activities``
    entry exactly (``exerciseId`` + ``prescribedSets``, not a display ``spec``)."""

    id: str = Field(min_length=1)
    exerciseId: str = Field(min_length=1)
    restSeconds: int = Field(ge=0)
    prescribedSets: list[PrescribedSet] = Field(min_length=1)


class DraftWorkout(_Content):
    """One weekday's workout — mirrors Kiln's ``workoutInputSchema`` exactly.
    A rest day is a ``type: "rest"`` workout with no activities."""

    weekday: Weekday
    name: str = Field(min_length=1)
    type: WorkoutType
    estimatedMinutes: int | None = Field(default=None, ge=0)
    activities: list[DraftActivity] = Field(default_factory=list)


def to_kiln_json(model: BaseModel) -> dict:
    """Serialize a contract model to the exact JSON shape Kiln's zod schemas
    accept: optional fields left unset are omitted entirely, never sent as an
    explicit ``null``.

    Verified directly against the real schema (kiln issue #170,
    ``mcp/schemas.js``, branch ``issue-170-plan-template-mcp-tools``): zod's
    ``.optional()`` (without a paired ``.nullable()``) — which is how
    ``estimatedMinutes`` and a ``PrescribedSet``'s ``value``/``unit`` are all
    declared — rejects an explicit JSON ``null`` with ``invalid_type``.
    Pydantic's default ``model_dump()`` renders an unset optional field as
    ``null``, which round-trips through ``json.dumps`` into exactly the shape
    zod rejects, so every caller handing this contract's content to Kiln
    (directly, or through the not-yet-built MCP adapter) must serialize
    through this helper (or the equivalent ``exclude_none=True, mode="json"``)
    rather than a bare ``model_dump()``.
    """
    return model.model_dump(exclude_none=True, mode="json")


class PlanDraft(_Content):
    """A fully-populated draft Plan — mirrors Kiln's ``planDraftInputSchema``
    exactly, so ``create_plan_draft`` can accept it unmodified once serialized
    through :func:`to_kiln_json`."""

    schemaVersion: Literal[1] = 1
    weekStart: str
    weekFocus: str = Field(min_length=1)
    workouts: list[DraftWorkout] = Field(min_length=1)


class PlanTemplateContent(_Content):
    """A named, dateless candidate week — mirrors Kiln's
    ``planTemplateInputSchema`` exactly, so ``create_plan_template``'s upsert
    can accept it unmodified once serialized through :func:`to_kiln_json`."""

    schemaVersion: Literal[1] = 1
    # Kiln's planTemplateInputSchema bounds both at .trim().min(1).max(200).
    name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=200)
    weekFocus: str = Field(min_length=1)
    workouts: list[DraftWorkout] = Field(min_length=1)


class TemplateUpdate(_Content):
    """One Plan Template's revised content to persist via ``create_plan_template``
    (an upsert keyed by ``template.name``), plus a structured reason a captain
    can sanity-check later (issue #63 captain user story #7)."""

    template: PlanTemplateContent
    reason: str = Field(min_length=1)


class ReplanDecision(_Content):
    """The Auto-replan decision: which Plan Template applies, the draft it
    produced, whether to activate it, and any template content to persist.

    ``selected_template`` and ``draft`` are ``None`` together — no template
    matched, so there is nothing to activate (``activate`` is always ``False``
    in that case). ``activate`` is otherwise ``True`` whenever a schema-valid
    draft was built: per ADR-0002 (kiln), Auto-replan adds no blocking
    guardrail beyond schema validation — Kiln's Plan history is the safety net,
    not a review gate.
    """

    selected_template: str | None
    draft: PlanDraft | None
    activate: bool
    template_updates: list[TemplateUpdate] = Field(default_factory=list)
    reason: str = Field(min_length=1)
