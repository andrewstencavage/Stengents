"""The stable Workout Review entry point: ``review_workout(workout_id)``.

Independent of the ``kiln_coach`` chat agent. It reads one finished Session by
its stable id from ``kiln_client`` and resolves the development-time model from
``model_source``, then hands both to generation. Generation is **stubbed** here
(a schema-valid empty review) — the real evidence-backed generation is ticket
#23; this ticket only establishes the seams and the contract return type.
"""

from __future__ import annotations

from typing import Callable

from farm_system.kiln_coach import kiln_client
from stengents.utilities.model_source import ModelConnection, resolve_model

from .contract import Limitation, WorkoutReview

# The review capability wants the same larger tool-calling model kiln_coach uses.
DEFAULT_MODEL_NAME = "qwen2.5:7b-8k"

# The seam onto Kiln: by-id fetch of one finished Session, injectable for tests.
Fetch = Callable[[str], "dict | None"]


def review_workout(
    workout_id: str,
    *,
    fetch: Fetch = kiln_client.fetch_workout,
    model: ModelConnection | None = None,
) -> WorkoutReview:
    """Review one finished Kiln Session, addressed by its stable ``workout_id``.

    Returns a schema-valid ``WorkoutReview`` in every case: an unknown id yields
    a review whose only content is a ``missing_data`` Limitation, never an
    exception. ``fetch`` and ``model`` are injectable so a caller (or a test)
    can supply frozen data and skip the endpoint.
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
    connection = model or resolve_model(DEFAULT_MODEL_NAME)
    return _generate_review(session, connection)


def _generate_review(session: dict, model: ModelConnection) -> WorkoutReview:
    """Produce the review for a fetched Session.

    STUB (ticket #23 replaces this): returns a schema-valid, empty-but-valid
    review that names the subject and asserts nothing. It deliberately makes no
    claims and cites no evidence, so it can never state anything unsupported.
    ``model`` is wired through but unused until real generation lands.
    """
    return WorkoutReview(workout_id=session["id"], summary="")
