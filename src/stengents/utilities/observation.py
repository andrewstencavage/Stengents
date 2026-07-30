"""Bounding logged payloads to a scalar-and-shape summary.

``reduce_value`` keeps the useful signal — scalar field values, collection
shapes — while never storing a large payload verbatim. It is the shared
primitive under turn logging and the harness's tool-event capture, so both can
record what an agent read or wrote without ever persisting an unbounded blob.

``build_call_record`` sits beside it: the other half of the pair ADR-0003
names as the reusable seam. It is the one shape both shells build around a
single named call (reduced args, a duration, and an outcome-tagged result or
error) — each shell keeps its own timing measurement and its own extra fields
layered on top, since *how* a call is timed and attributed differs (a
synchronous wrap in the harness vs. a before/after ADK callback pair in Turn
logging).
"""

from __future__ import annotations

MAX_STRING = 200

_NO_RESULT = object()


def reduce_value(value: object, *, depth: int = 0, max_string: int = MAX_STRING) -> object:
    """Shrink a value for logging: scalars verbatim, collections collapsed to counts.

    Keeps the useful signal (scalar field values, shapes) while never storing a
    large payload. Recurses one level into a top-level dict, then collapses.
    """
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + f"…(+{len(value) - max_string})"
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float) or value is None:
        return value
    if isinstance(value, list) or isinstance(value, tuple):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        if depth >= 1:
            return f"{{{len(value)} keys}}"
        return {str(key): reduce_value(item, depth=depth + 1, max_string=max_string) for key, item in value.items()}
    return reduce_value(str(value), depth=depth, max_string=max_string)


def build_call_record(
    name: str,
    args: dict,
    duration_ms: int,
    *,
    result: object = _NO_RESULT,
    error: BaseException | None = None,
) -> dict[str, object]:
    """One bounded record of a single named call: reduced args, duration, and
    an outcome-tagged result or error. Pass exactly one of ``result``/``error``.
    """
    record: dict[str, object] = {"name": name, "args": reduce_value(args), "duration_ms": duration_ms}
    if error is not None:
        record["outcome"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    else:
        record["outcome"] = "ok"
        record["result_summary"] = reduce_value(None if result is _NO_RESULT else result)
    return record
