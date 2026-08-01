"""The Auto-replan decision capability (issue #64; parent issue #63) and its
MCP adapter (issue #66).

Public surface: the ``decide_auto_replan`` pure entry point, the output
contract types, and ``run_auto_replan`` — the MCP-backed adapter that fetches
inputs, calls ``decide_auto_replan``, and executes its writes against a live
Kiln instance (see :mod:`.kiln_adapter`).
"""

from .contract import (
    DraftActivity,
    DraftWorkout,
    PlanDraft,
    PlanTemplateContent,
    PrescribedSet,
    ReplanDecision,
    TemplateUpdate,
    to_kiln_json,
)
from .decision import decide_auto_replan
from .kiln_adapter import run_auto_replan

__all__ = [
    "decide_auto_replan",
    "run_auto_replan",
    "ReplanDecision",
    "PlanDraft",
    "DraftWorkout",
    "DraftActivity",
    "PrescribedSet",
    "PlanTemplateContent",
    "TemplateUpdate",
    "to_kiln_json",
]
