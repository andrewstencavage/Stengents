"""The Auto-replan decision capability (issue #64; parent issue #63).

Public surface: the ``decide_auto_replan`` pure entry point and the output
contract types.
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

__all__ = [
    "decide_auto_replan",
    "ReplanDecision",
    "PlanDraft",
    "DraftWorkout",
    "DraftActivity",
    "PrescribedSet",
    "PlanTemplateContent",
    "TemplateUpdate",
    "to_kiln_json",
]
