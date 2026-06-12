from .compiler import RRTSkillCompiler, SkillCompilationError, SkillCompiler
from .models import (
    PreparedSkillExecution,
    SkillCall,
    SkillExecutionResult,
    SkillExecutionStatus,
    SkillPlan,
    SkillSpec,
    SkillValidationResult,
    ValidationIssue,
)
from .pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT, build_pack_grocery_skill_registry
from .registry import SkillRegistry
from .validation import PackGrocerySkillPlanValidator, SkillPlanValidator


def __getattr__(name):
    if name in ("RRTSkillExecutor", "SkillExecutor"):
        from .executor import RRTSkillExecutor, SkillExecutor

        return {"RRTSkillExecutor": RRTSkillExecutor, "SkillExecutor": SkillExecutor}[name]
    raise AttributeError(name)


__all__ = [
    "PreparedSkillExecution",
    "PackGrocerySkillPlanValidator",
    "PUT_OBJECT_IN_CONTAINER",
    "RRTSkillCompiler",
    "SkillCall",
    "SkillCompilationError",
    "SkillCompiler",
    "SkillExecutionResult",
    "SkillExecutionStatus",
    "SkillPlan",
    "SkillPlanValidator",
    "SkillRegistry",
    "SkillSpec",
    "SkillValidationResult",
    "ValidationIssue",
    "WAIT",
    "build_pack_grocery_skill_registry",
    "RRTSkillExecutor",
    "SkillExecutor",
]
