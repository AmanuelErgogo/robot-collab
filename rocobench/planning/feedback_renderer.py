"""Grounded feedback rendering for bounded replanning."""

from typing import Any, Iterable, Optional

from rocobench.skills.models import SkillCall, SkillValidationResult

from .replanning_policy import RecoveryAction
from .retry_budget import RetryBudget
from .state_summary import StateSummary


def _call_text(call: Optional[SkillCall]) -> str:
    if call is None:
        return "unknown"
    return str(call)


def _clean(text: Any, limit: int = 240) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) > limit:
        return cleaned[: limit - 3] + "..."
    return cleaned


class GroundedFeedbackRenderer(object):
    """Render planner feedback from current observation summaries."""

    def render_failure(
        self,
        call: Optional[SkillCall],
        status: str,
        failure_code: str,
        progress: Optional[str],
        state_summary: StateSummary,
        budget: RetryBudget,
        recovery_action: Optional[RecoveryAction] = None,
        reason: Optional[str] = None,
    ) -> str:
        lines = [
            "Agent: {}".format(call.agent_name if call is not None else "unknown"),
            "Skill: {}".format(_call_text(call)),
            "Status: {}".format(str(status).upper()),
            "Failure: {}".format(str(failure_code or "UNKNOWN").upper()),
        ]
        if progress:
            lines.append("Progress: {}".format(progress))
        if reason:
            lines.append("Executor reason: {}".format(_clean(reason)))
        lines.append("Measured current state:")
        for fact in state_summary.measured_facts:
            lines.append("- {}".format(fact))
        if state_summary.inferred_facts:
            lines.append("Inferred explanation:")
            for fact in state_summary.inferred_facts:
                lines.append("- {}".format(_clean(fact)))
        lines.append("Retry budget: {}".format(budget.to_dict()))
        if recovery_action is not None:
            lines.append("Deterministic recovery: {}".format(recovery_action.value))
        lines.append("Choose one valid skill plan from the available capabilities.")
        return "\n".join(lines)

    def render_validation_failure(
        self,
        validation: SkillValidationResult,
        state_summary: StateSummary,
        budget: RetryBudget,
    ) -> str:
        lines = [
            "Status: PLAN_REJECTED",
            "Validation issues:",
        ]
        for issue in validation.issues:
            lines.append("- [{}] {}".format(issue.code, _clean(issue.message)))
        lines.append("Measured current state:")
        for fact in state_summary.measured_facts:
            lines.append("- {}".format(fact))
        lines.append("Retry budget: {}".format(budget.to_dict()))
        lines.append("Choose one valid skill plan from the available capabilities.")
        return "\n".join(lines)

    def render_parse_failure(
        self,
        message: str,
        state_summary: StateSummary,
        budget: RetryBudget,
    ) -> str:
        lines = [
            "Status: PLAN_REJECTED",
            "Parse issue: {}".format(_clean(message)),
            "Measured current state:",
        ]
        for fact in state_summary.measured_facts:
            lines.append("- {}".format(fact))
        lines.append("Retry budget: {}".format(budget.to_dict()))
        lines.append("Re-format to the exact typed grammar and choose one valid skill plan.")
        return "\n".join(lines)

    def render_previous_outcome(self, outcome_lines: Iterable[str]) -> str:
        lines = [str(line) for line in outcome_lines if str(line)]
        if not lines:
            return "Previous structured outcome: none"
        return "Previous structured outcome:\n" + "\n".join("- {}".format(_clean(line)) for line in lines)
