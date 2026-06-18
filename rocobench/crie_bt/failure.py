"""Failure detection rules for CRIE-BT."""

from typing import Any, Optional

from .status import BTStatus, FailureCode, ProgressStage
from .types import ExecutionFeedback, FailureState, ProgressState, SkillCall, UncertaintyState


class FailureDetector(object):
    def __init__(self, no_progress_patience: int = 3, max_steps: int = 50, low_confidence_threshold: float = 0.7) -> None:
        self.no_progress_patience = int(no_progress_patience)
        self.max_steps = int(max_steps)
        self.low_confidence_threshold = float(low_confidence_threshold)

    def detect(
        self,
        skill_call: SkillCall,
        progress: ProgressState,
        uncertainty: UncertaintyState,
        observation: Any,
        executor_feedback: Optional[ExecutionFeedback] = None,
    ) -> FailureState:
        del skill_call, observation
        if executor_feedback is not None and executor_feedback.failure.is_failure:
            return executor_feedback.failure
        if executor_feedback is not None and executor_feedback.status == BTStatus.FAILURE:
            return FailureState(True, FailureCode.UNKNOWN, "error", executor_feedback.message, executor_feedback.raw_info)
        if progress.elapsed_steps >= self.max_steps:
            return FailureState(True, FailureCode.TIMEOUT, "error", "Maximum skill steps exceeded.", progress.evidence)
        if progress.stage == ProgressStage.STUCK or progress.stagnant_steps >= self.no_progress_patience:
            return FailureState(True, FailureCode.NO_PROGRESS, "warning", "Progress has stagnated.", progress.evidence)
        if progress.stage == ProgressStage.FAILED:
            return FailureState(True, FailureCode.POSTCONDITION_FAILED, "error", "Progress monitor reported failed state.", progress.evidence)
        if uncertainty.uncertainty >= self.low_confidence_threshold:
            return FailureState(True, FailureCode.LOW_CONFIDENCE, "warning", "Uncertainty is high.", uncertainty.evidence)
        if progress.stage == ProgressStage.STABLE_SUCCESS and not progress.postcondition_satisfied:
            return FailureState(True, FailureCode.POSTCONDITION_FAILED, "error", "Success stage without postcondition evidence.", progress.evidence)
        return FailureState(False, FailureCode.NONE, "none", "", {})
