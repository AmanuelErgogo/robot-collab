"""Uncertainty estimation for CRIE-BT."""

from typing import Any, Optional

from .status import BTStatus, FailureCode, ProgressStage
from .types import ExecutionFeedback, SkillCall, UncertaintyState


def _risk_from_uncertainty(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "medium"
    return "low"


class UncertaintyEstimator(object):
    def __init__(self, mode: str = "none") -> None:
        self.mode = str(mode or "none")
        self.failed_attempts = 0

    def reset(self) -> None:
        self.failed_attempts = 0

    def estimate(
        self,
        skill_call: SkillCall,
        observation: Any,
        executor_feedback: Optional[ExecutionFeedback] = None,
        action_chunk: Any = None,
    ) -> UncertaintyState:
        del skill_call, observation
        if self.mode == "none":
            return UncertaintyState(confidence=1.0, uncertainty=0.0, risk_level="low", source="none")
        if self.mode == "heuristic":
            return self._heuristic(executor_feedback)
        if self.mode == "ensemble_variance":
            return self._ensemble_variance(action_chunk)
        if self.mode == "policy_metadata":
            return self._policy_metadata(executor_feedback)
        return UncertaintyState(confidence=0.5, uncertainty=0.5, risk_level="medium", source=self.mode, evidence={"unknown_mode": self.mode})

    def _heuristic(self, feedback: Optional[ExecutionFeedback]) -> UncertaintyState:
        evidence = {}
        uncertainty = 0.1
        if feedback is not None:
            if feedback.status == BTStatus.FAILURE:
                self.failed_attempts += 1
                uncertainty += 0.45
            if feedback.progress.stage in (ProgressStage.STUCK, ProgressStage.FAILED):
                uncertainty += 0.35
            if feedback.progress.stagnant_steps >= 2:
                uncertainty += 0.2
            if feedback.failure.failure_code in (FailureCode.POSTCONDITION_FAILED, FailureCode.TIMEOUT):
                uncertainty += 0.25
            evidence = {
                "failed_attempts": self.failed_attempts,
                "progress_stage": feedback.progress.stage.value,
                "stagnant_steps": feedback.progress.stagnant_steps,
                "failure_code": feedback.failure.failure_code.value,
            }
        uncertainty = min(1.0, max(0.0, uncertainty))
        return UncertaintyState(
            confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            risk_level=_risk_from_uncertainty(uncertainty),
            source="heuristic",
            evidence=evidence,
        )

    def _ensemble_variance(self, action_chunk: Any) -> UncertaintyState:
        try:
            import numpy as np
            arr = np.asarray(action_chunk, dtype=float)
            variance = float(np.var(arr, axis=0).mean()) if arr.size else 0.0
        except Exception:
            variance = 0.0
        uncertainty = min(1.0, variance)
        return UncertaintyState(
            confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            risk_level=_risk_from_uncertainty(uncertainty),
            source="ensemble_variance",
            evidence={"variance": variance},
        )

    def _policy_metadata(self, feedback: Optional[ExecutionFeedback]) -> UncertaintyState:
        raw = dict(feedback.raw_info if feedback is not None else {})
        confidence = raw.get("confidence")
        entropy = raw.get("entropy")
        if confidence is not None:
            confidence = max(0.0, min(1.0, float(confidence)))
            uncertainty = 1.0 - confidence
        elif entropy is not None:
            uncertainty = max(0.0, min(1.0, float(entropy)))
            confidence = 1.0 - uncertainty
        else:
            confidence = 0.5
            uncertainty = 0.5
        return UncertaintyState(
            confidence=confidence,
            uncertainty=uncertainty,
            risk_level=_risk_from_uncertainty(uncertainty),
            source="policy_metadata",
            evidence=raw,
        )
