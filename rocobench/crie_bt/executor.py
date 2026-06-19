"""Skill executor interfaces and deterministic scripted executors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from .status import BTStatus, FailureCode, ProgressStage
from .types import ExecutionContext, ExecutionFeedback, FailureState, ProgressState, SkillCall, UncertaintyState


class BaseSkillExecutor(ABC):
    @abstractmethod
    def reset(self, env, context: ExecutionContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_skill(self, skill_call: SkillCall, observation: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, observation: Any) -> ExecutionFeedback:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class ScriptedSkillExecutor(BaseSkillExecutor):
    """Executor that simulates deterministic success/failure scenarios."""

    def __init__(
        self,
        scenario: str = "none",
        success_after_steps: int = 1,
        fail_attempts: int = 1,
        low_confidence: bool = False,
    ) -> None:
        self.scenario = str(scenario or "none")
        self.success_after_steps = int(success_after_steps)
        self.fail_attempts = int(fail_attempts)
        self.low_confidence = bool(low_confidence)
        self.context = ExecutionContext()
        self.active_skill = None  # type: Optional[SkillCall]
        self.tick_count = 0
        self.attempts_by_key = {}  # type: Dict[str, int]

    def reset(self, env, context: ExecutionContext) -> None:
        del env
        self.context = context
        self.active_skill = None
        self.tick_count = 0

    def start_skill(self, skill_call: SkillCall, observation: Any) -> None:
        del observation
        self.active_skill = skill_call
        self.tick_count = 0
        self.attempts_by_key[skill_call.key] = self.attempts_by_key.get(skill_call.key, 0) + 1

    def step(self, observation: Any) -> ExecutionFeedback:
        del observation
        if self.active_skill is None:
            raise RuntimeError("ScriptedSkillExecutor.step called before start_skill.")
        self.tick_count += 1
        attempt = self.attempts_by_key.get(self.active_skill.key, 1)
        failure = self._failure_for_attempt(attempt)
        if failure is not None:
            progress = ProgressState(
                stage=ProgressStage.FAILED,
                score=0.1,
                elapsed_steps=self.tick_count,
                evidence={"attempt": attempt, "scenario": self.scenario},
            )
            uncertainty = UncertaintyState(confidence=0.2, uncertainty=0.8, risk_level="high", source="scripted")
            return ExecutionFeedback(
                skill_call=self.active_skill,
                status=BTStatus.FAILURE,
                progress=progress,
                uncertainty=uncertainty,
                failure=failure,
                message=failure.message,
                raw_info={"attempt": attempt, "scenario": self.scenario},
            )

        if self.low_confidence and self.tick_count == 1:
            progress = ProgressState(
                stage=ProgressStage.APPROACHING_OBJECT,
                score=0.35,
                elapsed_steps=self.tick_count,
                evidence={"progress_continues": True},
            )
            uncertainty = UncertaintyState(confidence=0.42, uncertainty=0.58, risk_level="medium", source="scripted")
            return ExecutionFeedback(
                skill_call=self.active_skill,
                status=BTStatus.RUNNING,
                progress=progress,
                uncertainty=uncertainty,
                failure=FailureState(False, FailureCode.NONE),
                message="Progressing with moderate uncertainty.",
                raw_info={"confidence": 0.42},
            )

        if self.tick_count >= self.success_after_steps:
            progress = ProgressState(
                stage=ProgressStage.STABLE_SUCCESS,
                score=1.0,
                elapsed_steps=self.tick_count,
                postcondition_satisfied=True,
                evidence={"attempt": attempt},
            )
            return ExecutionFeedback(
                skill_call=self.active_skill,
                status=BTStatus.SUCCESS,
                progress=progress,
                uncertainty=UncertaintyState(confidence=1.0, uncertainty=0.0, risk_level="low", source="scripted"),
                failure=FailureState(False, FailureCode.NONE),
                message="Scripted skill succeeded.",
                raw_info={"attempt": attempt},
            )

        progress = ProgressState(stage=ProgressStage.TRANSPORTING, score=0.6, elapsed_steps=self.tick_count)
        return ExecutionFeedback(
            skill_call=self.active_skill,
            status=BTStatus.RUNNING,
            progress=progress,
            uncertainty=UncertaintyState(confidence=0.8, uncertainty=0.2, risk_level="low", source="scripted"),
            failure=FailureState(False, FailureCode.NONE),
            message="Scripted skill running.",
        )

    def stop(self) -> None:
        self.active_skill = None
        self.tick_count = 0

    def _failure_for_attempt(self, attempt: int) -> Optional[FailureState]:
        if self.scenario in ("none", "", "human_interrupt"):
            return None
        if attempt > self.fail_attempts and self.scenario not in ("no_progress_always",):
            return None
        mapping = {
            "missed_grasp": (FailureCode.MISSED_GRASP, "Missed grasp during scripted execution."),
            "slippage": (FailureCode.SLIPPAGE, "Object slipped during scripted execution."),
            "no_progress": (FailureCode.NO_PROGRESS, "No progress detected during scripted execution."),
            "no_progress_always": (FailureCode.NO_PROGRESS, "No progress detected during scripted execution."),
            "target_occupied": (FailureCode.TARGET_OCCUPIED, "Target is occupied."),
            "low_confidence": (FailureCode.LOW_CONFIDENCE, "Confidence is too low."),
            "timeout": (FailureCode.TIMEOUT, "Skill timed out."),
            "safety_conflict": (FailureCode.SAFETY_CONFLICT, "Safety conflict detected."),
        }
        if self.scenario not in mapping:
            return FailureState(True, FailureCode.UNKNOWN, "error", "Unknown scripted failure scenario.", {"scenario": self.scenario})
        code, message = mapping[self.scenario]
        severity = "warning" if code == FailureCode.LOW_CONFIDENCE else "error"
        return FailureState(True, code, severity, message, {"scenario": self.scenario, "attempt": attempt})


class RRTSkillExecutor(BaseSkillExecutor):
    """Clean adapter stub for existing RoCo RRT skill execution."""

    def reset(self, env, context: ExecutionContext) -> None:
        self.env = env
        self.context = context

    def start_skill(self, skill_call: SkillCall, observation: Any) -> None:
        del skill_call, observation
        raise NotImplementedError(
            "RRTSkillExecutor needs a RoCo SkillPlan compiler/executor injection before use in CRIE-BT."
        )

    def step(self, observation: Any) -> ExecutionFeedback:
        del observation
        raise NotImplementedError("RRTSkillExecutor.step is not available without an injected backend.")

    def stop(self) -> None:
        pass


class LearnedSkillExecutor(BaseSkillExecutor):
    """Stable interface for ACT/Diffusion/LeRobot skill-model executors."""

    def reset(self, env, context: ExecutionContext) -> None:
        self.env = env
        self.context = context

    def start_skill(self, skill_call: SkillCall, observation: Any) -> None:
        del skill_call, observation
        raise NotImplementedError(
            "LearnedSkillExecutor requires an injected policy handle/checkpoint and bridge-compatible action adapter."
        )

    def step(self, observation: Any) -> ExecutionFeedback:
        del observation
        raise NotImplementedError("LearnedSkillExecutor.step is not available without an injected learned policy.")

    def stop(self) -> None:
        pass
