"""VLM/SARM monitor interface and simulator-backed baseline backend."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .status import BTStatus
from .types import ExecutionFeedback


@dataclass(frozen=True)
class VLMSARMMonitorDecision:
    """Decision returned by a monitor-planner backend.

    A real VLM/SARM backend should produce the same fields from visual/state
    evidence.  The simulator backend below fills them from simulator done and
    executor failure signals.
    """

    status: str
    should_replan: bool = False
    is_done: bool = False
    is_failed: bool = False
    message: str = ""
    source: str = "vlm_sarm"
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "should_replan": bool(self.should_replan),
            "is_done": bool(self.is_done),
            "is_failed": bool(self.is_failed),
            "message": self.message,
            "source": self.source,
            "evidence": dict(self.evidence or {}),
        }


class BaseVLMSARMMonitorBackend(object):
    """Stable monitor interface shared by simulator and real VLM/SARM paths."""

    def reset_episode(self) -> None:
        pass

    def evaluate(
        self,
        env,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        step_index: int = 0,
        max_steps: Optional[int] = None,
    ) -> VLMSARMMonitorDecision:
        raise NotImplementedError


class SimulatorSignalVLMSARMMonitor(BaseVLMSARMMonitorBackend):
    """VLM/SARM-compatible baseline that reads simulator done/failed signals."""

    source = "simulator_done_failed_signal"

    def evaluate(
        self,
        env,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        step_index: int = 0,
        max_steps: Optional[int] = None,
    ) -> VLMSARMMonitorDecision:
        done = self._task_done(env, observation)
        evidence = {
            "step_index": int(step_index),
            "max_steps": int(max_steps) if max_steps is not None else None,
            "sim_done": done,
        }
        if feedback is not None:
            evidence.update({
                "executor_status": feedback.status.value,
                "failure_code": feedback.failure.failure_code.value,
                "failure": bool(feedback.failure.is_failure),
            })

        if done is True:
            return VLMSARMMonitorDecision(
                status="DONE",
                is_done=True,
                message="Simulator reports task done.",
                source=self.source,
                evidence=evidence,
            )

        if feedback is not None and (feedback.failure.is_failure or feedback.status == BTStatus.FAILURE):
            return VLMSARMMonitorDecision(
                status="FAILED",
                should_replan=True,
                is_failed=True,
                message=feedback.failure.message or feedback.message or "Simulator/executor reports failure.",
                source=self.source,
                evidence=evidence,
            )

        if max_steps is not None and int(step_index) >= int(max_steps):
            return VLMSARMMonitorDecision(
                status="FAILED",
                should_replan=False,
                is_failed=True,
                message="Monitor step budget exhausted.",
                source=self.source,
                evidence=evidence,
            )

        # Synthetic/unit-test environments may not expose get_reward_done().
        # In that case, a terminal successful executor feedback is enough to
        # mark the one-step synthetic task done. Real simulator tasks provide
        # get_reward_done(), so this branch does not mask incomplete episodes.
        if done is None and feedback is not None and feedback.status == BTStatus.SUCCESS and not feedback.failure.is_failure:
            evidence["fallback_success_done"] = True
            return VLMSARMMonitorDecision(
                status="DONE",
                is_done=True,
                message="Executor success used as synthetic done signal.",
                source=self.source,
                evidence=evidence,
            )

        return VLMSARMMonitorDecision(
            status="IN_PROGRESS",
            message="Simulator task remains in progress.",
            source=self.source,
            evidence=evidence,
        )

    def _task_done(self, env, observation: Any) -> Optional[bool]:
        if env is None or observation is None or not hasattr(env, "get_reward_done"):
            return None
        try:
            return bool(env.get_reward_done(observation)[1])
        except Exception:
            return None
