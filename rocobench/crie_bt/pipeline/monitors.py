"""Progress-monitor backends for the CRIE-BT / SARM pipeline.

Three backends, matching ``docs/crie_next_stage_plan/04_interface_api_spec.md``:

* :class:`CodedSimProgressMonitor` -- **privileged** coded/oracle simulator
  monitor used by CRIE-BT in Step 1 and Step 2.  It may read privileged
  simulator state (object poses, task predicates, RRT status, ``done``).
* :class:`SARMProgressMonitor` -- Stage-Aware Reward Modeling monitor used by
  CRIE-BT in the real-world Step 3.  It is **not** privileged: it scores
  current-stage progress from camera observations only.
* :class:`NoSeparateMonitor` -- logging placeholder for the baseline, which has
  no separate monitor and self-monitors inside the VLM planning loop.

Naming note: the baseline is never labelled SARM.  SARM belongs to the CRIE-BT
real-world monitor backend only (see
``docs/crie_next_stage_plan/10_coding_agent_prompt.md``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .interfaces import (
    ExecutionFeedback,
    MonitorDecision,
    ObservationBundle,
    ProgressMonitorInterface,
    SkillCall,
    StageStatus,
)


def _oracle(observation: ObservationBundle) -> Dict[str, Any]:
    return dict(observation.oracle_state or {})


class CodedSimProgressMonitor(ProgressMonitorInterface):
    """Coded/oracle simulator progress monitor (privileged).

    Reads whatever simulator/task signal reveals whether the current subtask is
    processing, complete, stuck, or failed.  It prefers structured oracle signals
    (``stage_progress``, ``stage_done``, ``task_done`` in ``oracle_state``) and
    falls back to executor feedback when those are absent, so it also works with
    simple synthetic environments and unit-test stubs.
    """

    backend_name = "CodedSim"
    privileged = True

    def __init__(self, no_progress_patience: int = 3) -> None:
        self.no_progress_patience = int(no_progress_patience)
        self.stage_id = ""
        self._last_score = 0.0
        self._stagnant_steps = 0

    def reset_stage(
        self,
        stage_id: str,
        skill_call: Optional[SkillCall],
        observation: ObservationBundle,
    ) -> None:
        self.stage_id = stage_id or (skill_call.stage_id if skill_call else "") or ""
        self._last_score = 0.0
        self._stagnant_steps = 0

    def update(
        self,
        observation: ObservationBundle,
        feedback: Optional[ExecutionFeedback],
        history: List[Dict[str, Any]],
    ) -> MonitorDecision:
        oracle = _oracle(observation)
        # Prefer the stage the executor actually acted on (feedback), then the
        # per-stage id set at reset, then whatever the oracle currently reports.
        feedback_stage = None
        if feedback is not None and isinstance(feedback.raw_info, dict):
            feedback_stage = feedback.raw_info.get("stage_id")
        stage_id = str(feedback_stage or self.stage_id or oracle.get("stage_id") or "unknown_stage")

        task_done = _first_bool(oracle.get("task_done"), feedback.done if feedback else None)
        stage_done = _first_bool(oracle.get("stage_done"))
        stuck = bool(oracle.get("stuck", False))

        score = oracle.get("stage_progress")
        if score is None and feedback is not None:
            score = _score_from_feedback(feedback)
        score = 0.0 if score is None else float(score)

        # No-progress detection when the oracle does not label ``stuck`` itself.
        if score <= self._last_score + 1e-4:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = 0
        self._last_score = max(self._last_score, score)

        evidence: Dict[str, Any] = {
            "oracle_keys": sorted(oracle.keys()),
            "stagnant_steps": self._stagnant_steps,
        }
        if feedback is not None:
            evidence["executor_status"] = feedback.status
            evidence["failure"] = feedback.status in ("failure", "timeout")

        failed = feedback is not None and feedback.status in ("failure", "timeout")

        # Privileged postcondition signal (from the RoCo executor's task predicate /
        # env.get_reward_done for this subtask). Mirrors the legacy FailureDetector
        # rule that a completed motion whose postcondition is unmet is NOT stage done.
        # Absent (e.g. synthetic env) -> None -> a motion "success" counts as done.
        postcondition = None
        if feedback is not None and isinstance(feedback.raw_info, dict):
            postcondition = feedback.raw_info.get("postcondition_satisfied")
        motion_success = feedback is not None and feedback.status == "success"
        stage_complete = bool(stage_done) or (motion_success and postcondition in (None, True))
        evidence["postcondition_satisfied"] = postcondition

        if task_done:
            return self._decision(stage_id, StageStatus.TASK_DONE, 1.0, is_stage_done=True, is_task_done=True,
                                   message="Simulator reports task done.", evidence=evidence)
        if failed:
            return self._decision(stage_id, StageStatus.FAILED, score, should_replan=True,
                                   message=feedback.message or "Executor reports failure.", evidence=evidence)
        # A completed motion whose postcondition holds means the acted stage is done.
        if stage_complete:
            return self._decision(stage_id, StageStatus.STAGE_DONE, 1.0,
                                   is_stage_done=True, message="Current stage complete.", evidence=evidence)
        if stuck or self._stagnant_steps >= self.no_progress_patience:
            return self._decision(stage_id, StageStatus.STUCK, score, should_replan=True,
                                   message="No progress detected; requesting replan.", evidence=evidence)
        return self._decision(stage_id, StageStatus.IN_PROGRESS, score,
                              message="Stage in progress.", evidence=evidence)

    def _decision(self, stage_id: str, status: StageStatus, score: float, **kwargs: Any) -> MonitorDecision:
        return MonitorDecision(
            stage_id=stage_id,
            status=status,
            progress_score=score,
            monitor_backend=self.backend_name,
            privileged=self.privileged,
            **kwargs,
        )


class SARMProgressMonitor(ProgressMonitorInterface):
    """Stage-Aware Reward Modeling progress monitor for real-world Step 3.

    SARM outputs a progress score in ``[0, 1]`` for the current stage from camera
    observations only.  It is **not** privileged.  A learned scorer is injected
    via ``scorer(stage_id, observation) -> float``; failure classification is
    optional and not required for the first real-world implementation.
    """

    backend_name = "SARM"
    privileged = False

    def __init__(
        self,
        scorer: Optional[Callable[[str, ObservationBundle], float]] = None,
        stage_done_threshold: float = 0.9,
    ) -> None:
        self.scorer = scorer
        self.stage_done_threshold = float(stage_done_threshold)
        self.stage_id = ""

    def reset_stage(
        self,
        stage_id: str,
        skill_call: Optional[SkillCall],
        observation: ObservationBundle,
    ) -> None:
        self.stage_id = stage_id or (skill_call.stage_id if skill_call else "") or ""

    def update(
        self,
        observation: ObservationBundle,
        feedback: Optional[ExecutionFeedback],
        history: List[Dict[str, Any]],
    ) -> MonitorDecision:
        stage_id = self.stage_id or "unknown_stage"
        if self.scorer is None:
            raise NotImplementedError(
                "SARMProgressMonitor requires an injected learned scorer "
                "scorer(stage_id, observation) -> float in [0, 1]. "
                "Train one from stage/progress labels before Step 3 runs."
            )
        score = self.scorer(stage_id, observation)
        is_stage_done = score >= self.stage_done_threshold
        status = StageStatus.STAGE_DONE if is_stage_done else StageStatus.IN_PROGRESS
        failed = feedback is not None and feedback.status in ("failure", "timeout")
        if failed:
            status = StageStatus.FAILED
        return MonitorDecision(
            stage_id=stage_id,
            status=status,
            progress_score=score,
            should_replan=failed,
            is_stage_done=is_stage_done,
            message="SARM stage progress {:.2f}.".format(max(0.0, min(1.0, float(score)))),
            evidence={"threshold": self.stage_done_threshold},
            monitor_backend=self.backend_name,
            privileged=self.privileged,
        )


class NoSeparateMonitor(ProgressMonitorInterface):
    """Logging placeholder for the baseline: the VLM planner self-monitors.

    This should not drive control.  It exists only so baseline runs can record a
    ``monitor_backend`` of ``VLM-self`` without instantiating a real monitor.
    """

    backend_name = "VLM-self"
    privileged = False

    def update(
        self,
        observation: ObservationBundle,
        feedback: Optional[ExecutionFeedback],
        history: List[Dict[str, Any]],
    ) -> MonitorDecision:
        raise NotImplementedError(
            "NoSeparateMonitor is a logging placeholder; the baseline VLM planner "
            "self-monitors and must not call an explicit monitor."
        )


def build_monitor(backend: str, **kwargs: Any) -> ProgressMonitorInterface:
    if backend == "CodedSim":
        return CodedSimProgressMonitor(**kwargs)
    if backend == "SARM":
        return SARMProgressMonitor(**kwargs)
    if backend == "VLM-self":
        return NoSeparateMonitor()
    raise ValueError("Unknown monitor backend: {!r}".format(backend))


def _first_bool(*values: Any) -> Optional[bool]:
    for value in values:
        if value is not None:
            return bool(value)
    return None


def _score_from_feedback(feedback: ExecutionFeedback) -> float:
    if feedback.status == "success":
        return 1.0
    if feedback.status in ("failure", "timeout"):
        return 0.0
    # Running: use any progress hint the executor exposes, else a small default.
    hint = feedback.raw_info.get("progress_score") if isinstance(feedback.raw_info, dict) else None
    if hint is not None:
        return float(hint)
    return 0.5


__all__ = [
    "CodedSimProgressMonitor",
    "NoSeparateMonitor",
    "SARMProgressMonitor",
    "build_monitor",
]
