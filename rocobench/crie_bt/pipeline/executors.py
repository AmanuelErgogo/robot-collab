"""Skill executor backends for the pipeline.

* :class:`SyntheticSkillExecutor` -- deterministic executor over
  :class:`SyntheticEnvironmentAdapter`, used for sim runs and tests.
* :class:`RRTSkillExecutorAdapter` -- Step 1/Step 2 extension point wrapping the
  existing RoCo RRT skill stack.
* :class:`LearnedSkillExecutorAdapter` -- Step 3 extension point for VLA/ACT/Octo
  learned visual skills.
"""

from __future__ import annotations

from typing import Any, Optional

from .interfaces import (
    ExecutionFeedback,
    ObservationBundle,
    SkillCall,
    SkillExecutorInterface,
)


class SyntheticSkillExecutor(SkillExecutorInterface):
    """Executes a skill in one env step via ``SyntheticEnvironmentAdapter``."""

    backend_name = "RRT"  # reported as the RRT backend in Step 1/Step 2 sim runs

    def __init__(self) -> None:
        self.env = None
        self._active: Optional[SkillCall] = None

    def reset(self, env) -> None:
        self.env = env
        self._active = None

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:
        self._active = skill_call

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:
        if self._active is None:
            raise RuntimeError("SyntheticSkillExecutor.step called before start().")
        if self.env is None or not hasattr(self.env, "execute_skill"):
            raise RuntimeError("SyntheticSkillExecutor requires a SyntheticEnvironmentAdapter.")
        result = self.env.execute_skill(self._active)
        done = bool(self.env.get_task_done())
        return ExecutionFeedback(
            agent_id=self._active.agent_id,
            skill_call=self._active,
            status=result.status,
            done=done,
            message=result.message,
            raw_info={"stage_id": result.stage_id, "matched": result.matched},
        )

    def stop(self) -> None:
        self._active = None


class RRTSkillExecutorAdapter(SkillExecutorInterface):
    """Wrap the existing RoCo RRT skill executor for Step 1/Step 2 sim runs."""

    backend_name = "RRT"

    def __init__(self, rrt_executor: Any = None) -> None:
        self._rrt = rrt_executor

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:  # pragma: no cover
        raise NotImplementedError(
            "RRTSkillExecutorAdapter needs an injected RoCo RRTSkillExecutor / "
            "SkillPlan compiler. Bridge SkillCall -> RoCo action here."
        )

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:  # pragma: no cover
        raise NotImplementedError("RRTSkillExecutorAdapter.step needs an injected RRT backend.")


class LearnedSkillExecutorAdapter(SkillExecutorInterface):
    """Wrap a learned visual skill policy (VLA/ACT/Octo) for real-world Step 3."""

    backend_name = "LearnedSkill"

    def __init__(self, policy: Any = None) -> None:
        self._policy = policy

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:  # pragma: no cover
        raise NotImplementedError(
            "LearnedSkillExecutorAdapter needs an injected learned policy handle "
            "and a bridge-compatible action adapter before Step 3 runs."
        )

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:  # pragma: no cover
        raise NotImplementedError("LearnedSkillExecutorAdapter.step needs an injected learned policy.")


__all__ = [
    "LearnedSkillExecutorAdapter",
    "RRTSkillExecutorAdapter",
    "SyntheticSkillExecutor",
]
