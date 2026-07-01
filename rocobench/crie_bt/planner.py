"""Planner interfaces for CRIE-BT."""

import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from .types import CollaborativePlan, ExecutionContext, ExecutionFeedback, PlanStep, SkillCall


def _first_match(patterns, text, default):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return default


class BasePlanner(ABC):
    @abstractmethod
    def generate_plan(
        self,
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> CollaborativePlan:
        raise NotImplementedError

    def notify_result(self, success: bool) -> None:
        """Called by the controller after executing the plan from the last generate_plan() call.

        Planners that maintain history across rounds (e.g. LegacyPromptPlanner) use
        this to record whether the action succeeded so the next LLM call sees an
        accurate round history.  The default implementation is a no-op.
        """

    def reset_episode(self) -> None:
        """Called by the controller at the start of each episode.

        Clears per-episode state (round history, failed-plan list, pending results)
        so the planner starts fresh.  The default implementation is a no-op.
        """


class ScriptedPlanner(BasePlanner):
    """Deterministic planner for tests, debugging, and offline ablations."""

    def __init__(self, default_agent: str = "Alice") -> None:
        self.default_agent = default_agent
        self.calls = 0

    def generate_plan(
        self,
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> CollaborativePlan:
        del observation
        self.calls += 1
        task_name = (context.task_name if context is not None else "") or task_goal
        task_lower = str(task_name).lower()
        goal_lower = str(task_goal).lower()
        feedback_code = ""
        if feedback is not None:
            feedback_code = feedback.failure.failure_code.value

        if "sort" in task_lower or "sort" in goal_lower:
            skill = self._sort_skill(task_goal)
        elif "pack" in task_lower or "bin" in goal_lower or "container" in goal_lower:
            skill = self._pack_skill(task_goal, feedback_code)
        else:
            skill = self._generic_pick_place_skill(task_goal)

        step = PlanStep(
            step_id="step_{:03d}".format(self.calls),
            skill_call=skill,
            role_assignment={skill.agent: "robot_executor", "Human": "verifier"},
            explanation="{} executes {} while the human can verify progress.".format(skill.agent, skill.skill_name),
        )
        return CollaborativePlan(
            plan_id="scripted_plan_{:03d}".format(self.calls),
            task_goal=task_goal,
            steps=[step],
            metadata={"planner": "scripted", "feedback_code": feedback_code},
        )

    def _pack_skill(self, task_goal: str, feedback_code: str) -> SkillCall:
        obj = _first_match([r"object=([A-Za-z0-9_]+)", r"put ([A-Za-z0-9_]+)", r"pack ([A-Za-z0-9_]+)"], task_goal, "apple")
        target = _first_match([r"container=([A-Za-z0-9_]+)", r"into ([A-Za-z0-9_]+)", r"target=([A-Za-z0-9_]+)"], task_goal, "bin_front_left")
        if feedback_code == "TARGET_OCCUPIED":
            target = "bin_front_right" if target != "bin_front_right" else "bin_back_right"
        return SkillCall(
            agent=self.default_agent,
            skill_name="PUT_OBJECT_IN_CONTAINER",
            arguments={"object": obj, "container": target},
            instruction="Put {} into {}.".format(obj, target),
        )

    def _sort_skill(self, task_goal: str) -> SkillCall:
        obj = _first_match([r"object=([A-Za-z0-9_]+)", r"sort ([A-Za-z0-9_]+)"], task_goal, "red_block")
        target = _first_match([r"target=([A-Za-z0-9_]+)", r"into ([A-Za-z0-9_]+)"], task_goal, "red_bin")
        return SkillCall(
            agent=self.default_agent,
            skill_name="SORT_OBJECT",
            arguments={"object": obj, "target": target},
            instruction="Sort {} into {}.".format(obj, target),
        )

    def _generic_pick_place_skill(self, task_goal: str) -> SkillCall:
        obj = _first_match([r"object=([A-Za-z0-9_]+)", r"pick ([A-Za-z0-9_]+)", r"move ([A-Za-z0-9_]+)"], task_goal, "object")
        target = _first_match([r"target=([A-Za-z0-9_]+)", r"to ([A-Za-z0-9_]+)", r"place .* into ([A-Za-z0-9_]+)"], task_goal, "target")
        return SkillCall(
            agent=self.default_agent,
            skill_name="PICK_PLACE",
            arguments={"object": obj, "target": target},
            instruction="Pick {} and place it at {}.".format(obj, target),
        )


class LLMPlannerAdapter(BasePlanner):
    """Adapter placeholder for existing RoCo LLM planning code."""

    def __init__(self, planner: Optional[Any] = None) -> None:
        self.planner = planner

    def generate_plan(
        self,
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> CollaborativePlan:
        del task_goal, observation, feedback, context
        if self.planner is None:
            raise NotImplementedError(
                "LLMPlannerAdapter requires an injected RoCo planner that returns structured CRIE-BT plans."
            )
        if not hasattr(self.planner, "generate_plan"):
            raise NotImplementedError("Injected planner must expose generate_plan(...).")
        plan = self.planner.generate_plan()
        if isinstance(plan, CollaborativePlan):
            return plan
        if isinstance(plan, Mapping):
            return CollaborativePlan.from_dict(plan)
        raise NotImplementedError("Injected planner returned unsupported plan type: {}".format(type(plan)))
