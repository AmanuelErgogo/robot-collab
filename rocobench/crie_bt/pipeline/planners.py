"""Planner backends for the pipeline.

:class:`ScriptedStagePlanner` is an oracle-free planner used for sim runs and
tests.  It proposes the next stage's action purely from
``observation.public_percepts`` -- never from ``oracle_state`` -- so it obeys the
fairness rule and stands in for a real VLM planner.  When the next stage is
assigned to a human it emits a :class:`HumanInstruction` step instead of a skill
call.

:class:`LLMPlannerAdapter` is the extension point for wrapping the existing RoCo
LLM prompters (centralized ``SingleThreadPrompter`` / dialog ``DialogPrompter``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .interfaces import (
    ExecutionFeedback,
    HumanInstruction,
    MonitorDecision,
    ObservationBundle,
    PlanStep,
    PlannerInterface,
    PlanUpdate,
    SkillCall,
    planner_safe_observation,
)


class ScriptedStagePlanner(PlannerInterface):
    """Deterministic planner that proposes the next stage from public percepts."""

    def __init__(self, agent_id: Optional[str] = None) -> None:
        # agent_id, when set, restricts this planner to its own agent's stages
        # (used by dialog/distributed controllers where each agent has a planner).
        self.agent_id = agent_id
        self._calls = 0

    def reset_episode(self, task_id: str, goal: str, capabilities: Dict[str, Any]) -> None:
        self._calls = 0

    def propose_next(
        self,
        observation: ObservationBundle,
        goal: str,
        history: List[Dict[str, Any]],
        dialogue: List[Dict[str, str]],
        feedback: Optional[ExecutionFeedback] = None,
        monitor_decision: Optional[MonitorDecision] = None,
    ) -> PlanUpdate:
        # Fairness: never read oracle_state, even though scripted logic wouldn't.
        safe = planner_safe_observation(observation)
        self._calls += 1
        percepts = safe.public_percepts or {}
        next_stage = percepts.get("next_stage")
        if not next_stage:
            return PlanUpdate(steps=[], rationale="No remaining stages; task appears complete.")

        step = self._step_from_stage(next_stage)
        rationale = "Proposing stage {} for {}.".format(next_stage.get("stage_id"), next_stage.get("agent_id"))
        dialogue_message = None
        if step.is_human_step:
            dialogue_message = step.human_instruction.text
        return PlanUpdate(steps=[step], dialogue_message=dialogue_message, rationale=rationale)

    def _step_from_stage(self, stage: Dict[str, Any]) -> PlanStep:
        stage_id = str(stage.get("stage_id"))
        agent_id = str(stage.get("agent_id"))
        skill_name = str(stage.get("skill_name"))
        args = dict(stage.get("args", {}))
        step_id = "{}__call{:03d}".format(stage_id, self._calls)
        if stage.get("assignee") == "human":
            instruction = HumanInstruction(
                text=_human_instruction_text(skill_name, args),
                target_human_id=agent_id,
                expected_stage_id=stage_id,
            )
            return PlanStep(step_id=step_id, agent_id=agent_id, human_instruction=instruction,
                            success_conditions=[stage_id])
        skill_call = SkillCall(agent_id=agent_id, skill_name=skill_name, args=args, stage_id=stage_id)
        return PlanStep(step_id=step_id, agent_id=agent_id, skill_call=skill_call, success_conditions=[stage_id])


class LLMPlannerAdapter(PlannerInterface):
    """Bridge a legacy ``rocobench.crie_bt.planner.BasePlanner`` into the pipeline.

    This is the merge point between the two code paths: the real-LLM
    ``LegacyPromptPlanner`` (or ``LegacyActionPlanner`` / ``ScriptedPlanner``) is a
    ``BasePlanner`` whose ``generate_plan`` returns a ``CollaborativePlan`` with a
    raw RoCoBench ``EXECUTE`` block in ``SkillCall.arguments["response"]``.  This
    adapter converts that into a pipeline :class:`PlanUpdate` (carrying the same
    ``response`` in ``SkillCall.args``) so pipeline controllers can drive the real
    LLM planner unchanged, while ``RoCoRRTSkillExecutorAdapter`` executes the block.

    The legacy planner needs the raw ``EnvState`` (it prompts via ``describe_obs``)
    and a per-step success signal (``notify_result``) to keep LLM history correct.
    Bind the RoCo env adapter via :meth:`bind_env`; ``notify_result`` is inferred
    from the ``feedback`` passed to the next :meth:`propose_next`.
    """

    def __init__(self, legacy_planner: Any, env_adapter: Any = None) -> None:
        self._legacy = legacy_planner
        self._env_adapter = env_adapter
        self._pending = False
        self._goal = ""

    def bind_env(self, env_adapter: Any) -> None:
        self._env_adapter = env_adapter

    def reset_episode(self, task_id: str, goal: str, capabilities: Dict[str, Any]) -> None:
        self._goal = goal
        self._pending = False
        if hasattr(self._legacy, "reset_episode"):
            self._legacy.reset_episode()

    def propose_next(
        self,
        observation: ObservationBundle,
        goal: str,
        history: List[Dict[str, Any]],
        dialogue: List[Dict[str, str]],
        feedback: Optional[ExecutionFeedback] = None,
        monitor_decision: Optional[MonitorDecision] = None,
    ) -> PlanUpdate:
        planner_safe_observation(observation)  # honour the fairness gate

        # Tell the legacy planner how the previous step went so its LLM history
        # (successful rounds / failed-plan list) stays accurate.
        if self._pending and feedback is not None and hasattr(self._legacy, "notify_result"):
            self._legacy.notify_result(bool(feedback.is_success))
            self._pending = False

        raw_obs = self._raw_obs(observation)
        legacy_feedback = self._to_legacy_feedback(feedback)
        plan = self._legacy.generate_plan(goal or self._goal, raw_obs,
                                          feedback=legacy_feedback, context=None)
        steps = self._to_plan_steps(plan)
        self._pending = bool(steps)
        rationale = "Legacy planner produced {} step(s).".format(len(steps))
        return PlanUpdate(steps=steps, rationale=rationale)

    # -- helpers ---------------------------------------------------------------
    def _raw_obs(self, observation: ObservationBundle) -> Any:
        if self._env_adapter is not None and hasattr(self._env_adapter, "raw_observation"):
            return self._env_adapter.raw_observation()
        percepts = observation.public_percepts or {}
        return percepts.get("raw_obs")

    def _to_plan_steps(self, plan: Any) -> List[PlanStep]:
        steps: List[PlanStep] = []
        for old_step in getattr(plan, "steps", []) or []:
            old_call = old_step.skill_call
            new_call = SkillCall(
                agent_id=str(getattr(old_call, "agent", "ALL")),
                skill_name=str(getattr(old_call, "skill_name", "execute_block")),
                args=dict(getattr(old_call, "arguments", {}) or {}),
                stage_id=str(getattr(old_step, "step_id", "")),
            )
            steps.append(PlanStep(step_id=str(getattr(old_step, "step_id", "")),
                                  agent_id=new_call.agent_id, skill_call=new_call))
        return steps

    def _to_legacy_feedback(self, feedback: Optional[ExecutionFeedback]) -> Any:
        # The legacy planner only augments prompts on failure; pass None otherwise.
        if feedback is None or feedback.status not in ("failure", "timeout"):
            return None
        from rocobench.crie_bt.status import BTStatus, FailureCode
        from rocobench.crie_bt.types import (
            ExecutionFeedback as LegacyFeedback,
            FailureState,
            SkillCall as LegacySkillCall,
        )
        return LegacyFeedback(
            skill_call=LegacySkillCall(agent=feedback.agent_id, skill_name="execute_block", arguments={}),
            status=BTStatus.FAILURE,
            failure=FailureState(True, FailureCode.UNKNOWN, "error", feedback.message or "execution failed"),
            message=feedback.message or "",
        )


def _human_instruction_text(skill_name: str, args: Dict[str, Any]) -> str:
    if skill_name == "verify":
        return "Please verify the {}.".format(args.get("item", "item"))
    if skill_name == "add":
        return "Please add the {}.".format(args.get("item", "ingredient"))
    obj = args.get("object") or args.get("item")
    target = args.get("target")
    if obj and target:
        return "Please {} the {} and place it at the {}.".format(skill_name, obj, target)
    if obj:
        return "Please {} the {}.".format(skill_name, obj)
    return "Please perform: {} {}.".format(skill_name, args)


__all__ = [
    "LLMPlannerAdapter",
    "ScriptedStagePlanner",
]
