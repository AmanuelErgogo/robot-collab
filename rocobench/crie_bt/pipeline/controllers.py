"""Collaboration controllers for the eight conditions.

Four concrete controllers implement the two method families crossed with the two
coordination modes:

* :class:`VLMCentralizedController` / :class:`VLMDialogController` -- the
  monolithic VLM baseline.  It has **no** explicit progress monitor; the planner
  self-monitors from image/status/history/dialogue feedback and must re-plan on
  every failure.
* :class:`CRIEBTCentralizedController` / :class:`CRIEBTDialogController` -- the
  CRIE-BT controller.  It runs an **explicit** progress monitor (coded simulator
  monitor in Step 1/2, SARM in Step 3), does cheap local retries before a full
  re-plan, and only re-plans when the monitor reports stuck/failed.

The centralized variants use one planner for the whole team; the dialog variants
give each robot-side agent its own planner and coordinate through logged
dialogue.  All four honour the fairness rule via
:func:`~rocobench.crie_bt.pipeline.interfaces.planner_safe_observation`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .communication import ScriptedHumanCommunicationInterface
from .conditions import ConditionConfig
from .episode_logger import EpisodeLogger
from .executors import SyntheticSkillExecutor
from .interfaces import (
    CollaborationController,
    CommunicationInterface,
    EnvironmentAdapter,
    ExecutionFeedback,
    HumanResponse,
    MonitorDecision,
    PlanStep,
    PlannerInterface,
    ProgressMonitorInterface,
    SkillCall,
    SkillExecutorInterface,
    StageStatus,
    planner_safe_observation,
)
from .monitors import CodedSimProgressMonitor, SARMProgressMonitor
from .planners import ScriptedStagePlanner

PlannerFactory = Callable[[Optional[str]], PlannerInterface]


def default_planner_factory(agent_id: Optional[str] = None) -> PlannerInterface:
    return ScriptedStagePlanner(agent_id=agent_id)


class PipelineController(CollaborationController):
    """Shared episode loop; subclasses fix family + coordination mode."""

    controller_family = "VLM"       # "VLM" or "CRIE-BT"
    coordination_mode = "Cent"      # "Cent" or "Dialog"

    def __init__(
        self,
        condition: ConditionConfig,
        planner_factory: Optional[PlannerFactory] = None,
        executor: Optional[SkillExecutorInterface] = None,
        monitor: Optional[ProgressMonitorInterface] = None,
        communication: Optional[CommunicationInterface] = None,
        max_local_retries: int = 2,
    ) -> None:
        self.condition = condition
        self.planner_factory = planner_factory or default_planner_factory
        self.executor = executor or SyntheticSkillExecutor()
        self.communication = communication or ScriptedHumanCommunicationInterface()
        self.max_local_retries = int(max_local_retries)
        self.monitor = monitor
        self._validate_monitor()

    @property
    def uses_explicit_monitor(self) -> bool:
        return self.controller_family == "CRIE-BT"

    def _validate_monitor(self) -> None:
        if self.uses_explicit_monitor:
            if self.monitor is None:
                self.monitor = CodedSimProgressMonitor()
            if getattr(self.monitor, "backend_name", None) == "VLM-self":
                raise ValueError("CRIE-BT controllers require an explicit progress monitor, not VLM-self.")
        else:
            # The baseline self-monitors inside the planning loop; an explicit
            # monitor must never drive baseline control.
            self.monitor = None

    # -- episode loop ----------------------------------------------------------
    def run_episode(
        self,
        condition_name: str,
        env: EnvironmentAdapter,
        task_id: str,
        goal: str,
        max_steps: int,
        seed: int = 0,
        episode_index: int = 0,
        events_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger = EpisodeLogger(self.condition, task_id, seed, episode_index, events_path=events_path)
        logger.event("episode_start", stage=self.condition.stage, goal=goal,
                     controller_family=self.controller_family, coordination_mode=self.coordination_mode)

        obs = env.reset(task_id, seed)
        self.executor.reset(env)
        planners = self._init_planners(env, task_id, goal)
        history: List[Dict[str, Any]] = []
        dialogue: List[Dict[str, str]] = []
        last_feedback: Optional[ExecutionFeedback] = None
        last_decision: Optional[MonitorDecision] = None
        task_done = bool(env.get_task_done())

        while not task_done and logger.counters["num_steps"] < max_steps:
            # Always plan against fresh state: local retries or human actions in
            # the previous round may have advanced the environment.
            obs = env.observe()
            planner, speaker_id = self._select_planner(planners, obs)
            logger.event("planner_call_start", speaker_id=speaker_id)
            logger.incr("planner_calls")
            try:
                plan = planner.propose_next(
                    planner_safe_observation(obs),
                    goal,
                    history,
                    dialogue,
                    feedback=last_feedback,
                    monitor_decision=last_decision if self.uses_explicit_monitor else None,
                )
            except Exception as exc:
                # The planner (e.g. an LLM that returned no parseable plan) failed.
                # End the episode gracefully, as the legacy controller does, rather
                # than crash — the run is logged as unsuccessful with the reason.
                logger.incr("planner_errors")
                logger.event("planner_call_end", planner_error=str(exc))
                break
            logger.event("planner_call_end", rationale=plan.rationale, num_plan_steps=len(plan.steps))

            if not plan.steps:
                task_done = bool(env.get_task_done())
                break

            if self.coordination_mode == "Dialog":
                message = plan.dialogue_message or "{}: I will handle {}.".format(
                    speaker_id, plan.steps[0].step_id)
                dialogue.append({"speaker": speaker_id, "text": message})
                self.communication.send_robot_dialogue(speaker_id, message)
                logger.incr("dialogue_turns")
                logger.event("dialogue_message", speaker_id=speaker_id, message=message)

            step = plan.steps[0]
            if self.uses_explicit_monitor:
                self.monitor.reset_stage(self._stage_id_of(step), step.skill_call, obs)
            last_feedback = self._run_step(env, step, obs, history, logger)
            obs = env.observe()

            if self.uses_explicit_monitor:
                last_decision, task_done = self._monitor_and_recover(
                    env, step, obs, last_feedback, history, logger)
            else:
                # Baseline: the VLM planner self-monitors. On failure it must
                # re-plan (an expensive planner call); on success it advances.
                if last_feedback is not None and last_feedback.status in ("failure", "timeout"):
                    logger.incr("failed_subtasks")
                    logger.incr("replans")
                    logger.event("replan_triggered", reason="baseline_self_monitor_failure")
                task_done = bool(env.get_task_done())

        success = bool(env.get_task_done())
        logger.event("episode_end", success=success, task_done=success,
                     num_steps=logger.counters["num_steps"])
        row = logger.build_row(success=success, task_done=success)
        row["events"] = logger.events
        row["dialogue"] = dialogue
        logger.close()
        return row

    # -- planner selection -----------------------------------------------------
    def _init_planners(self, env: EnvironmentAdapter, task_id: str, goal: str) -> Dict[str, PlannerInterface]:
        capabilities = {"task_id": task_id}
        if self.coordination_mode == "Cent":
            planner = self.planner_factory(None)
            planner.reset_episode(task_id, goal, capabilities)
            return {"central": planner}
        # Dialog: one planner per robot-side agent discovered from the task.
        planners: Dict[str, PlannerInterface] = {}
        for agent_id in self._robot_agent_ids(env):
            planner = self.planner_factory(agent_id)
            planner.reset_episode(task_id, goal, capabilities)
            planners[agent_id] = planner
        if not planners:  # fall back to a single planner
            planner = self.planner_factory(None)
            planner.reset_episode(task_id, goal, capabilities)
            planners["central"] = planner
        return planners

    def _robot_agent_ids(self, env: EnvironmentAdapter) -> List[str]:
        stages = getattr(env, "stages", [])
        ids: List[str] = []
        for stage in stages:
            if getattr(stage, "assignee", "robot") == "robot" and stage.agent_id not in ids:
                ids.append(stage.agent_id)
        return ids

    def _select_planner(self, planners: Dict[str, PlannerInterface], obs) -> (PlannerInterface, str):
        if "central" in planners:
            return planners["central"], "central_planner"
        next_stage = (obs.public_percepts or {}).get("next_stage") or {}
        agent_id = next_stage.get("agent_id")
        # A human stage is proposed by the robot the human is collaborating with.
        if agent_id in planners:
            return planners[agent_id], agent_id
        # Human or unknown agent: use the first robot planner as the speaker.
        first_id = next(iter(planners))
        return planners[first_id], first_id

    # -- step execution --------------------------------------------------------
    def _run_step(self, env, step: PlanStep, obs, history, logger: EpisodeLogger) -> ExecutionFeedback:
        if step.is_human_step:
            feedback = self._run_human_step(env, step, obs, logger)
        else:
            feedback = self._run_robot_step(env, step, obs, logger)
        history.append({
            "step_id": step.step_id,
            "agent_id": step.agent_id,
            "status": feedback.status,
            "stage_id": feedback.raw_info.get("stage_id") if isinstance(feedback.raw_info, dict) else None,
        })
        return feedback

    def _run_robot_step(self, env, step: PlanStep, obs, logger: EpisodeLogger) -> ExecutionFeedback:
        skill = step.skill_call
        logger.event("skill_start", agent_id=step.agent_id, skill=skill.to_dict())
        self.executor.start(skill, obs)
        feedback: Optional[ExecutionFeedback] = None
        while logger.counters["num_steps"] < 10_000:
            feedback = self.executor.step(env.observe())
            logger.incr("num_steps")
            if feedback.is_terminal:
                break
        self.executor.stop()
        logger.event("skill_feedback", agent_id=step.agent_id, status=feedback.status, message=feedback.message)
        logger.event("skill_success" if feedback.is_success else "skill_failure",
                     agent_id=step.agent_id, stage_id=feedback.raw_info.get("stage_id"))
        return feedback

    def _run_human_step(self, env, step: PlanStep, obs, logger: EpisodeLogger) -> ExecutionFeedback:
        instruction = step.human_instruction
        logger.event("human_instruction", target=instruction.target_human_id, text=instruction.text)
        self.communication.send_to_human(instruction)
        response = self.communication.read_human_response()
        logger.event("human_response", action=response.action, text=response.text)

        if response.action in ("counter_propose", "reject"):
            logger.incr("human_interventions")

        if response.action == "reject":
            logger.incr("num_steps")
            logger.incr("failed_subtasks")
            return ExecutionFeedback(
                agent_id=step.agent_id, skill_call=None, status="failure",
                done=bool(env.get_task_done()), message="Human rejected the instruction.",
                raw_info={"stage_id": instruction.expected_stage_id, "human_action": response.action},
            )

        # accept / done / counter_propose: the human performs the current stage.
        skill = self._human_skill_from_env(env, step)
        env.apply_human_action(step.agent_id, skill)
        logger.incr("num_steps")
        feedback = ExecutionFeedback(
            agent_id=step.agent_id, skill_call=skill, status="success",
            done=bool(env.get_task_done()), message="Human completed the instruction.",
            raw_info={"stage_id": instruction.expected_stage_id, "human_action": response.action},
        )
        logger.event("skill_success", agent_id=step.agent_id, stage_id=instruction.expected_stage_id, human=True)
        return feedback

    def _human_skill_from_env(self, env, step: PlanStep) -> Optional[SkillCall]:
        stage = env.current_stage() if hasattr(env, "current_stage") else None
        if stage is None:
            return None
        return SkillCall(agent_id=stage.agent_id, skill_name=stage.skill_name,
                         args=dict(stage.args), stage_id=stage.stage_id)

    def _stage_id_of(self, step: PlanStep) -> str:
        if step.skill_call is not None and step.skill_call.stage_id:
            return step.skill_call.stage_id
        if step.human_instruction is not None and step.human_instruction.expected_stage_id:
            return step.human_instruction.expected_stage_id
        return step.step_id

    # -- CRIE-BT monitor + recovery -------------------------------------------
    def _monitor_and_recover(self, env, step, obs, feedback, history, logger):
        decision = self.monitor.update(obs, feedback, history)
        logger.incr("monitor_updates")
        logger.event("monitor_update", stage_id=decision.stage_id, status=decision.status.value,
                     progress_score=decision.progress_score, should_replan=decision.should_replan,
                     privileged=decision.privileged, message=decision.message)

        if decision.is_task_done:
            return decision, True

        if decision.status in (StageStatus.FAILED, StageStatus.TIMEOUT):
            logger.incr("failed_subtasks")
            # CRIE-BT recovery: try cheap local retries before a full re-plan.
            if self._local_retry(env, step, obs, history, logger):
                return decision, bool(env.get_task_done())
            logger.incr("replans")
            logger.event("replan_triggered", reason="monitor_failed_after_retries", stage_id=decision.stage_id)
            return decision, bool(env.get_task_done())

        if decision.status == StageStatus.STUCK or decision.should_replan:
            logger.incr("replans")
            logger.event("replan_triggered", reason="monitor_stuck", stage_id=decision.stage_id)
            return decision, bool(env.get_task_done())

        # in_progress / stage_done: advance; the planner proposes the next stage.
        return decision, bool(env.get_task_done())

    def _local_retry(self, env, step, obs, history, logger) -> bool:
        """Re-execute the same robot step without a planner call. Returns True on success."""
        if step.is_human_step:
            return False
        for _ in range(self.max_local_retries):
            if logger.counters["num_steps"] >= 10_000:
                break
            logger.incr("local_retries")
            logger.event("local_retry", stage_id=step.skill_call.stage_id, agent_id=step.agent_id)
            feedback = self._run_robot_step(env, step, env.observe(), logger)
            history.append({"step_id": step.step_id, "agent_id": step.agent_id,
                            "status": feedback.status, "retry": True})
            decision = self.monitor.update(env.observe(), feedback, history)
            logger.incr("monitor_updates")
            logger.event("monitor_update", stage_id=decision.stage_id, status=decision.status.value,
                         progress_score=decision.progress_score, retry=True)
            if feedback.is_success or decision.is_stage_done or decision.is_task_done:
                return True
        return False


class VLMCentralizedController(PipelineController):
    controller_family = "VLM"
    coordination_mode = "Cent"


class VLMDialogController(PipelineController):
    controller_family = "VLM"
    coordination_mode = "Dialog"


class CRIEBTCentralizedController(PipelineController):
    controller_family = "CRIE-BT"
    coordination_mode = "Cent"


class CRIEBTDialogController(PipelineController):
    controller_family = "CRIE-BT"
    coordination_mode = "Dialog"


_CONTROLLER_BY_KEY = {
    ("VLM", "Cent"): VLMCentralizedController,
    ("VLM", "Dialog"): VLMDialogController,
    ("CRIE-BT", "Cent"): CRIEBTCentralizedController,
    ("CRIE-BT", "Dialog"): CRIEBTDialogController,
}


def build_monitor_for_condition(condition: ConditionConfig, **kwargs: Any) -> Optional[ProgressMonitorInterface]:
    if condition.baseline or condition.monitor_backend == "VLM-self":
        return None
    if condition.monitor_backend == "CodedSim":
        return CodedSimProgressMonitor(**kwargs)
    if condition.monitor_backend == "SARM":
        return SARMProgressMonitor(**kwargs)
    raise ValueError("Unknown monitor backend {!r}".format(condition.monitor_backend))


def build_collaboration_controller(
    condition: ConditionConfig,
    planner_factory: Optional[PlannerFactory] = None,
    executor: Optional[SkillExecutorInterface] = None,
    communication: Optional[CommunicationInterface] = None,
    monitor: Optional[ProgressMonitorInterface] = None,
    max_local_retries: int = 2,
) -> PipelineController:
    key = (condition.controller_family, condition.coordination_mode)
    if key not in _CONTROLLER_BY_KEY:
        raise ValueError("No controller for {}".format(key))
    controller_cls = _CONTROLLER_BY_KEY[key]
    if monitor is None:
        monitor = build_monitor_for_condition(condition)
    return controller_cls(
        condition=condition,
        planner_factory=planner_factory,
        executor=executor,
        monitor=monitor,
        communication=communication,
        max_local_retries=max_local_retries,
    )


__all__ = [
    "CRIEBTCentralizedController",
    "CRIEBTDialogController",
    "PipelineController",
    "VLMCentralizedController",
    "VLMDialogController",
    "build_collaboration_controller",
    "build_monitor_for_condition",
    "default_planner_factory",
]
