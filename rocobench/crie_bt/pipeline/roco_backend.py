"""Real RoCoBench (MuJoCo + RRT) backend for the pipeline.

This module bridges the clean pipeline interfaces to the existing, working
RoCoBench simulator and RRT skill stack so Step 1 / Step 2 can run on the real
environment with the same controllers used for the synthetic backend.

It is intentionally kept out of ``pipeline/__init__.py``: importing it *does*
pull in MuJoCo/dm_control and the legacy RoCo stack.  Set an offscreen GL
backend before use, e.g. ``MUJOCO_GL=egl``.

Components
----------
* :class:`RoCoBenchEnvironmentAdapter` -- wraps a ``rocobench.envs`` task; maps
  ``EnvState`` into an :class:`ObservationBundle` (scene description as a
  non-privileged percept) and exposes ``get_reward_done`` as privileged
  ``oracle_state``.
* :class:`RoCoRRTSkillExecutorAdapter` -- delegates to the legacy
  ``LegacyTaskRRTExecutorAdapter``; a pipeline :class:`SkillCall` carries a raw
  RoCoBench ``EXECUTE`` block in ``args["response"]``.
* :class:`RoCoScriptedPlanner` -- emits provided ``EXECUTE`` blocks as
  :class:`PlanUpdate`s (stand-in for the LLM planner, which drops in via
  ``LLMPlannerAdapter`` once an LLM client + credentials are configured).
* :func:`build_roco_step1` -- wires all three to a task and returns
  ``(env_adapter, executor, planner_factory)`` ready for a controller.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from .interfaces import (
    EnvironmentAdapter,
    ExecutionFeedback,
    MonitorDecision,
    ObservationBundle,
    PlanStep,
    PlannerInterface,
    PlanUpdate,
    SkillCall,
    SkillExecutorInterface,
    planner_safe_observation,
)


class RoCoBenchEnvironmentAdapter(EnvironmentAdapter):
    """Wrap a real ``rocobench.envs`` task as a pipeline environment."""

    environment = "sim"

    def __init__(self, roco_env: Any, task_id: str) -> None:
        self._env = roco_env
        self.task_id = task_id
        self._raw_obs = None

    def reset(self, task_id: str, seed: int) -> ObservationBundle:
        # The task is built + seeded by build_roco_step1; refresh the observation.
        self._raw_obs = self._env.get_obs()
        return self.observe()

    def raw_observation(self) -> Any:
        """The underlying ``EnvState`` the RRT executor/parser needs."""
        if self._raw_obs is None:
            self._raw_obs = self._env.get_obs()
        return self._raw_obs

    def refresh(self) -> Any:
        self._raw_obs = self._env.get_obs()
        return self._raw_obs

    def observe(self) -> ObservationBundle:
        raw = self.raw_observation()
        scene = ""
        if hasattr(self._env, "describe_obs"):
            try:
                scene = self._env.describe_obs(raw)
            except Exception:
                scene = ""
        done = self._task_done(raw)
        return ObservationBundle(
            public_percepts={"scene": scene, "task_id": self.task_id},
            oracle_state={"task_done": done, "scene": scene},
            timestamp=0.0,
        )

    def step_robot(self, agent_id: str, low_level_action: Any) -> ObservationBundle:
        # The RRT executor advances the MuJoCo sim directly; just re-observe.
        return self.observe()

    def apply_human_action(self, human_id: str, action: Any) -> ObservationBundle:
        return self.observe()

    def get_task_done(self) -> Optional[bool]:
        return self._task_done(self.refresh())

    def get_oracle_state(self) -> Dict[str, Any]:
        return {"task_done": self.get_task_done()}

    def _task_done(self, raw: Any) -> bool:
        if not hasattr(self._env, "get_reward_done") or raw is None:
            return False
        try:
            return bool(self._env.get_reward_done(raw)[1])
        except Exception:
            return False


def _map_status(old_feedback: Any) -> str:
    """Map a legacy ``ExecutionFeedback`` (old BTStatus enum) to the new status."""
    failure = getattr(old_feedback, "failure", None)
    if failure is not None and getattr(failure, "is_failure", False):
        return "failure"
    status = getattr(old_feedback, "status", None)
    name = getattr(status, "value", status)
    if name in ("SUCCESS", "success"):
        return "success"
    if name in ("FAILURE", "failure"):
        return "failure"
    return "running"


class RoCoRRTSkillExecutorAdapter(SkillExecutorInterface):
    """Delegate pipeline skill execution to the legacy RRT executor adapter."""

    backend_name = "RRT"

    def __init__(self, legacy_adapter: Any) -> None:
        self._legacy = legacy_adapter
        self._env_adapter: Optional[RoCoBenchEnvironmentAdapter] = None
        self._active: Optional[SkillCall] = None

    def reset(self, env: RoCoBenchEnvironmentAdapter) -> None:
        from rocobench.crie_bt.types import ExecutionContext as LegacyContext

        self._env_adapter = env
        self._legacy.reset(env._env, LegacyContext(task_name=env.task_id))
        self._active = None

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:
        from rocobench.crie_bt.types import SkillCall as LegacySkillCall

        self._active = skill_call
        raw = self._env_adapter.raw_observation()
        legacy_call = LegacySkillCall(
            agent=skill_call.agent_id,
            skill_name=skill_call.skill_name,
            arguments={"response": skill_call.args.get("response", "")},
            instruction=skill_call.args.get("response", ""),
        )
        self._legacy.start_skill(legacy_call, raw)

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:
        raw = self._env_adapter.raw_observation()
        old_feedback = self._legacy.step(raw)
        self._env_adapter.refresh()
        # Preserve the legacy feedback's privileged signals so the coded monitor can
        # apply the plan's postcondition/timeout checks (see 01_locked_decisions.md §2):
        #  - failure_code (POSTCONDITION_FAILED, TIMEOUT, ...) — a real RRT timeout is
        #    surfaced as status "timeout";
        #  - postcondition_satisfied — a completed motion whose subtask goal is unmet
        #    must NOT count as stage-done.
        status = _map_status(old_feedback)
        failure = getattr(old_feedback, "failure", None)
        failure_code = getattr(getattr(failure, "failure_code", None), "value", None)
        if status == "failure" and failure_code == "TIMEOUT":
            status = "timeout"
        progress = getattr(old_feedback, "progress", None)
        postcondition = getattr(progress, "postcondition_satisfied", None)
        return ExecutionFeedback(
            agent_id=self._active.agent_id if self._active else "robot",
            skill_call=self._active,
            status=status,
            done=self._env_adapter.get_task_done(),
            message=getattr(old_feedback, "message", ""),
            raw_info={
                "stage_id": self._active.stage_id if self._active else None,
                "legacy_status": getattr(getattr(old_feedback, "status", None), "value", None),
                "failure_code": failure_code,
                "postcondition_satisfied": postcondition,
            },
        )

    def stop(self) -> None:
        self._legacy.stop()


class RoCoScriptedPlanner(PlannerInterface):
    """Emit provided RoCoBench EXECUTE blocks as pipeline plan steps.

    Stand-in for the LLM planner so the real backend can be validated without an
    LLM client.  Swap for ``LLMPlannerAdapter`` wrapping a RoCo prompter to get
    real planning.
    """

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def reset_episode(self, task_id: str, goal: str, capabilities: Dict[str, Any]) -> None:
        self._index = 0

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
        if self._index >= len(self._responses):
            return PlanUpdate(steps=[], rationale="No further scripted RoCo actions.")
        block = self._responses[self._index]
        self._index += 1
        agent = _first_agent(block)
        stage_id = "roco_stage_{:03d}".format(self._index)
        skill_call = SkillCall(agent_id=agent, skill_name="execute_block",
                               args={"response": block}, stage_id=stage_id)
        step = PlanStep(step_id="s{:03d}".format(self._index), agent_id=agent, skill_call=skill_call)
        return PlanUpdate(steps=[step], rationale="Scripted RoCo EXECUTE block {}.".format(self._index))


def _first_agent(execute_block: str) -> str:
    match = re.search(r"NAME\s+(\w+)", execute_block)
    return match.group(1) if match else "robot"


def build_roco_step1(
    task_id: str,
    seed: int = 0,
    responses: Optional[List[str]] = None,
    max_sim_steps: int = 3000,
    artifact_dir: Optional[str] = None,
):
    """Build a real RoCoBench Step 1 backend.

    Returns ``(env_adapter, executor, planner_factory)``.  If ``responses`` is
    None, a single WAIT action for every agent is used (valid, no motion
    planning) -- enough to validate the integration path end to end.

    Pass ``artifact_dir`` to record ``<artifact_dir>/execute.mp4``.
    """
    from prompting.parser import LLMResponseParser
    from rocobench.crie_bt.legacy_tasks import (
        FakeLegacyUncertaintyReporter,
        LegacyTaskRRTExecutorAdapter,
        agent_names_for_env,
        make_legacy_task_env,
        make_wait_response,
    )
    from rocobench.skills import RRTSkillExecutor

    env = make_legacy_task_env(task_id, seed=seed)
    agent_names = agent_names_for_env(env)
    if responses is None:
        responses = [make_wait_response(agent_names)]

    parser = LLMResponseParser(
        env,
        "action_only",
        env.robot_name_map,
        ["NAME", "ACTION"],
        use_prepick=getattr(env, "use_prepick", False),
        use_preplace=getattr(env, "use_preplace", False),
    )
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    rrt_executor = RRTSkillExecutor(env=env, robots=env.get_sim_robots(), max_sim_steps=int(max_sim_steps))
    legacy_adapter = LegacyTaskRRTExecutorAdapter(
        env=env,
        task_id=task_id,
        parser=parser,
        executor=rrt_executor,
        artifact_dir=artifact_dir,
        uncertainty_reporter=FakeLegacyUncertaintyReporter("nominal"),
    )

    env_adapter = RoCoBenchEnvironmentAdapter(env, task_id)
    executor = RoCoRRTSkillExecutorAdapter(legacy_adapter)

    def planner_factory(agent_id: Optional[str] = None) -> PlannerInterface:
        return RoCoScriptedPlanner(list(responses))

    return env_adapter, executor, planner_factory


def build_roco_condition(
    condition: Any,
    task: str,
    seed: int = 0,
    llm_source: str = "gpt-4",
    api_key_path: str = "",
    max_sim_steps: int = 5000,
    prompt_save_dir: Optional[str] = None,
    max_local_retries: int = 2,
    artifact_dir: Optional[str] = None,
):
    """Build the **real** backend for a pipeline condition (registry -> real run).

    This is the convergence entry point: it resolves a :class:`ConditionConfig`
    into a real RoCoBench env + real RRT executor + real-LLM planner (the same
    Gemini/OpenAI prompter wiring the legacy runner uses, via ``roco_runtime``) +
    the matching pipeline controller/monitor.  Coordination selects the prompter:
    ``Cent`` -> ``SingleThreadPrompter`` (chat), ``Dialog`` -> ``DialogPrompter``.

    Returns ``(controller, env_adapter)``; call
    ``controller.run_episode(condition.condition_name, env_adapter, task, goal, max_steps)``.

    Pass ``artifact_dir`` to record an episode video: the RRT executor writes
    ``<artifact_dir>/execute.mp4`` (the legacy task env renders its ``teaser``
    camera; frames accumulate across the episode).
    """
    from rocobench.crie_bt.legacy_tasks import make_legacy_task_env
    from rocobench.crie_bt.roco_runtime import build_legacy_prompt_planner, build_legacy_rrt_executor

    from .controllers import build_collaboration_controller
    from .planners import LLMPlannerAdapter

    env = make_legacy_task_env(task, seed=seed)
    env_adapter = RoCoBenchEnvironmentAdapter(env, task)

    # The RoCo prompter writes prompt artifacts to save_dir; an empty/None dir
    # makes it write to "/". Default to a real results dir like the legacy runner.
    if not prompt_save_dir:
        prompt_save_dir = os.path.join(
            _REPO_ROOT, "results", "roco_pipeline_prompts", "{}_{}_seed{}".format(
                getattr(condition, "code_name", "cond"), task, seed))
    os.makedirs(prompt_save_dir, exist_ok=True)

    planner_mode = "dialog" if getattr(condition, "coordination_mode", "Cent") == "Dialog" else "chat"
    legacy_planner = build_legacy_prompt_planner(
        env, task, planner_mode, llm_source=llm_source,
        api_key_path=api_key_path, save_dir=prompt_save_dir,
    )
    llm_adapter = LLMPlannerAdapter(legacy_planner, env_adapter=env_adapter)

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    legacy_executor = build_legacy_rrt_executor(
        env, task, max_sim_steps=max_sim_steps, artifact_dir=artifact_dir)
    executor = RoCoRRTSkillExecutorAdapter(legacy_executor)

    # One shared LLM planner: multi-agent dialogue lives inside the DialogPrompter,
    # so both Cent and Dialog controllers select the same adapter.
    def planner_factory(agent_id: Optional[str] = None) -> PlannerInterface:
        return llm_adapter

    controller = build_collaboration_controller(
        condition, planner_factory=planner_factory, executor=executor,
        max_local_retries=max_local_retries,
    )
    return controller, env_adapter


__all__ = [
    "RoCoBenchEnvironmentAdapter",
    "RoCoRRTSkillExecutorAdapter",
    "RoCoScriptedPlanner",
    "build_roco_condition",
    "build_roco_step1",
]
