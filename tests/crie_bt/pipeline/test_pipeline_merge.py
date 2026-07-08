"""Merge tests: legacy BasePlanner (real-LLM family) driven through the pipeline.

These validate the merge mechanism with a mock legacy planner + a fake
RoCo-style env/executor, so no MuJoCo or LLM is needed. The real path
(``build_roco_condition``) uses the same ``LLMPlannerAdapter`` bridge with a real
Gemini/OpenAI ``LegacyPromptPlanner``.
"""

from typing import Any, List, Optional

from rocobench.crie_bt.pipeline import build_condition, build_collaboration_controller
from rocobench.crie_bt.pipeline.interfaces import (
    EnvironmentAdapter,
    ExecutionFeedback,
    ObservationBundle,
    SkillCall,
    SkillExecutorInterface,
)
from rocobench.crie_bt.pipeline.planners import LLMPlannerAdapter

# Legacy (old) CRIE-BT types the real LegacyPromptPlanner also emits.
from rocobench.crie_bt.types import CollaborativePlan
from rocobench.crie_bt.types import PlanStep as LegacyPlanStep
from rocobench.crie_bt.types import SkillCall as LegacySkillCall


class FakeLegacyPlanner:
    """Mimics rocobench.crie_bt.planner.BasePlanner (as LegacyPromptPlanner does).

    Emits one EXECUTE block per call from a fixed script; empty plan when done.
    Records notify_result / reset_episode so the bridge's history calls are testable.
    """

    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.index = 0
        self.notify_calls: List[bool] = []
        self.reset_calls = 0

    def generate_plan(self, task_goal, observation, feedback=None, context=None) -> CollaborativePlan:
        if self.index >= len(self.responses):
            return CollaborativePlan(steps=[], plan_id="empty", task_goal=task_goal)
        block = self.responses[self.index]
        self.index += 1
        step = LegacyPlanStep(
            step_id="legacy_step_{:03d}".format(self.index),
            skill_call=LegacySkillCall(agent="ALL", skill_name="LEGACY_ACTION_PLAN",
                                       arguments={"task": "t", "response": block}),
        )
        return CollaborativePlan(steps=[step], plan_id="p{}".format(self.index), task_goal=task_goal)

    def notify_result(self, success: bool) -> None:
        self.notify_calls.append(bool(success))

    def reset_episode(self) -> None:
        self.reset_calls += 1


class FakeRoCoEnv(EnvironmentAdapter):
    """A RoCo-style env whose 'skills' are EXECUTE blocks; N blocks complete it."""

    environment = "sim"

    def __init__(self, num_stages: int = 3) -> None:
        self.num_stages = num_stages
        self.done_count = 0

    def reset(self, task_id, seed):
        self.done_count = 0
        return self.observe()

    def raw_observation(self):
        return {"env_state": "fake", "remaining": self.num_stages - self.done_count}

    def observe(self) -> ObservationBundle:
        done = self.get_task_done()
        return ObservationBundle(public_percepts={"scene": "fake"}, oracle_state={"task_done": done})

    def apply_response(self, response: str) -> None:
        self.done_count += 1

    def get_task_done(self):
        return self.done_count >= self.num_stages


class FakeResponseExecutor(SkillExecutorInterface):
    """Executes an EXECUTE block by advancing the fake env one stage."""

    backend_name = "RRT"

    def reset(self, env) -> None:
        self._env = env
        self._active = None

    def start(self, skill_call: SkillCall, observation) -> None:
        self._active = skill_call

    def step(self, observation) -> ExecutionFeedback:
        self._env.apply_response(self._active.args.get("response", ""))
        return ExecutionFeedback(
            agent_id=self._active.agent_id, skill_call=self._active, status="success",
            done=bool(self._env.get_task_done()), message="fake executed",
            raw_info={"stage_id": self._active.stage_id},
        )

    def stop(self) -> None:
        self._active = None


_BLOCKS = ["EXECUTE\nNAME Alice ACTION WAIT", "EXECUTE\nNAME Bob ACTION WAIT",
           "EXECUTE\nNAME Alice ACTION WAIT"]


def test_adapter_converts_legacy_plan_and_notifies_result():
    legacy = FakeLegacyPlanner(_BLOCKS)
    adapter = LLMPlannerAdapter(legacy)
    adapter.reset_episode("t", "goal", {})
    assert legacy.reset_calls == 1

    obs = ObservationBundle(public_percepts={"raw_obs": {"x": 1}})
    plan = adapter.propose_next(obs, "goal", [], [])
    assert len(plan.steps) == 1
    assert plan.steps[0].skill_call.args["response"] == _BLOCKS[0]
    assert legacy.notify_calls == []  # nothing to notify yet

    success = ExecutionFeedback(agent_id="ALL", skill_call=None, status="success")
    adapter.propose_next(obs, "goal", [], [], feedback=success)
    assert legacy.notify_calls == [True]  # previous step reported successful


def test_baseline_condition_runs_real_llm_planner_through_pipeline():
    legacy = FakeLegacyPlanner(_BLOCKS)
    env = FakeRoCoEnv(num_stages=3)
    adapter = LLMPlannerAdapter(legacy, env_adapter=env)
    condition = build_condition("VLM-RR-Cent", "step1")
    controller = build_collaboration_controller(
        condition, planner_factory=lambda agent_id=None: adapter, executor=FakeResponseExecutor())
    row = controller.run_episode("VLM-RR-Cent", env, "sandwich", "goal", max_steps=20)

    assert row["success"] is True
    assert row["planner_calls"] >= 3
    assert row["monitor_updates"] == 0          # baseline: no explicit monitor
    assert row["monitor_backend"] == "VLM-self"
    assert row["condition_name"] == "VLM-RR-Cent"


def test_criebt_condition_runs_real_llm_planner_with_coded_monitor():
    legacy = FakeLegacyPlanner(_BLOCKS)
    env = FakeRoCoEnv(num_stages=3)
    adapter = LLMPlannerAdapter(legacy, env_adapter=env)
    condition = build_condition("CRIE-BT-RR-Cent", "step1")
    controller = build_collaboration_controller(
        condition, planner_factory=lambda agent_id=None: adapter, executor=FakeResponseExecutor())
    row = controller.run_episode("CRIE-BT-RR-Cent", env, "sandwich", "goal", max_steps=20)

    assert row["success"] is True
    assert row["monitor_updates"] >= 3          # coded monitor read real oracle_state
    assert row["monitor_backend"] == "CodedSim"
    assert row["monitor_privileged"] is True


def test_build_roco_condition_importable():
    # The real builder must import without MuJoCo (heavy imports are lazy).
    import sys

    from rocobench.crie_bt.pipeline.roco_backend import build_roco_condition

    assert callable(build_roco_condition)
    assert "mujoco" not in sys.modules
