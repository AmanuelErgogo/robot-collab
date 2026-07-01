"""Tests for the learned subtask skill pipeline.

Covers:
  - Skill registry construction for all tasks
  - Success checkers (PICK, PLACE, STACK_ON, OPEN_CABINET)
  - MockACTHandle: parse_instruction, metadata
  - SubtaskLearnedExecutor: full execution loop with mock policy
  - UncertaintyEstimator: all three modes
  - SubtaskEnvAdapter: action encoding, state_vector
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

import numpy as np
import pytest

from rocobench.skills.learned.subtask_skills import (
    build_pick_place_registry,
    build_cabinet_registry,
    build_sandwich_registry,
    build_sweep_registry,
    SKILL_PICK, SKILL_PLACE, SKILL_PICK_AND_PLACE,
    SKILL_OPEN_CABINET, SKILL_STACK_ON, SKILL_WAIT,
)
from rocobench.skills.learned.subtask_success import (
    PickSuccessChecker,
    PlaceSuccessChecker,
    StackOnSuccessChecker,
    OpenCabinetSuccessChecker,
    build_success_checker,
)
from rocobench.skills.learned.mock_handle import MockACTHandle
from rocobench.skills.learned.subtask_executor import (
    SubtaskLearnedExecutor,
    SubtaskEnvAdapter,
    _SimpleQueue,
    _summarise_uncertainty,
)
from rocobench.skills.learned.registry import LearnedPolicyRegistry
from rocobench.skills.learned.models import LearnedPolicySpec
from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache, NativeActionChunk
from rocobench.skills.models import SkillCall, SkillPlan, SkillExecutionStatus
from rocobench.crie_bt.uncertainty import UncertaintyEstimator
from rocobench.crie_bt.types import SkillCall as BTSkillCall


# ===========================================================================
# Fake environment objects (no MuJoCo required)
# ===========================================================================

@dataclass
class FakeSite:
    name: str
    xpos: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.1, 0.15]))
    xmat: np.ndarray = field(default_factory=lambda: np.eye(3).flatten())
    xquat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=float))


@dataclass
class FakeObjectState:
    name: str
    xpos: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.1, 0.05]))
    xquat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=float))
    contacts: Set[str] = field(default_factory=set)
    sites: Dict[str, FakeSite] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sites:
            self.sites = {f"{self.name}_top": FakeSite(f"{self.name}_top", self.xpos + [0, 0, 0.05])}

    @property
    def top_height(self):
        return max(s.xpos[2] for s in self.sites.values()) if self.sites else self.xpos[2]

    @property
    def bottom_height(self):
        return min(s.xpos[2] for s in self.sites.values()) if self.sites else self.xpos[2]


@dataclass
class FakeAgentState:
    name: str
    ee_xpos: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.1, 0.2]))
    ee_xquat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=float))
    ee_xmat: np.ndarray = field(default_factory=lambda: np.eye(3).flatten())
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(7))
    contacts: Set[str] = field(default_factory=set)

    @property
    def ee_pose(self):
        return np.concatenate([self.ee_xpos, self.ee_xquat])


class FakeObs:
    def __init__(self, agent_name="ur5e_robotiq", held_object=None, obj_on_table=True):
        self.agent_name = agent_name
        obj_z = 0.05 if obj_on_table else 0.25  # lifted if held
        apple = FakeObjectState(
            "apple",
            xpos=np.array([0.3, 0.1, obj_z]),
            contacts={agent_name} if held_object == "apple" else set(),
        )
        board = FakeObjectState(
            "cutting_board",
            xpos=np.array([0.0, 0.5, 0.02]),
            contacts=set(),
        )
        self.objects = {"apple": apple, "cutting_board": board}
        agent_contacts = {"apple"} if held_object == "apple" else set()
        agent = FakeAgentState(agent_name, contacts=agent_contacts)
        setattr(self, agent_name, agent)


class FakeEnv:
    def __init__(self, agent_name="ur5e_robotiq"):
        self.agent_name = agent_name
        self.agent_configs = {agent_name: {}}
        self._step_count = 0
        self._held_object: Optional[str] = None
        self.physics = MagicMock()
        self.physics.model.actuator_ctrlrange = np.tile([-np.pi, np.pi], (64, 1))
        self.physics.named.data.qpos = MagicMock()
        self.physics.named.data.qpos.__getitem__ = MagicMock(return_value=np.radians(45))

    def get_graspable_objects(self):
        return {"Alice": ["apple"]}

    def get_allowed_collision_pairs(self):
        return []

    def get_grasp_site(self, obj_name):
        return f"{obj_name}_top"

    def get_target_pos(self, agent_name, target_name):
        if target_name == "cutting_board":
            return np.array([0.0, 0.5, 0.05])
        return None

    def get_target_quat(self, agent_name, target_name):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_sim_robots(self):
        return {}

    def step(self, sim_action, verbose=False):
        self._step_count += 1
        obs = self._make_obs()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def get_obs(self):
        return self._make_obs()

    def _make_obs(self):
        return FakeObs(self.agent_name, held_object=self._held_object)


class FakeSimRobot:
    def __init__(self, name, n_joints=7):
        self.name = name
        self.joint_idxs_in_ctrl = list(range(n_joints))
        self.joint_idxs_in_qpos = list(range(n_joints))
        self.grasp_actuator = "robotiq_fingers_actuator"
        self.grasp_idx_in_ctrl = n_joints - 1

    @property
    def grasp_idx(self):
        return self.grasp_idx_in_ctrl


# ===========================================================================
# 1. Skill registries
# ===========================================================================

class TestSkillRegistries:
    def test_pick_place_registry_has_correct_skills(self):
        reg = build_pick_place_registry(["Alice", "Bob"])
        names = set(reg._skills.keys())
        assert "PICK" in names
        assert "PLACE" in names
        assert "WAIT" in names

    def test_cabinet_registry_has_open_cabinet(self):
        reg = build_cabinet_registry(["Alice"])
        names = set(reg._skills.keys())
        assert "OPEN_CABINET" in names
        assert "PICK" in names

    def test_sandwich_registry_has_stack_on(self):
        reg = build_sandwich_registry(["Alice", "Bob"])
        names = set(reg._skills.keys())
        assert "STACK_ON" in names
        assert "PICK" in names
        assert "PUT_OBJECT_IN_CONTAINER" not in names

    def test_sweep_registry(self):
        reg = build_sweep_registry(["Alice"])
        names = set(reg._skills.keys())
        assert "SWEEP" in names

    def test_pick_spec_required_args(self):
        reg = build_pick_place_registry(["Alice"])
        pick = reg.get("PICK")
        assert "object" in pick.required_arguments

    def test_wait_has_no_required_args(self):
        reg = build_pick_place_registry(["Alice"])
        wait = reg.get("WAIT")
        assert len(wait.required_arguments) == 0


# ===========================================================================
# 2. Success checkers
# ===========================================================================

class TestPickSuccessChecker:
    def test_fails_when_not_holding(self):
        checker = PickSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs(held_object=None)
        assert not checker.check(env, obs, {"object": "apple"})

    def test_succeeds_when_holding_and_lifted(self):
        checker = PickSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs(held_object="apple", obj_on_table=False)
        # Make object appear lifted
        obs.objects["apple"].xpos = np.array([0.3, 0.1, 0.25])
        obs.objects["apple"].contacts = {"ur5e_robotiq"}
        obs.ur5e_robotiq.contacts = {"apple"}
        assert checker.check(env, obs, {"object": "apple"})

    def test_requires_stable_checks(self):
        checker = PickSuccessChecker(stable_checks=3)
        env = FakeEnv()
        obs = FakeObs(held_object="apple", obj_on_table=False)
        obs.objects["apple"].xpos = np.array([0.3, 0.1, 0.25])
        obs.objects["apple"].contacts = {"ur5e_robotiq"}
        obs.ur5e_robotiq.contacts = {"apple"}
        assert not checker.check(env, obs, {"object": "apple"})
        assert not checker.check(env, obs, {"object": "apple"})
        assert checker.check(env, obs, {"object": "apple"})  # 3rd consecutive

    def test_missing_object_returns_false(self):
        checker = PickSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs()
        assert not checker.check(env, obs, {"object": "nonexistent"})


class TestPlaceSuccessChecker:
    def test_succeeds_when_near_target_and_released(self):
        checker = PlaceSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs(held_object=None)
        # Move apple near the target
        obs.objects["apple"].xpos = np.array([0.01, 0.5, 0.06])
        assert checker.check(env, obs, {"object": "apple", "target": "cutting_board"})

    def test_fails_when_still_held(self):
        checker = PlaceSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs(held_object="apple")
        obs.objects["apple"].xpos = np.array([0.01, 0.5, 0.06])
        assert not checker.check(env, obs, {"object": "apple", "target": "cutting_board"})


class TestOpenCabinetSuccessChecker:
    def test_open_when_angle_sufficient(self):
        checker = OpenCabinetSuccessChecker(stable_checks=1)
        env = FakeEnv()
        # Mock 45° angle
        env.physics.named.data.qpos.__getitem__ = MagicMock(return_value=np.radians(45))
        obs = FakeObs()
        assert checker.check(env, obs, {"door": "left_door"})

    def test_fails_when_angle_small(self):
        checker = OpenCabinetSuccessChecker(stable_checks=1)
        env = FakeEnv()
        env.physics.named.data.qpos.__getitem__ = MagicMock(return_value=np.radians(5))
        obs = FakeObs()
        assert not checker.check(env, obs, {"door": "left_door"})

    def test_unknown_door_returns_false(self):
        checker = OpenCabinetSuccessChecker(stable_checks=1)
        env = FakeEnv()
        obs = FakeObs()
        assert not checker.check(env, obs, {"door": "nonexistent_door"})


class TestBuildSuccessChecker:
    def test_valid_skills(self):
        for skill in ("PICK", "PLACE", "PICK_AND_PLACE", "OPEN_CABINET", "STACK_ON"):
            checker = build_success_checker(skill)
            assert isinstance(checker, object)

    def test_unknown_skill_raises(self):
        with pytest.raises(ValueError, match="No success checker"):
            build_success_checker("UNKNOWN_SKILL")


# ===========================================================================
# 3. MockACTHandle
# ===========================================================================

class TestMockACTHandle:
    def _make_handle(self):
        env = FakeEnv()
        robots = {"ur5e_robotiq": FakeSimRobot("ur5e_robotiq")}
        return MockACTHandle(env=env, robots=robots, chunk_size=5,
                             uncertainty_mode="policy_metadata")

    def test_reset_clears_state(self):
        h = self._make_handle()
        h._action_idx = 99
        h.reset()
        assert h._action_idx == 0
        assert h._action_buffer == []

    def test_parse_instruction_pick(self):
        h = self._make_handle()
        skill, args = h._parse_instruction({"canonical": "PICK(object=apple)"})
        assert skill == "PICK"
        assert args["object"] == "apple"

    def test_parse_instruction_place(self):
        h = self._make_handle()
        skill, args = h._parse_instruction({"canonical": "PLACE(target=cutting_board)"})
        assert skill == "PLACE"
        assert args["target"] == "cutting_board"

    def test_parse_instruction_bare_string(self):
        h = self._make_handle()
        skill, args = h._parse_instruction("WAIT")
        assert skill == "WAIT"
        assert args == {}

    def test_predict_returns_native_chunk(self):
        h = self._make_handle()
        obs = FakeObs()
        low = np.full(7, -np.pi, dtype=np.float32)
        high = np.full(7, np.pi, dtype=np.float32)
        chunk = h.predict_native_chunk(obs, {"canonical": "WAIT()"}, low, high)
        assert isinstance(chunk, NativeActionChunk)
        assert chunk.actions.shape == (5, 7)

    def test_metadata_confidence_in_range(self):
        h = self._make_handle()
        obs = FakeObs()
        low = np.full(7, -np.pi, dtype=np.float32)
        high = np.full(7, np.pi, dtype=np.float32)
        chunk = h.predict_native_chunk(obs, {"canonical": "WAIT()"}, low, high)
        conf = chunk.metadata.get("confidence")
        assert conf is None or (0.0 <= conf <= 1.0)

    def test_health_check(self):
        h = self._make_handle()
        spec = MagicMock()
        spec.schema_hash = "abc"
        meta = h.health_check(spec)
        assert meta["policy_type"] == "act"
        assert meta["chunk_size"] == 5


# ===========================================================================
# 4. SubtaskEnvAdapter
# ===========================================================================

class TestSubtaskEnvAdapter:
    def _make_adapter(self):
        env = FakeEnv()
        robots = {"ur5e_robotiq": FakeSimRobot("ur5e_robotiq", n_joints=7)}
        return SubtaskEnvAdapter(env, robots), env

    def test_action_bounds_shape(self):
        adapter, env = self._make_adapter()
        low, high = adapter.action_bounds(env)
        assert len(low) == 7
        assert len(high) == 7
        assert np.all(low < high)

    def test_state_vector_from_obs(self):
        adapter, env = self._make_adapter()
        obs = FakeObs()
        state = adapter.state_vector(obs)
        assert state.dtype == np.float32
        assert state.ndim == 1

    def test_step_calls_env_step(self):
        adapter, env = self._make_adapter()
        action = np.zeros(7, dtype=np.float32)
        obs = env.get_obs()
        # step should not raise even with the fake env
        result = adapter.step(env, action)
        assert len(result) == 4


# ===========================================================================
# 5. _SimpleQueue
# ===========================================================================

class TestSimpleQueue:
    def test_basic_pop(self):
        actions = np.arange(30, dtype=np.float32).reshape(3, 10)
        q = _SimpleQueue(actions, execution_horizon=2)
        item = q.pop()
        assert np.allclose(item.action, actions[0])
        # execution_horizon=2 caps reported len; 2 actions remain from 3
        assert len(q) == 2

    def test_load_chunk_resets(self):
        actions = np.zeros((5, 7), dtype=np.float32)
        q = _SimpleQueue(actions, execution_horizon=5)
        q.pop()
        new_actions = np.ones((5, 7), dtype=np.float32)
        q.load_chunk(new_actions)
        item = q.pop()
        assert np.allclose(item.action, 1.0)

    def test_empty_raises(self):
        q = _SimpleQueue(np.zeros((1, 3)), execution_horizon=1)
        q.pop()
        with pytest.raises(IndexError):
            q.pop()


# ===========================================================================
# 6. Uncertainty summariser
# ===========================================================================

class TestSummariseUncertainty:
    def test_empty_trace(self):
        result = _summarise_uncertainty([])
        assert result["n_chunks"] == 0

    def test_counts_risk_levels(self):
        trace = [
            {"step": 0, "confidence": 0.9, "uncertainty": 0.1, "risk_level": "low"},
            {"step": 1, "confidence": 0.5, "uncertainty": 0.5, "risk_level": "medium"},
            {"step": 2, "confidence": 0.2, "uncertainty": 0.8, "risk_level": "high"},
        ]
        result = _summarise_uncertainty(trace)
        assert result["n_chunks"] == 3
        assert result["high_risk_chunks"] == 1
        assert result["medium_risk_chunks"] == 1
        assert result["low_risk_chunks"] == 1
        assert abs(result["mean_confidence"] - (0.9 + 0.5 + 0.2) / 3) < 1e-5


# ===========================================================================
# 7. UncertaintyEstimator (all modes)
# ===========================================================================

class TestUncertaintyEstimator:
    def test_none_mode_always_confident(self):
        est = UncertaintyEstimator("none")
        skill = BTSkillCall("Alice", "PICK")
        state = est.estimate(skill, {})
        assert state.confidence == 1.0
        assert state.uncertainty == 0.0
        assert state.risk_level == "low"

    def test_heuristic_baseline_is_low_risk(self):
        est = UncertaintyEstimator("heuristic")
        skill = BTSkillCall("Alice", "PICK")
        state = est.estimate(skill, {}, executor_feedback=None)
        assert state.risk_level == "low"

    def test_heuristic_increases_on_failure(self):
        from rocobench.crie_bt.status import BTStatus, FailureCode, ProgressStage
        from rocobench.crie_bt.types import ExecutionFeedback, FailureState, ProgressState

        est = UncertaintyEstimator("heuristic")
        skill = BTSkillCall("Alice", "PICK")
        fb = ExecutionFeedback(
            skill,
            status=BTStatus.FAILURE,
            progress=ProgressState(stage=ProgressStage.STUCK, stagnant_steps=5),
            failure=FailureState(True, FailureCode.NO_PROGRESS, "warn", "stuck", {}),
        )
        state = est.estimate(skill, {}, executor_feedback=fb)
        assert state.risk_level == "high"

    def test_policy_metadata_reads_confidence(self):
        from rocobench.crie_bt.types import ExecutionFeedback

        est = UncertaintyEstimator("policy_metadata")
        skill = BTSkillCall("Alice", "PICK")
        fb = ExecutionFeedback(skill, raw_info={"confidence": 0.15})
        state = est.estimate(skill, {}, executor_feedback=fb)
        assert state.confidence == pytest.approx(0.15)
        assert state.risk_level == "high"

    def test_ensemble_variance_zero_for_uniform(self):
        est = UncertaintyEstimator("ensemble_variance")
        skill = BTSkillCall("Alice", "PICK")
        uniform_chunk = np.zeros((5, 7), dtype=np.float32)
        state = est.estimate(skill, {}, action_chunk=uniform_chunk)
        assert state.uncertainty == pytest.approx(0.0)
        assert state.risk_level == "low"

    def test_reset_clears_failed_attempts(self):
        from rocobench.crie_bt.status import BTStatus, FailureCode, ProgressStage
        from rocobench.crie_bt.types import ExecutionFeedback, FailureState, ProgressState

        est = UncertaintyEstimator("heuristic")
        skill = BTSkillCall("Alice", "PICK")
        fb = ExecutionFeedback(
            skill,
            status=BTStatus.FAILURE,
            progress=ProgressState(stage=ProgressStage.STUCK, stagnant_steps=1),
            failure=FailureState(True, FailureCode.NO_PROGRESS, "w", "s", {}),
        )
        est.estimate(skill, {}, executor_feedback=fb)
        assert est.failed_attempts == 1
        est.reset()
        assert est.failed_attempts == 0


# ===========================================================================
# 8. SubtaskLearnedExecutor — full loop with a minimal stub handle
# ===========================================================================

class StubHandle:
    """Returns constant zero actions for N calls, then plan-exhausted zeros."""

    def __init__(self, chunk_size=5, n_ctrl=7, max_calls=10):
        self.chunk_size = chunk_size
        self.n_ctrl = n_ctrl
        self.max_calls = max_calls
        self._calls = 0

    def reset(self):
        self._calls = 0

    def predict_native_chunk(self, obs, instruction, action_low, action_high):
        self._calls += 1
        actions = np.zeros((self.chunk_size, self.n_ctrl), dtype=np.float32)
        return NativeActionChunk(actions=actions, metadata={"confidence": 0.85, "plan_success": True})

    def health_check(self, spec):
        return {"policy_type": "act", "chunk_size": self.chunk_size,
                "schema_hash": spec.schema_hash, "action_representation": "joint_ctrl"}


def _make_executor_with_stub(task_name="pack", skill="PICK", max_steps=30):
    env = FakeEnv()
    robots = {"ur5e_robotiq": FakeSimRobot("ur5e_robotiq")}

    stub_handle = StubHandle()
    registry = LearnedPolicyRegistry()
    spec = LearnedPolicySpec(
        policy_id=f"{task_name}_{skill}_ur5e_robotiq_test",
        skill_name=skill,
        agent_name="ur5e_robotiq",
        embodiment_id="ur5e_robotiq",
        task_id=task_name,
        checkpoint="mock://test",
        checkpoint_revision="test",
        policy_type="act",
        schema_hash="test",
        action_representation="joint_ctrl",
        cameras=(),
        max_steps=max_steps,
        execution_horizon=5,
        success_monitor="stable",
        failure_monitors=("NO_PROGRESS",),
    )
    registry.register(spec)

    # Patch validate_static to allow mock:// paths
    registry.validate_static = lambda s: True

    cache = BoundedPolicyHandleCache(loader=lambda s: stub_handle, max_size=1)
    from rocobench.skills.learned.config import LearnedExecutorConfig
    config = LearnedExecutorConfig(task_id=task_name, embodiment_id="ur5e_robotiq")

    executor = SubtaskLearnedExecutor(
        env=env,
        robots=robots,
        policy_registry=registry,
        policy_cache=cache,
        config=config,
        uncertainty_mode="policy_metadata",
        stable_success_checks=2,
        max_steps=max_steps,
    )
    return executor, env, robots


def _make_plan(skill, agent_name="ur5e_robotiq", call_args=None):
    calls = [
        SkillCall(agent_name=agent_name, skill_name=skill,
                  arguments=call_args or {"object": "apple"}, raw_action=""),
    ]
    return SkillPlan(calls=calls, parsed_proposal=f"test:{skill}")


class TestSubtaskLearnedExecutor:
    def test_timeout_when_success_never_fires(self):
        executor, env, _ = _make_executor_with_stub(max_steps=15)
        plan = _make_plan("PICK")
        obs = env.get_obs()
        result = executor.execute(plan, obs)
        # Success checker never passes with zero actions and fake env.
        # NO_PROGRESS monitor may fire before explicit timeout — both are acceptable failures.
        assert result.status in (
            SkillExecutionStatus.TIMEOUT,
            SkillExecutionStatus.EXECUTION_FAILED,
        )
        assert result.num_sim_steps > 0

    def test_artifacts_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor, env, _ = _make_executor_with_stub(max_steps=10)
            plan = _make_plan("PICK")
            obs = env.get_obs()
            executor.execute(plan, obs, artifact_dir=tmpdir)
            files = os.listdir(tmpdir)
            assert "skill_call.json" in files
            assert "instruction.json" in files
            assert "result.json" in files

    def test_instruction_json_has_canonical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor, env, _ = _make_executor_with_stub(max_steps=10)
            plan = _make_plan("PICK", call_args={"object": "apple"})
            obs = env.get_obs()
            executor.execute(plan, obs, artifact_dir=tmpdir)
            with open(os.path.join(tmpdir, "instruction.json")) as f:
                instr = json.load(f)
            assert "canonical" in instr
            assert "PICK" in instr["canonical"]

    def test_policy_registry_miss_returns_invalid_plan(self):
        executor, env, _ = _make_executor_with_stub()
        # Use a skill not registered
        plan = _make_plan("OPEN_CABINET", call_args={"door": "left_door"})
        obs = env.get_obs()
        result = executor.execute(plan, obs)
        assert result.status == SkillExecutionStatus.INVALID_PLAN

    def test_two_active_calls_rejected(self):
        executor, env, _ = _make_executor_with_stub()
        calls = [
            SkillCall("ur5e_robotiq", "PICK", {"object": "apple"}, ""),
            SkillCall("panda", "PICK", {"object": "banana"}, ""),
        ]
        plan = SkillPlan(calls=calls, parsed_proposal="dual")
        obs = env.get_obs()
        result = executor.execute(plan, obs)
        # Two non-WAIT calls is a structural error — reported as EXECUTION_FAILED
        # (LearnedExecutionError path) rather than INVALID_PLAN (PolicyRegistryError path).
        assert result.success is False
        assert "one non-WAIT call" in result.reason or "active" in result.reason.lower()

    def test_not_reentrant(self):
        executor, env, _ = _make_executor_with_stub(max_steps=5)
        executor._active = True  # simulate already running
        plan = _make_plan("PICK")
        obs = env.get_obs()
        result = executor.execute(plan, obs)
        assert result.success is False
        assert "active" in result.reason.lower()
        executor._active = False  # cleanup


# ===========================================================================
# 9. Skill registry __init__ exports
# ===========================================================================

class TestLearnedPackageExports:
    def test_subtask_skills_importable(self):
        from rocobench.skills.learned import subtask_skills  # noqa: F401

    def test_subtask_executor_importable(self):
        from rocobench.skills.learned import subtask_executor  # noqa: F401

    def test_subtask_success_importable(self):
        from rocobench.skills.learned import subtask_success  # noqa: F401

    def test_mock_handle_importable(self):
        from rocobench.skills.learned import mock_handle  # noqa: F401
