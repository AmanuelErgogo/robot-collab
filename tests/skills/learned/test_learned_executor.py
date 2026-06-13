import json
import os

import numpy as np

from rocobench.skills.learned.config import LearnedExecutorConfig
from rocobench.skills.learned.executor import LearnedSkillExecutor
from rocobench.skills.learned.fallback import FallbackConfig
from rocobench.skills.learned.models import LearnedPolicySpec, MutableCancellationToken
from rocobench.skills.learned.policy_handle import LearnedPolicyHandle, NativeActionChunk
from rocobench.skills.learned.registry import LearnedPolicyRegistry
from rocobench.skills.models import SkillCall, SkillExecutionStatus, SkillPlan


class FakeActionSpace:
    low = np.asarray([-1.0, 0.0], dtype=np.float32)
    high = np.asarray([1.0, 2.0], dtype=np.float32)


class FakeEnv:
    def __init__(self, success_after=2, constant_state=False):
        self.action_space = FakeActionSpace()
        self.robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}
        self.steps = 0
        self.success_after = success_after
        self.constant_state = constant_state
        self.restored = False
        self.packed_slot = None

    def step(self, action):
        self.steps += 1
        if self.steps >= self.success_after:
            self.packed_slot = "bin_front_left"
        state_value = 0.0 if self.constant_state else float(self.steps)
        return {"agent_pos": np.asarray([state_value, state_value], dtype=np.float32)}, 0.0, False, {"is_success": False}

    def get_agent_held_object(self, obs, agent_name):
        return None

    def get_packed_slot_for_object(self, obs, object_name):
        return self.packed_slot

    def get_slot_occupancy(self, obs):
        return {"bin_front_left": "apple" if self.packed_slot == "bin_front_left" else None}

    def snapshot(self):
        return {"steps": self.steps, "packed_slot": self.packed_slot}

    def restore(self, snapshot):
        self.steps = snapshot["steps"]
        self.packed_slot = snapshot["packed_slot"]
        self.restored = True


class FakePolicy(LearnedPolicyHandle):
    def __init__(self, action=None, raises=False):
        self.action = np.asarray(action if action is not None else [[0.0, 1.0], [0.1, 1.1]], dtype=np.float32)
        self.raises = raises
        self.reset_count = 0
        self.predict_count = 0

    def reset(self):
        self.reset_count += 1

    def health_check(self, spec):
        return {
            "schema_hash": spec.schema_hash,
            "action_representation": spec.action_representation,
            "policy_type": spec.policy_type,
            "chunk_size": self.action.shape[0],
        }

    def predict_native_chunk(self, observation, instruction, action_low, action_high):
        self.predict_count += 1
        if self.raises:
            raise RuntimeError("forced inference error")
        return NativeActionChunk(self.action, {"predict_count": self.predict_count})


def make_spec(tmp_path, max_steps=5, failure_monitors=("NO_PROGRESS", "POLICY_INFERENCE_FAILURE")):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    return LearnedPolicySpec(
        policy_id="act_pack_put_alice",
        skill_name="PUT_OBJECT_IN_CONTAINER",
        agent_name="Alice",
        embodiment_id="ur5e_robotiq",
        task_id="pack",
        checkpoint=str(checkpoint),
        checkpoint_revision="rev",
        policy_type="ACT",
        schema_hash="schema",
        action_representation="absolute_joint_position_plus_gripper",
        cameras=("front", "active_agent"),
        max_steps=max_steps,
        execution_horizon=1,
        success_monitor="stable",
        failure_monitors=failure_monitors,
    )


def make_plan():
    return SkillPlan(
        [
            SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin_front_left"}, ""),
            SkillCall("Bob", "WAIT", {}, ""),
        ],
        "EXECUTE",
    )


def make_executor(tmp_path, env, policy, config=None):
    spec = make_spec(tmp_path, max_steps=(config.max_active_calls if False else 5))
    registry = LearnedPolicyRegistry([spec])
    return LearnedSkillExecutor(
        env,
        registry,
        policy_loader=lambda loaded_spec: policy,
        config=config or LearnedExecutorConfig(stable_success_checks=1, no_progress_window=10),
    )


def test_learned_executor_success_state_machine_and_artifacts(tmp_path):
    env = FakeEnv(success_after=2)
    policy = FakePolicy()
    spec = make_spec(tmp_path)
    executor = LearnedSkillExecutor(
        env,
        LearnedPolicyRegistry([spec]),
        policy_loader=lambda loaded_spec: policy,
        config=LearnedExecutorConfig(stable_success_checks=1, no_progress_window=10),
    )

    result = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)}, artifact_dir=str(tmp_path / "artifacts"))

    assert result.success is True
    assert result.status == SkillExecutionStatus.SUCCESS
    assert result.metadata["learned_success"] is True
    assert result.metadata["fallback_used"] is False
    assert policy.reset_count == 1
    assert os.path.exists(tmp_path / "artifacts" / "actions.npz")
    with open(tmp_path / "artifacts" / "state_machine.jsonl", "r", encoding="utf-8") as f:
        states = [json.loads(line)["state"] for line in f]
    assert "RUNNING" in states
    assert "SUCCEEDED" in states
    assert states[-1] == "CLOSED"


def test_learned_executor_timeout_and_no_hidden_rollback(tmp_path):
    env = FakeEnv(success_after=999)
    snapshot = env.snapshot()
    spec = make_spec(tmp_path, max_steps=2, failure_monitors=("POLICY_INFERENCE_FAILURE",))
    policy = FakePolicy()
    executor = LearnedSkillExecutor(
        env,
        LearnedPolicyRegistry([spec]),
        policy_loader=lambda loaded_spec: policy,
        config=LearnedExecutorConfig(stable_success_checks=1, no_progress_window=10),
    )

    result = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})

    assert result.success is False
    assert result.status == SkillExecutionStatus.TIMEOUT
    assert result.metadata["failure_code"] == "TIMEOUT"
    assert env.restored is False
    env.restore(snapshot)
    assert env.restored is True


def test_learned_executor_inference_error_requests_fallback(tmp_path):
    env = FakeEnv(success_after=999)
    spec = make_spec(tmp_path, max_steps=4)
    policy = FakePolicy(raises=True)
    config = LearnedExecutorConfig(
        stable_success_checks=1,
        fallback=FallbackConfig(mode="return_request", allowed_reasons=("POLICY_INFERENCE_FAILURE", "NO_PROGRESS")),
    )
    executor = LearnedSkillExecutor(env, LearnedPolicyRegistry([spec]), policy_loader=lambda loaded_spec: policy, config=config)

    result = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})

    assert result.success is False
    assert result.metadata["failure_code"] == "POLICY_INFERENCE_FAILURE"
    assert result.metadata["fallback_recommended"] is True
    assert result.metadata["fallback_used"] is False


def test_learned_executor_out_of_bounds_and_cancellation(tmp_path):
    env = FakeEnv(success_after=999)
    spec = make_spec(tmp_path)
    policy = FakePolicy(action=[[3.0, 1.0]])
    executor = LearnedSkillExecutor(
        env,
        LearnedPolicyRegistry([spec]),
        policy_loader=lambda loaded_spec: policy,
        config=LearnedExecutorConfig(stable_success_checks=1),
    )
    result = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})
    assert result.metadata["failure_code"] == "ACTION_OUT_OF_BOUNDS"

    token = MutableCancellationToken()
    token.cancel()
    executor = LearnedSkillExecutor(
        env,
        LearnedPolicyRegistry([spec]),
        policy_loader=lambda loaded_spec: FakePolicy(),
        config=LearnedExecutorConfig(stable_success_checks=1),
        cancellation_token=token,
    )
    result = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})
    assert result.status == SkillExecutionStatus.INTERRUPTED
    assert result.metadata["failure_code"] == "MANUAL_INTERRUPT"


def test_learned_executor_no_progress_and_two_resets(tmp_path):
    env = FakeEnv(success_after=999, constant_state=True)
    spec = make_spec(tmp_path, max_steps=5, failure_monitors=("NO_PROGRESS",))
    policy = FakePolicy(action=[[0.0, 1.0], [0.0, 1.0]])
    config = LearnedExecutorConfig(
        stable_success_checks=1,
        no_progress_window=1,
        no_progress_patience=1,
        fallback=FallbackConfig(mode="return_request", allowed_reasons=("NO_PROGRESS",)),
    )
    executor = LearnedSkillExecutor(env, LearnedPolicyRegistry([spec]), policy_loader=lambda loaded_spec: policy, config=config)

    first = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})
    second = executor.execute(make_plan(), {"agent_pos": np.zeros((2,), dtype=np.float32)})

    assert first.metadata["failure_code"] == "NO_PROGRESS"
    assert second.metadata["failure_code"] == "NO_PROGRESS"
    assert first.metadata["fallback_recommended"] is True
    assert policy.reset_count == 2

