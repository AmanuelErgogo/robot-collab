import json
import os
from dataclasses import dataclass

import numpy as np

from integrations.lerobot_roco.evaluation.artifacts import EpisodeArtifactWriter
from integrations.lerobot_roco.evaluation.policy_loader import PolicyInferenceTrace
from integrations.lerobot_roco.evaluation.rollout import (
    ACTION_OUT_OF_BOUNDS,
    NO_PROGRESS,
    SUCCESS,
    NoProgressMonitor,
    run_policy_rollout,
)


class FakeActionSpace:
    def __init__(self):
        self.low = np.asarray([-1.0, 0.0], dtype=np.float32)
        self.high = np.asarray([1.0, 2.0], dtype=np.float32)
        self.shape = (2,)


class FakeEnv:
    def __init__(self, success_after=2):
        self.action_space = FakeActionSpace()
        self.metadata = {"render_fps": 5}
        self.success_after = success_after
        self.step_count = 0
        self.closed = False

    def reset(self, seed=None):
        self.step_count = 0
        return {
            "pixels": {
                "front": np.zeros((4, 5, 3), dtype=np.uint8),
                "active_agent": np.zeros((4, 5, 3), dtype=np.uint8),
            },
            "agent_pos": np.zeros((2,), dtype=np.float32),
        }, {"is_success": False}

    def step(self, action):
        self.step_count += 1
        success = self.step_count >= self.success_after
        return {
            "pixels": {
                "front": np.zeros((4, 5, 3), dtype=np.uint8),
                "active_agent": np.zeros((4, 5, 3), dtype=np.uint8),
            },
            "agent_pos": np.asarray([self.step_count, self.step_count], dtype=np.float32),
        }, float(success), False, False, {"is_success": success}

    def render(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)


class FakePolicy:
    def __init__(self, chunk):
        self.chunk = np.asarray(chunk, dtype=np.float32)
        self.chunk_size = int(self.chunk.shape[0])
        self.input_features = {
            "observation.state": {"shape": [2], "type": "STATE"},
            "observation.images.front": {"shape": [3, 4, 5], "type": "VISUAL"},
            "observation.images.active_agent": {"shape": [3, 4, 5], "type": "VISUAL"},
        }
        self.metadata = {"policy_type": "fake"}
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict_native_chunk(self, policy_inputs, action_low, action_high, tolerance=0.0):
        trace = PolicyInferenceTrace(
            latency_ms=1.5,
            raw_chunk_shape=self.chunk.shape,
            native_chunk_shape=self.chunk.shape,
            raw_chunk_dtype="float32",
            native_chunk_dtype="float32",
            device="cpu",
            validation={"violation_count": 0},
        )
        return self.chunk.copy(), trace


@dataclass(frozen=True)
class FakeConfig:
    execution_horizon: int = 1
    action_bound_tolerance: float = 0.0
    no_progress_window: int = 5
    no_progress_patience: int = 2
    no_progress_state_epsilon: float = 1e-4
    no_progress_action_epsilon: float = 1e-4
    collision_limit: int = None
    max_steps: int = 4
    record_video: bool = False
    render_every_steps: int = 1
    task_instruction: str = None

    def to_dict(self):
        return self.__dict__.copy()


def test_artifact_writer_aligns_actions_states_and_trace(tmp_path):
    writer = EpisodeArtifactWriter(str(tmp_path), "episode_000000")
    obs = {"agent_pos": np.asarray([1.0, 2.0], dtype=np.float32)}
    queued = type("Queued", (), {"chunk_id": 0, "chunk_offset": 0, "execution_offset": 0})()
    writer.write_episode_config({"seed": 1})
    writer.write_manifest({"phase": 4})
    writer.record_transition(1, obs, np.asarray([0.0, 1.0], dtype=np.float32), queued, 0.0, {"is_success": False})
    writer.append_chunk_trace({"env_step": 1, "chunk_id": 0, "chunk_offset": 0})
    writer.finalize({"success": False, "num_env_steps": 1, "termination_reason": "MAX_STEPS"})

    with np.load(os.path.join(writer.artifact_dir, "actions.npz")) as actions:
        assert actions["actions"].shape == (1, 2)
        assert actions["chunk_ids"].tolist() == [0]
    with np.load(os.path.join(writer.artifact_dir, "states.npz")) as states:
        assert states["agent_pos"].shape == (1, 2)
    with open(os.path.join(writer.artifact_dir, "result.json"), "r", encoding="utf-8") as f:
        assert json.load(f)["artifact_dir"] == writer.artifact_dir


def test_rollout_success_resets_policy_and_uses_task_predicate(tmp_path):
    env = FakeEnv(success_after=2)
    policy = FakePolicy(np.asarray([[0.0, 1.0], [0.1, 1.1]], dtype=np.float32))
    writer = EpisodeArtifactWriter(str(tmp_path), "episode_success")

    result = run_policy_rollout(env, policy, FakeConfig(), writer, seed=123)

    assert result.success is True
    assert result.termination_reason == SUCCESS
    assert result.num_env_steps == 2
    assert policy.reset_count == 1


def test_rollout_rejects_out_of_bounds_action(tmp_path):
    env = FakeEnv(success_after=10)
    policy = FakePolicy(np.asarray([[3.0, 1.0]], dtype=np.float32))
    writer = EpisodeArtifactWriter(str(tmp_path), "episode_oob")

    result = run_policy_rollout(env, policy, FakeConfig(), writer, seed=123)

    assert result.success is False
    assert result.termination_reason == ACTION_OUT_OF_BOUNDS
    assert result.num_env_steps == 0


def test_no_progress_monitor_is_conservative():
    monitor = NoProgressMonitor(window=3, patience=2, state_epsilon=1e-6, action_epsilon=1e-6)
    assert monitor.observe([0.0], [0.0]) is False
    assert monitor.observe([0.0], [0.0]) is False
    assert monitor.observe([0.0], [0.0]) is False
    assert monitor.observe([0.0], [0.0]) is True
    monitor.reset()
    assert monitor.observe([0.0], [0.0]) is False
    assert NO_PROGRESS == "NO_PROGRESS"

