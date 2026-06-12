import numpy as np
import pytest

from integrations.lerobot_roco.common.types import ArraySpec, CameraSpec, RoCoEnvSpec

pytestmark = pytest.mark.client


class FakeGymClient:
    def __init__(self):
        self.closed_episode = False
        self.step_index = 0

    def hello(self):
        return {"selected_protocol_version": "0.1"}

    def get_spec(self):
        return RoCoEnvSpec(
            protocol_version="0.1",
            task="pack",
            task_description="Pack grocery items into the grocery bin.",
            active_agent="Alice",
            passive_agents=("Bob",),
            max_episode_steps=3,
            effective_fps=5.0,
            cameras=(CameraSpec("front", 4, 5, 3),),
            observation_state=ArraySpec("agent_pos", (2,), "float32", field_names=("a", "b")),
            action=ArraySpec("action", (2,), "float32", low=[-1, 0], high=[1, 255], field_names=("j", "g")),
            action_mode="absolute_joint_position_plus_gripper",
            success_semantics="done or reward",
            metadata={},
        )

    def reset(self, seed, active_agent):
        self.step_index = 0
        return {
            "observation": {
                "pixels": {"front": np.zeros((4, 5, 3), dtype=np.uint8)},
                "agent_pos": np.zeros((2,), dtype=np.float32),
            },
            "info": {"is_success": False, "hold_action": np.asarray([0.0, 0.0], dtype=np.float32)},
            "episode_id": "episode",
            "step_index": 0,
        }

    def step(self, episode_id, step_index, action):
        self.step_index = step_index + 1
        return {
            "observation": {
                "pixels": {"front": np.zeros((4, 5, 3), dtype=np.uint8)},
                "agent_pos": np.ones((2,), dtype=np.float32),
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {"is_success": False},
            "episode_id": episode_id,
            "step_index": self.step_index,
        }

    def render(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def close_episode(self):
        self.closed_episode = True

    def close(self):
        pass


def test_gym_env_reset_step_render_close():
    pytest.importorskip("gymnasium")
    from lerobot_roco_env.env import RoCoGymEnv

    client = FakeGymClient()
    env = RoCoGymEnv(_client=client)
    obs, info = env.reset(seed=0)
    assert info["is_success"] is False
    assert env.observation_space.contains(obs)
    action = env.hold_action()
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["is_success"] is False
    assert env.render().shape == (4, 5, 3)
    env.close()
    assert client.closed_episode is True


def test_lerobot_compatibility_report_uses_installed_feature_utils():
    pytest.importorskip("gymnasium")
    pytest.importorskip("lerobot")
    from lerobot_roco_env.env import RoCoGymEnv
    from lerobot_roco_env.lerobot_compat import get_lerobot_compatibility_report

    env = RoCoGymEnv(_client=FakeGymClient())
    report = get_lerobot_compatibility_report(env)
    assert report["lerobot_version"]
    assert report["verified_lerobot_key_mapping"]["agent_pos"] == "observation.state"
    assert report["verified_lerobot_key_mapping"]["pixels"] == ["observation.images.front"]
    assert report["verified_lerobot_key_mapping"]["action"] == "action"
    assert report["lerobot_policy_features"]["observation.images.front"]["shape"] == [3, 4, 5]
    env.close()
