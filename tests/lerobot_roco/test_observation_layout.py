import numpy as np
import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoObservationError
from integrations.lerobot_roco.roco_runtime.observation_adapter import RoCoObservationAdapter
from conftest import FakeEnv


def test_observation_adapter_formats_pixels_and_state():
    env = FakeEnv()
    adapter = RoCoObservationAdapter(env, "Alice", {"front": "teaser", "active_agent": "face_ur5e"}, 8, 9)
    obs = adapter.format(object())
    assert sorted(obs.keys()) == ["agent_pos", "pixels"]
    assert sorted(obs["pixels"].keys()) == ["active_agent", "front"]
    assert obs["pixels"]["front"].shape == (8, 9, 3)
    assert obs["pixels"]["front"].dtype == np.uint8
    assert obs["agent_pos"].dtype == np.float32
    assert obs["agent_pos"].shape == (5,)
    np.testing.assert_allclose(
        obs["agent_pos"],
        np.asarray([0.1, 0.2, -0.03, -0.04, 30.0], dtype=np.float32),
    )
    assert adapter.layout.state_field_names == (
        "qpos.alice_j0",
        "qpos.alice_j1",
        "qvel.alice_j0",
        "qvel.alice_j1",
        "ctrl.alice_gripper",
    )


def test_render_dtype_shape_validation():
    env = FakeEnv()

    def bad_render(camera_id, height, width):
        return np.zeros((height, width), dtype=np.uint8)

    env.physics.render = bad_render
    adapter = RoCoObservationAdapter(env, "Alice", {"front": "teaser"}, 8, 9)
    with pytest.raises(RoCoObservationError) as exc:
        adapter.format(object())
    assert exc.value.code == ErrorCode.OBSERVATION_SHAPE_MISMATCH
