import numpy as np
import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoActionError
from integrations.lerobot_roco.roco_runtime.action_adapter import RoCoActionAdapter, build_action_layout
from conftest import FakeEnv


def test_action_layout_derived_from_robot_and_model():
    env = FakeEnv()
    layout = build_action_layout(env, "Alice")
    assert layout.active_robot_name == "ur5e_robotiq"
    assert layout.joint_ctrl_indices == (0, 1)
    assert layout.joint_qpos_indices == (0, 1)
    assert layout.gripper_ctrl_index == 2
    assert layout.field_names == ("alice_j0", "alice_j1", "alice_gripper")
    np.testing.assert_array_equal(layout.low, np.asarray([-1.0, -2.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(layout.high, np.asarray([1.0, 2.0, 255.0], dtype=np.float32))


def test_action_adapter_builds_active_and_passive_controls():
    env = FakeEnv()
    adapter = RoCoActionAdapter(env, "Alice")
    kwargs, info = adapter.build_sim_action_kwargs(np.asarray([0.5, -0.5, 128.0], dtype=np.float32))
    assert info["action_clipped"] is False
    np.testing.assert_array_equal(kwargs["ctrl_idxs"], np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32))
    np.testing.assert_allclose(kwargs["ctrl_vals"], np.asarray([0.5, -0.5, 128.0, -0.3, -0.4, 40.0], dtype=np.float32))
    np.testing.assert_array_equal(kwargs["qpos_idxs"], np.asarray([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(kwargs["qpos_target"], np.asarray([0.5, -0.5, -0.3, -0.4], dtype=np.float32))


def test_invalid_action_shape_is_rejected():
    adapter = RoCoActionAdapter(FakeEnv(), "Alice")
    with pytest.raises(RoCoActionError) as exc:
        adapter.build_sim_action_kwargs(np.zeros((2,), dtype=np.float32))
    assert exc.value.code == ErrorCode.INVALID_ACTION_SHAPE


def test_nonfinite_action_is_rejected():
    adapter = RoCoActionAdapter(FakeEnv(), "Alice")
    with pytest.raises(RoCoActionError) as exc:
        adapter.build_sim_action_kwargs(np.asarray([0.0, np.nan, 0.0], dtype=np.float32))
    assert exc.value.code == ErrorCode.NONFINITE_ACTION


def test_out_of_bounds_reject_and_clip():
    reject = RoCoActionAdapter(FakeEnv(), "Alice", out_of_bounds="reject")
    with pytest.raises(RoCoActionError) as exc:
        reject.build_sim_action_kwargs(np.asarray([2.0, 0.0, 0.0], dtype=np.float32))
    assert exc.value.code == ErrorCode.ACTION_OUT_OF_BOUNDS

    clip = RoCoActionAdapter(FakeEnv(), "Alice", out_of_bounds="clip")
    kwargs, info = clip.build_sim_action_kwargs(np.asarray([2.0, 0.0, 300.0], dtype=np.float32))
    assert info["action_clipped"] is True
    np.testing.assert_allclose(kwargs["ctrl_vals"][:3], np.asarray([1.0, 0.0, 255.0], dtype=np.float32))
