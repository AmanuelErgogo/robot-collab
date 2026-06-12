import pytest


@pytest.mark.integration
@pytest.mark.mujoco
def test_pack_bridge_reset_step_requires_mujoco_runtime():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    pytest.skip("Run this integration test in the Python 3.8 RoCo runtime with a live bridge.")
