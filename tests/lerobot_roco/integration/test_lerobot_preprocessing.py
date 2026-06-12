import pytest


@pytest.mark.integration
@pytest.mark.lerobot
def test_lerobot_preprocessing_requires_lerobot_runtime():
    pytest.importorskip("lerobot")
    pytest.skip("Run this integration test in the Python 3.12 LeRobot runtime with a live bridge.")
