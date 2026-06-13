import numpy as np
import pytest

from integrations.lerobot_roco.evaluation.processor_adapter import (
    ProcessorContractError,
    RoCoPolicyObservationAdapter,
)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_processor_adapter_maps_raw_roco_observation_to_policy_inputs():
    adapter = RoCoPolicyObservationAdapter(
        {
            "observation.state": {"shape": [2], "type": "STATE"},
            "observation.images.front": {"shape": [3, 4, 5], "type": "VISUAL"},
            "observation.images.active_agent": {"shape": [3, 4, 5], "type": "VISUAL"},
        }
    )
    front = np.full((4, 5, 3), 255, dtype=np.uint8)
    active = np.zeros((4, 5, 3), dtype=np.uint8)
    obs = {
        "pixels": {"front": front, "active_agent": active},
        "agent_pos": np.asarray([1.0, 2.0], dtype=np.float32),
    }

    policy_inputs = adapter.to_policy_inputs(obs)

    front = _as_numpy(policy_inputs["observation.images.front"])
    assert tuple(policy_inputs["observation.state"].shape) == (2,)
    assert tuple(policy_inputs["observation.images.front"].shape) == (3, 4, 5)
    assert front.dtype == np.float32
    assert float(front.max()) == 1.0
    assert adapter.last_trace is not None
    assert adapter.last_trace.raw_camera_shapes["front"] == (4, 5, 3)


def test_processor_adapter_rejects_wrong_state_and_missing_instruction():
    with pytest.raises(ProcessorContractError):
        RoCoPolicyObservationAdapter(
            {
                "observation.state": {"shape": [2], "type": "STATE"},
                "task": {"shape": [], "type": "TEXT"},
            }
        ).to_policy_inputs({"pixels": {}, "agent_pos": np.zeros((2,), dtype=np.float32)})

    adapter = RoCoPolicyObservationAdapter({"observation.state": {"shape": [3], "type": "STATE"}})
    with pytest.raises(ProcessorContractError):
        adapter.to_policy_inputs({"pixels": {}, "agent_pos": np.zeros((2,), dtype=np.float32)})
