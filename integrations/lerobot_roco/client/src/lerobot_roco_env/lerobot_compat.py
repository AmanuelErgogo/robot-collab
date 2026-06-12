import importlib.metadata
from typing import Any, Dict

import gymnasium
import numpy as np


def _jsonable_features(features: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in features.items():
        item = dict(value)
        if "shape" in item:
            item["shape"] = list(item["shape"])
        if "type" in item:
            item["type"] = str(item["type"])
        out[key] = item
    return out


def get_lerobot_compatibility_report(env: Any) -> Dict[str, Any]:
    obs, info = env.reset(seed=0)
    state = np.asarray(obs["agent_pos"], dtype=np.float32)
    pixels = obs["pixels"]
    report: Dict[str, Any] = {
        "python_raw_observation_keys": sorted(obs.keys()),
        "raw_pixels_type": type(pixels).__name__,
        "raw_camera_shapes": {},
        "raw_state_shape": list(state.shape),
        "action_shape": list(env.action_space.shape),
        "task_description": getattr(env, "task_description", ""),
        "is_success_present": "is_success" in info,
        "gymnasium_version": gymnasium.__version__,
    }
    if isinstance(pixels, dict):
        for key, value in pixels.items():
            report["raw_camera_shapes"][key] = list(np.asarray(value).shape)
    else:
        report["raw_camera_shapes"]["pixels"] = list(np.asarray(pixels).shape)
    try:
        report["lerobot_version"] = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        report["lerobot_version"] = None
        report["blocked_reason"] = "LeRobot is not installed in this Python environment."
        return report

    from lerobot.datasets.feature_utils import (
        build_dataset_frame,
        dataset_to_policy_features,
        hw_to_dataset_features,
    )
    from lerobot.utils.constants import ACTION, OBS_STR

    state_names = list(getattr(env.spec.observation_state, "field_names", ()))
    if not state_names:
        state_names = ["state_%d" % i for i in range(state.shape[0])]
    if len(state_names) != state.shape[0]:
        raise ValueError("LeRobot compatibility check found state names/state shape mismatch.")

    observation_hw_features = {name: float for name in state_names}
    image_values = {}
    if isinstance(pixels, dict):
        for camera_name, image in pixels.items():
            arr = np.asarray(image)
            observation_hw_features[camera_name] = tuple(arr.shape)
            image_values[camera_name] = arr
    else:
        arr = np.asarray(pixels)
        observation_hw_features["pixels"] = tuple(arr.shape)
        image_values["pixels"] = arr

    action_names = list(getattr(env.spec.action, "field_names", ()))
    if not action_names:
        action_names = ["action_%d" % i for i in range(env.action_space.shape[0])]
    action_hw_features = {name: float for name in action_names}

    observation_features = hw_to_dataset_features(observation_hw_features, prefix=OBS_STR, use_video=False)
    action_features = hw_to_dataset_features(action_hw_features, prefix=ACTION, use_video=False)
    dataset_features = {**observation_features, **action_features}
    values = {name: float(state[i]) for i, name in enumerate(state_names)}
    values.update(image_values)
    frame = build_dataset_frame(observation_features, values, prefix=OBS_STR)
    policy_features = dataset_to_policy_features(dataset_features)

    report["lerobot_observation_features"] = _jsonable_features(observation_features)
    report["lerobot_action_features"] = _jsonable_features(action_features)
    report["lerobot_policy_features"] = {
        key: {"type": str(value.type), "shape": list(value.shape)} for key, value in policy_features.items()
    }
    report["preprocessed_keys"] = sorted(frame.keys())
    report["preprocessed_shapes"] = {
        key: list(np.asarray(value).shape) for key, value in frame.items()
    }
    report["verified_lerobot_key_mapping"] = {
        "pixels": sorted(key for key in frame if key.startswith("observation.images.")),
        "agent_pos": "observation.state" if "observation.state" in frame else None,
        "action": "action" if "action" in action_features else None,
    }
    return report
