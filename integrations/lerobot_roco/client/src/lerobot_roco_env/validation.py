"""Gymnasium space construction and observation validation."""

from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoActionError, RoCoObservationError
from integrations.lerobot_roco.common.types import RoCoEnvSpec


def build_observation_space(spec: RoCoEnvSpec) -> gym.Space:
    pixel_spaces = {}
    for camera in spec.cameras:
        pixel_spaces[camera.name] = spaces.Box(
            low=0,
            high=255,
            shape=(camera.height, camera.width, camera.channels),
            dtype=np.uint8,
        )
    return spaces.Dict(
        {
            "pixels": spaces.Dict(pixel_spaces),
            "agent_pos": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(spec.observation_state.shape),
                dtype=np.float32,
            ),
        }
    )


def build_action_space(spec: RoCoEnvSpec) -> gym.Space:
    if spec.action.low is None or spec.action.high is None:
        raise RoCoActionError("Action spec must include finite bounds.", code=ErrorCode.INVALID_ACTION_SHAPE)
    return spaces.Box(
        low=np.asarray(spec.action.low, dtype=np.float32),
        high=np.asarray(spec.action.high, dtype=np.float32),
        shape=tuple(spec.action.shape),
        dtype=np.float32,
    )


def validate_observation(observation_space: gym.Space, obs: Dict[str, Any]) -> None:
    if "pixels" not in obs or "agent_pos" not in obs:
        raise RoCoObservationError("Observation must include pixels and agent_pos.", code=ErrorCode.OBSERVATION_SHAPE_MISMATCH)
    obs = dict(obs)
    obs["agent_pos"] = np.asarray(obs["agent_pos"], dtype=np.float32)
    for key, image in list(obs["pixels"].items()):
        obs["pixels"][key] = np.asarray(image, dtype=np.uint8)
    if not observation_space.contains(obs):
        raise RoCoObservationError("Observation does not match Gymnasium space.", code=ErrorCode.OBSERVATION_SHAPE_MISMATCH)
