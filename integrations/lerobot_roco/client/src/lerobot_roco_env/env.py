"""Gymnasium environment wrapper for the remote RoCo bridge."""

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoActionError, RoCoEpisodeError
from .client import RemoteRoCoClient
from .config import RoCoGymConfig
from .validation import build_action_space, build_observation_space, validate_observation


class RoCoGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        active_agent: str = "Alice",
        render_mode: Optional[str] = "rgb_array",
        request_timeout_ms: int = 30000,
        auto_start_server: bool = False,
        server_command: Optional[Sequence[str]] = None,
        max_episode_steps: Optional[int] = None,
        action_out_of_bounds: str = "reject",
        _client: Optional[Any] = None,
    ) -> None:
        if auto_start_server or server_command is not None:
            raise NotImplementedError("auto_start_server is not implemented in Phase 0.")
        self.config = RoCoGymConfig(
            endpoint=endpoint,
            active_agent=active_agent,
            render_mode=render_mode,
            request_timeout_ms=request_timeout_ms,
            max_episode_steps=max_episode_steps,
            action_out_of_bounds=action_out_of_bounds,
        )
        self.render_mode = render_mode
        self.client = _client or RemoteRoCoClient(endpoint=endpoint, request_timeout_ms=request_timeout_ms)
        self.client.hello()
        self.spec = self.client.get_spec()
        if self.spec.active_agent != active_agent:
            raise ValueError("Server active agent does not match client active_agent.")
        self.task = self.spec.task
        self.task_description = self.spec.task_description
        self._max_episode_steps = max_episode_steps or self.spec.max_episode_steps
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.spec.effective_fps,
        }
        self.observation_space = build_observation_space(self.spec)
        self.action_space = build_action_space(self.spec)
        self._episode_id: Optional[str] = None
        self._step_index = 0
        self._episode_done = False
        self._last_hold_action: Optional[np.ndarray] = None

    def hold_action(self) -> np.ndarray:
        if self._last_hold_action is not None:
            return np.asarray(self._last_hold_action, dtype=np.float32).copy()
        return ((self.action_space.low + self.action_space.high) / 2.0).astype(np.float32)

    def _coerce_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action)
        if arr.dtype.hasobject:
            raise RoCoActionError("Object dtype action is not allowed.", code=ErrorCode.INVALID_ACTION_DTYPE)
        if tuple(arr.shape) != tuple(self.action_space.shape):
            raise RoCoActionError(
                "Action shape mismatch.",
                code=ErrorCode.INVALID_ACTION_SHAPE,
                details={"expected": list(self.action_space.shape), "received": list(arr.shape)},
            )
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RoCoActionError("Action contains NaN or Inf.", code=ErrorCode.NONFINITE_ACTION)
        if self.config.action_out_of_bounds == "reject" and not self.action_space.contains(arr):
            raise RoCoActionError("Action is outside action_space bounds.", code=ErrorCode.ACTION_OUT_OF_BOUNDS)
        if self.config.action_out_of_bounds == "clip":
            arr = np.clip(arr, self.action_space.low, self.action_space.high).astype(np.float32)
        return arr

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._episode_id is not None:
            self.client.close_episode()
            self._episode_id = None
        payload = self.client.reset(seed=seed, active_agent=self.config.active_agent)
        obs = payload["observation"]
        validate_observation(self.observation_space, obs)
        info = dict(payload.get("info", {}))
        info["is_success"] = bool(info.get("is_success", False))
        hold_action = info.get("hold_action")
        if hold_action is not None:
            self._last_hold_action = np.ascontiguousarray(hold_action, dtype=np.float32)
        self._episode_id = str(payload["episode_id"])
        self._step_index = int(payload["step_index"])
        self._episode_done = False
        return obs, info

    def step(self, action: Any):
        if self._episode_id is None or self._episode_done:
            raise RoCoEpisodeError("step() requires an active episode.", code=ErrorCode.EPISODE_NOT_ACTIVE)
        arr = self._coerce_action(action)
        payload = self.client.step(self._episode_id, self._step_index, arr)
        obs = payload["observation"]
        validate_observation(self.observation_space, obs)
        info = dict(payload.get("info", {}))
        info["is_success"] = bool(info.get("is_success", False))
        reward = float(payload["reward"])
        terminated = bool(payload["terminated"] or info["is_success"])
        self._step_index = int(payload["step_index"])
        truncated = bool(payload["truncated"] or (self._step_index >= self._max_episode_steps and not terminated))
        self._episode_done = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        image = np.asarray(self.client.render(), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("render() must return an HWC RGB image.")
        return np.ascontiguousarray(image, dtype=np.uint8)

    def close(self) -> None:
        try:
            if self._episode_id is not None:
                self.client.close_episode()
        except Exception:
            pass
        self._episode_id = None
        try:
            self.client.close()
        except Exception:
            pass
