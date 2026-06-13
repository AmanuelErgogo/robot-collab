"""Action replay for committed local episodes."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from integrations.lerobot_roco.roco_runtime.action_adapter import RoCoActionAdapter, decode_sim_action

from .episode_sampler import VariationSpec, reset_env_for_variation
from .writer import load_episode_arrays, load_episode_metadata, load_native_action_payloads


@dataclass
class ReplayResult:
    episode_id: str
    success: bool
    frame_count: int
    termination_reason: str
    drift: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "success": bool(self.success),
            "frame_count": int(self.frame_count),
            "termination_reason": self.termination_reason,
            "drift": dict(self.drift),
            "details": dict(self.details),
        }


def _state_digest(env: Any) -> Dict[str, np.ndarray]:
    data = env.physics.data
    return {
        "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        "ctrl": np.asarray(data.ctrl, dtype=np.float64).copy(),
    }


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b))


def replay_episode(
    dataset_root: str,
    episode_id: str,
    env: Any,
    active_agent: str = "Alice",
    compare: bool = False,
) -> ReplayResult:
    episode_path = os.path.join(dataset_root, "episodes", episode_id)
    metadata = load_episode_metadata(episode_path)
    arrays = load_episode_arrays(episode_path)
    native_payloads = load_native_action_payloads(episode_path)
    variation = VariationSpec.from_dict(metadata["variation"])
    reset_env_for_variation(env, variation)
    adapter = RoCoActionAdapter(env, active_agent)
    before = _state_digest(env) if compare else {}
    success = False
    termination_reason = "exhausted"
    obs = None
    reward = 0.0
    done = False
    info: Dict[str, Any] = {}
    use_native = len(native_payloads) == int(arrays["action"].shape[0])
    model = env.physics.model
    model_nu = int(getattr(model, "nu", len(getattr(model, "actuator_ctrlrange", []))))
    model_neq = int(getattr(model, "neq", len(getattr(model, "eq_active", []))))
    for index, action in enumerate(arrays["action"]):
        if use_native:
            sim_action = decode_sim_action(native_payloads[index], model_nu=model_nu, model_neq=model_neq)
        else:
            sim_action, _ = adapter.to_sim_action(action)
        obs, reward, done, info = env.step(sim_action, verbose=False)
        if done:
            termination_reason = "environment_done"
            break
    if obs is not None and hasattr(env, "get_reward_done"):
        try:
            reward, done = env.get_reward_done(obs)
        except Exception:
            pass
    if hasattr(env, "get_packed_slot_for_object"):
        try:
            success = env.get_packed_slot_for_object(obs, metadata["object_name"]) == metadata["target_name"]
        except Exception:
            success = False
    else:
        success = bool(done or float(reward) > 0 or info.get("is_success", False))
    if success:
        termination_reason = "success"
    drift: Dict[str, float] = {}
    if compare:
        after = _state_digest(env)
        for key, value in before.items():
            drift[key] = _l2(value, after[key])
    return ReplayResult(
        episode_id=episode_id,
        success=success,
        frame_count=int(arrays["action"].shape[0]),
        termination_reason=termination_reason,
        drift=drift,
        details={
            "expected_success": metadata.get("success"),
            "replay_action_source": "native_sim_action" if use_native else "public_action",
        },
    )
