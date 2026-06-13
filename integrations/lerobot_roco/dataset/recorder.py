"""Transition recording for expert skill execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol

import numpy as np

from integrations.lerobot_roco.roco_runtime.action_adapter import build_action_layout, encode_sim_action
from integrations.lerobot_roco.roco_runtime.observation_adapter import RoCoObservationAdapter

from .episode_sampler import VariationSpec
from .schema import SkillDataSchema


class TransitionObserver(Protocol):
    def before_step(self, observation: Any, action: Any, metadata: Mapping[str, Any]) -> None:
        ...

    def after_step(self, result: Mapping[str, Any]) -> None:
        ...


@dataclass
class TransitionFrame:
    observation: Mapping[str, Any]
    action: np.ndarray
    timestamp: float
    frame_index: int
    episode_index: int
    env_step_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    subtask_stage: Optional[str] = None
    native_action: Optional[Mapping[str, np.ndarray]] = None

    def to_feature_dict(self) -> Dict[str, Any]:
        pixels = self.observation["pixels"]
        frame = {
            "observation.state": np.ascontiguousarray(self.observation["agent_pos"], dtype=np.float32),
            "action": np.ascontiguousarray(self.action, dtype=np.float32),
            "timestamp": float(self.timestamp),
            "frame_index": int(self.frame_index),
            "episode_index": int(self.episode_index),
            "task_index": 0,
        }
        for alias, image in sorted(pixels.items()):
            frame["observation.images.{}".format(alias)] = np.ascontiguousarray(image, dtype=np.uint8)
        if self.subtask_stage is not None:
            frame["subtask_stage"] = self.subtask_stage
        return frame


@dataclass
class EpisodeRecord:
    episode_id: str
    episode_index: int
    variation: VariationSpec
    schema_hash: str
    frames: List[TransitionFrame]
    metadata: Mapping[str, Any]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def variation_id(self) -> str:
        return self.variation.variation_id


def _sim_time(env: Any, fps: float) -> float:
    try:
        return float(env.physics.data.time)
    except Exception:
        pass
    step_index = int(getattr(env, "timestep", 0))
    return float(step_index) / float(fps)


def _latest_value(values: np.ndarray) -> np.float32:
    if len(values) == 0:
        raise ValueError("cannot select from empty values")
    if len(values) > 1 and not np.allclose(values, values[-1]):
        raise ValueError("SimAction contains conflicting values for one control index")
    return np.float32(values[-1])


def public_action_from_sim_action(sim_action: Any, action_layout: Any, env: Optional[Any] = None) -> np.ndarray:
    """Extract the active-agent public action from a RoCo SimAction."""
    ctrl_idxs = np.asarray(sim_action.ctrl_idxs, dtype=np.int64)
    ctrl_vals = np.asarray(sim_action.ctrl_vals, dtype=np.float32)
    if ctrl_idxs.shape != ctrl_vals.shape:
        raise ValueError("SimAction ctrl index/value lengths differ")
    qpos_idxs = np.asarray(getattr(sim_action, "qpos_idxs", []), dtype=np.int64)
    qpos_target = np.asarray(getattr(sim_action, "qpos_target", []), dtype=np.float32)
    if qpos_idxs.shape != qpos_target.shape:
        raise ValueError("SimAction qpos index/target lengths differ")
    current_ctrl = None
    if env is not None:
        current_ctrl = np.asarray(env.physics.data.ctrl, dtype=np.float32)

    values = []
    joint_pairs = list(zip(action_layout.joint_ctrl_indices, action_layout.joint_qpos_indices))
    for ctrl_index, qpos_index in joint_pairs:
        matches = np.where(ctrl_idxs == int(ctrl_index))[0]
        if len(matches) > 0:
            values.append(_latest_value(ctrl_vals[matches]))
            continue
        qpos_matches = np.where(qpos_idxs == int(qpos_index))[0]
        if len(qpos_matches) > 0:
            values.append(_latest_value(qpos_target[qpos_matches]))
            continue
        if current_ctrl is None:
            raise ValueError("SimAction omits active joint control index {}".format(ctrl_index))
        values.append(np.float32(current_ctrl[int(ctrl_index)]))

    gripper_matches = np.where(ctrl_idxs == int(action_layout.gripper_ctrl_index))[0]
    if len(gripper_matches) > 0:
        values.append(_latest_value(ctrl_vals[gripper_matches]))
    elif current_ctrl is not None:
        values.append(np.float32(current_ctrl[int(action_layout.gripper_ctrl_index)]))
    else:
        raise ValueError("SimAction omits active gripper control index {}".format(action_layout.gripper_ctrl_index))

    action = np.ascontiguousarray(values, dtype=np.float32)
    if action.shape != (action_layout.action_dim,):
        raise ValueError("public action shape mismatch")
    return action


def _metadata_without_arrays(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            safe[str(key)] = {
                "array_shape": list(value.shape),
                "array_dtype": str(value.dtype),
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


class RocoTransitionObserver:
    """Record pre-action observations and applied public actions."""

    def __init__(
        self,
        env: Any,
        schema: SkillDataSchema,
        active_agent: str,
        episode_index: int,
        camera_aliases: Optional[Mapping[str, str]] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        self.env = env
        self.schema = schema
        self.active_agent = active_agent
        self.episode_index = int(episode_index)
        self.camera_aliases = dict(camera_aliases or schema.camera_aliases)
        front_feature = schema.observation_features["observation.images.front"]
        self.image_height = int(image_height or front_feature.shape[0])
        self.image_width = int(image_width or front_feature.shape[1])
        self.observation_adapter = RoCoObservationAdapter(
            env,
            active_agent,
            self.camera_aliases,
            self.image_height,
            self.image_width,
        )
        self.action_layout = build_action_layout(env, active_agent)
        self.frames: List[TransitionFrame] = []
        self.after_step_results: List[Dict[str, Any]] = []

    def before_step(self, observation: Any, action: Any, metadata: Mapping[str, Any]) -> None:
        del observation
        formatted = self.observation_adapter.format(None)
        public_action = public_action_from_sim_action(action, self.action_layout, env=self.env)
        native_action = encode_sim_action(action)
        if not np.all(np.isfinite(public_action)):
            raise ValueError("public action contains NaN or Inf")
        low = self.schema.action_feature.low
        high = self.schema.action_feature.high
        if low is not None and np.any(public_action < np.asarray(low, dtype=np.float32)):
            raise ValueError("public action below schema bounds")
        if high is not None and np.any(public_action > np.asarray(high, dtype=np.float32)):
            raise ValueError("public action above schema bounds")
        frame_index = len(self.frames)
        self.frames.append(
            TransitionFrame(
                observation=formatted,
                action=public_action,
                timestamp=_sim_time(self.env, self.schema.fps),
                frame_index=frame_index,
                episode_index=self.episode_index,
                env_step_index=int(getattr(self.env, "timestep", frame_index)),
                metadata=_metadata_without_arrays(metadata),
                native_action=native_action,
            )
        )

    def after_step(self, result: Mapping[str, Any]) -> None:
        self.after_step_results.append(_metadata_without_arrays(result))

    def to_episode_record(
        self,
        episode_id: str,
        variation: VariationSpec,
        metadata: Mapping[str, Any],
    ) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=str(episode_id),
            episode_index=self.episode_index,
            variation=variation,
            schema_hash=self.schema.schema_hash,
            frames=list(self.frames),
            metadata=_metadata_without_arrays(metadata),
        )


def derive_subtask_stages(frame_count: int) -> List[Optional[str]]:
    """Return conservative deterministic stages only when enough frames exist."""
    stages = [
        "approach_object",
        "grasp_object",
        "lift_object",
        "transport_to_target",
        "lower_object",
        "release_object",
        "retreat",
    ]
    if frame_count < len(stages):
        return [None] * frame_count
    boundaries = np.linspace(0, frame_count, num=len(stages) + 1, dtype=np.int64)
    assigned: List[Optional[str]] = []
    for idx in range(frame_count):
        stage_index = int(np.searchsorted(boundaries[1:], idx, side="right"))
        assigned.append(stages[min(stage_index, len(stages) - 1)])
    return assigned


def annotate_record_stages(record: EpisodeRecord) -> EpisodeRecord:
    stages = derive_subtask_stages(record.frame_count)
    for frame, stage in zip(record.frames, stages):
        frame.subtask_stage = stage
    return record
