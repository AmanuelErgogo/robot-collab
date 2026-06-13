"""Deterministic variation sampling and replay helpers."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


DEFAULT_OBJECTS = ("apple", "banana", "milk", "soda_can", "bread", "cereal")
DEFAULT_TARGETS = (
    "bin_front_left",
    "bin_front_right",
    "bin_front_middle",
    "bin_back_left",
    "bin_back_right",
    "bin_back_middle",
)


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Mapping[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]


def _as_float_list(values: Iterable[Any]) -> List[float]:
    return [float(x) for x in values]


@dataclass(frozen=True)
class ObjectPose:
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [float(x) for x in self.position],
            "quaternion": [float(x) for x in self.quaternion],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectPose":
        return cls(
            position=tuple(_as_float_list(data["position"])),  # type: ignore[arg-type]
            quaternion=tuple(_as_float_list(data["quaternion"])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class VariationSpec:
    """Canonical replay unit for one skill attempt."""

    seed: int
    variation_index: int
    object_name: str
    target_name: str
    agent_name: str = "Alice"
    object_poses: Mapping[str, ObjectPose] = field(default_factory=dict)
    distractor_arrangement: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "variation_index": int(self.variation_index),
            "object_name": self.object_name,
            "target_name": self.target_name,
            "agent_name": self.agent_name,
            "object_poses": {
                key: value.to_dict() for key, value in sorted(self.object_poses.items())
            },
            "distractor_arrangement": dict(self.distractor_arrangement),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VariationSpec":
        return cls(
            seed=int(data["seed"]),
            variation_index=int(data["variation_index"]),
            object_name=str(data["object_name"]),
            target_name=str(data["target_name"]),
            agent_name=str(data.get("agent_name", "Alice")),
            object_poses={
                str(key): ObjectPose.from_dict(value)
                for key, value in data.get("object_poses", {}).items()
            },
            distractor_arrangement=dict(data.get("distractor_arrangement", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def variation_id(self) -> str:
        return "var_{}".format(stable_hash(self.to_dict()))


class DeterministicVariationSampler:
    """Sample skill variations without depending on hidden RNG call order."""

    def __init__(
        self,
        master_seed: int,
        object_names: Iterable[str] = DEFAULT_OBJECTS,
        target_names: Iterable[str] = DEFAULT_TARGETS,
        agent_name: str = "Alice",
    ) -> None:
        self.master_seed = int(master_seed)
        self.object_names = tuple(str(x) for x in object_names)
        self.target_names = tuple(str(x) for x in target_names)
        self.agent_name = str(agent_name)
        if not self.object_names:
            raise ValueError("object_names must not be empty")
        if not self.target_names:
            raise ValueError("target_names must not be empty")

    def _episode_seed(self, variation_index: int) -> int:
        payload = {
            "master_seed": self.master_seed,
            "variation_index": int(variation_index),
            "sampler": "roco_phase2_v1",
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    def sample(self, variation_index: int) -> VariationSpec:
        seed = self._episode_seed(variation_index)
        rng = np.random.RandomState(seed)
        object_name = self.object_names[int(rng.randint(0, len(self.object_names)))]
        target_name = self.target_names[int(rng.randint(0, len(self.target_names)))]
        distractors = {
            "object_order": list(rng.permutation(self.object_names)),
        }
        return VariationSpec(
            seed=seed,
            variation_index=int(variation_index),
            object_name=object_name,
            target_name=target_name,
            agent_name=self.agent_name,
            distractor_arrangement=distractors,
            metadata={"master_seed": self.master_seed},
        )


class VariationReplayError(RuntimeError):
    pass


def capture_object_poses(env: Any, object_names: Iterable[str]) -> Dict[str, ObjectPose]:
    poses: Dict[str, ObjectPose] = {}
    for name in object_names:
        position = None
        quaternion = None
        try:
            body = env.physics.data.body(name)
            position = np.asarray(body.xpos, dtype=np.float64)
            quaternion = np.asarray(body.xquat, dtype=np.float64)
        except Exception:
            try:
                position, quaternion = env.get_body_pos_quat(name)
                position = np.asarray(position, dtype=np.float64)
                quaternion = np.asarray(quaternion, dtype=np.float64)
            except Exception:
                continue
        if position is None or quaternion is None:
            continue
        poses[str(name)] = ObjectPose(
            position=tuple(float(x) for x in position[:3]),
            quaternion=tuple(float(x) for x in quaternion[:4]),
        )
    return poses


def with_observed_poses(env: Any, variation: VariationSpec, object_names: Iterable[str]) -> VariationSpec:
    poses = capture_object_poses(env, object_names)
    data = variation.to_dict()
    data["object_poses"] = {key: value.to_dict() for key, value in poses.items()}
    return VariationSpec.from_dict(data)


def reset_env_for_variation(env: Any, variation: VariationSpec, reload: bool = True) -> Any:
    """Reset an environment and apply exact object poses when present."""
    if hasattr(env, "seed"):
        env.seed(np_seed=int(variation.seed))

    try:
        obs = env.reset(reload=reload)
    except TypeError:
        obs = env.reset()

    if variation.object_poses:
        if not hasattr(env, "reset_body_pose") or not hasattr(env, "reset_qpos"):
            raise VariationReplayError("environment cannot apply object poses")
        for object_name, pose in sorted(variation.object_poses.items()):
            position = np.asarray(pose.position, dtype=np.float64)
            quaternion = np.asarray(pose.quaternion, dtype=np.float64)
            env.reset_body_pose(object_name, pos=position, quat=quaternion)
            if hasattr(env, "get_object_joint_name"):
                joint_name = env.get_object_joint_name(object_name)
            else:
                joint_name = "{}_joint".format(object_name)
            env.reset_qpos(joint_name, pos=position, quat=quaternion)
        try:
            env.physics.forward()
        except Exception:
            pass
        if hasattr(env, "get_obs"):
            obs = env.get_obs()
    return obs
