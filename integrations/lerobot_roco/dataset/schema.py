"""Versioned dataset schema for RoCo skill demonstrations."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from integrations.lerobot_roco.common.protocol import PROTOCOL_VERSION
from integrations.lerobot_roco.common.types import ArraySpec, CameraSpec, RoCoEnvSpec


SCHEMA_VERSION = "1.0"
DEFAULT_TASK_ID = "pack"
DEFAULT_SKILL_ID = "PUT_OBJECT_IN_CONTAINER"
DEFAULT_ACTION_REPRESENTATION = "absolute_joint_position_plus_gripper"


def _tuple_or_none(values: Optional[Iterable[Any]]) -> Optional[Tuple[Any, ...]]:
    if values is None:
        return None
    return tuple(values)


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class FeatureSpec:
    """A fixed-shape numeric or image feature."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    field_names: Tuple[str, ...] = ()
    low: Optional[Tuple[float, ...]] = None
    high: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "dtype", str(np.dtype(self.dtype)) if self.dtype != "image" else "image")
        object.__setattr__(self, "field_names", tuple(str(x) for x in self.field_names))
        object.__setattr__(self, "low", _tuple_or_none(self.low))
        object.__setattr__(self, "high", _tuple_or_none(self.high))
        if self.low is not None and len(self.low) != int(np.prod(self.shape)):
            raise ValueError("low bounds length must match flattened feature shape")
        if self.high is not None and len(self.high) != int(np.prod(self.shape)):
            raise ValueError("high bounds length must match flattened feature shape")

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "field_names": list(self.field_names),
        }
        if self.low is not None:
            data["low"] = [float(x) for x in self.low]
        if self.high is not None:
            data["high"] = [float(x) for x in self.high]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            name=str(data["name"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
            field_names=tuple(str(x) for x in data.get("field_names", ())),
            low=tuple(float(x) for x in data["low"]) if data.get("low") is not None else None,
            high=tuple(float(x) for x in data["high"]) if data.get("high") is not None else None,
        )

    def to_lerobot_feature(self) -> Dict[str, Any]:
        """Return the public LeRobot feature declaration for this feature."""
        if self.dtype == "uint8" and len(self.shape) == 3:
            return {
                "dtype": "image",
                "shape": tuple(self.shape),
                "names": ["height", "width", "channel"],
            }
        names = list(self.field_names) if self.field_names else [self.name]
        return {
            "dtype": self.dtype,
            "shape": tuple(self.shape),
            "names": names,
        }


@dataclass(frozen=True)
class SkillDataSchema:
    """Cross-phase contract for skill demonstration data."""

    schema_version: str
    task_id: str
    skill_id: str
    embodiment_id: str
    observation_features: Mapping[str, FeatureSpec]
    action_feature: FeatureSpec
    fps: float
    action_representation: str
    camera_aliases: Mapping[str, str]
    state_field_names: Tuple[str, ...]
    action_field_names: Tuple[str, ...]
    bridge_protocol_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_features", dict(self.observation_features))
        object.__setattr__(self, "camera_aliases", dict(self.camera_aliases))
        object.__setattr__(self, "state_field_names", tuple(self.state_field_names))
        object.__setattr__(self, "action_field_names", tuple(self.action_field_names))
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if "observation.state" not in self.observation_features:
            raise ValueError("schema requires observation.state")
        if "observation.images.front" not in self.observation_features:
            raise ValueError("schema requires observation.images.front")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "embodiment_id": self.embodiment_id,
            "observation_features": {
                key: value.to_dict() for key, value in sorted(self.observation_features.items())
            },
            "action_feature": self.action_feature.to_dict(),
            "fps": float(self.fps),
            "action_representation": self.action_representation,
            "camera_aliases": dict(sorted(self.camera_aliases.items())),
            "state_field_names": list(self.state_field_names),
            "action_field_names": list(self.action_field_names),
            "bridge_protocol_version": self.bridge_protocol_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillDataSchema":
        return cls(
            schema_version=str(data["schema_version"]),
            task_id=str(data["task_id"]),
            skill_id=str(data["skill_id"]),
            embodiment_id=str(data["embodiment_id"]),
            observation_features={
                str(key): FeatureSpec.from_dict(value)
                for key, value in data["observation_features"].items()
            },
            action_feature=FeatureSpec.from_dict(data["action_feature"]),
            fps=float(data["fps"]),
            action_representation=str(data["action_representation"]),
            camera_aliases={str(k): str(v) for k, v in data["camera_aliases"].items()},
            state_field_names=tuple(str(x) for x in data["state_field_names"]),
            action_field_names=tuple(str(x) for x in data["action_field_names"]),
            bridge_protocol_version=str(data["bridge_protocol_version"]),
        )

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_lerobot_features(self) -> Dict[str, Dict[str, Any]]:
        features = {
            key: value.to_lerobot_feature()
            for key, value in sorted(self.observation_features.items())
        }
        features["action"] = self.action_feature.to_lerobot_feature()
        return features

    def validate_feature_keys(self, frame: Mapping[str, Any]) -> None:
        required = set(self.observation_features.keys())
        required.add("action")
        missing = sorted(required.difference(frame.keys()))
        if missing:
            raise ValueError("frame missing required features: {}".format(missing))


def _spec_from_any(spec: Any) -> RoCoEnvSpec:
    if isinstance(spec, RoCoEnvSpec):
        return spec
    if isinstance(spec, Mapping):
        return RoCoEnvSpec.from_dict(spec)
    if hasattr(spec, "to_dict"):
        return RoCoEnvSpec.from_dict(spec.to_dict())
    raise TypeError("Expected RoCoEnvSpec or mapping")


def _camera_feature(camera: CameraSpec) -> FeatureSpec:
    return FeatureSpec(
        name="observation.images.{}".format(camera.name),
        shape=(camera.height, camera.width, camera.channels),
        dtype=camera.dtype,
    )


def build_schema_from_env_spec(
    env_spec: Any,
    schema_version: str = SCHEMA_VERSION,
    task_id: Optional[str] = None,
    skill_id: str = DEFAULT_SKILL_ID,
) -> SkillDataSchema:
    """Derive the Phase 2 schema from the runtime bridge metadata."""
    spec = _spec_from_any(env_spec)
    observation_features = {}
    for camera in spec.cameras:
        observation_features["observation.images.{}".format(camera.name)] = _camera_feature(camera)
    observation_features["observation.state"] = FeatureSpec(
        name="observation.state",
        shape=spec.observation_state.shape,
        dtype=spec.observation_state.dtype,
        field_names=spec.observation_state.field_names,
    )
    action = spec.action
    action_feature = FeatureSpec(
        name="action",
        shape=action.shape,
        dtype=action.dtype,
        field_names=action.field_names,
        low=tuple(float(x) for x in action.low) if action.low is not None else None,
        high=tuple(float(x) for x in action.high) if action.high is not None else None,
    )
    metadata = dict(spec.metadata)
    return SkillDataSchema(
        schema_version=schema_version,
        task_id=task_id or spec.task or DEFAULT_TASK_ID,
        skill_id=skill_id,
        embodiment_id=str(metadata.get("robot_model", spec.active_agent)),
        observation_features=observation_features,
        action_feature=action_feature,
        fps=float(spec.effective_fps),
        action_representation=spec.action_mode or DEFAULT_ACTION_REPRESENTATION,
        camera_aliases={str(k): str(v) for k, v in metadata.get("camera_aliases", {}).items()},
        state_field_names=tuple(spec.observation_state.field_names),
        action_field_names=tuple(spec.action.field_names),
        bridge_protocol_version=spec.protocol_version or PROTOCOL_VERSION,
    )


def default_schema_for_tests(
    state_dim: int = 3,
    action_dim: int = 2,
    image_shape: Tuple[int, int, int] = (4, 5, 3),
    fps: float = 10.0,
) -> SkillDataSchema:
    """Small schema used by tests and offline tooling."""
    state_names = tuple("state_{}".format(i) for i in range(state_dim))
    action_names = tuple("action_{}".format(i) for i in range(action_dim))
    return SkillDataSchema(
        schema_version=SCHEMA_VERSION,
        task_id=DEFAULT_TASK_ID,
        skill_id=DEFAULT_SKILL_ID,
        embodiment_id="test_robot",
        observation_features={
            "observation.images.front": FeatureSpec("observation.images.front", image_shape, "uint8"),
            "observation.images.active_agent": FeatureSpec("observation.images.active_agent", image_shape, "uint8"),
            "observation.state": FeatureSpec("observation.state", (state_dim,), "float32", state_names),
        },
        action_feature=FeatureSpec(
            "action",
            (action_dim,),
            "float32",
            action_names,
            low=tuple([-1.0] * action_dim),
            high=tuple([1.0] * action_dim),
        ),
        fps=fps,
        action_representation=DEFAULT_ACTION_REPRESENTATION,
        camera_aliases={"front": "teaser", "active_agent": "face_ur5e"},
        state_field_names=state_names,
        action_field_names=action_names,
        bridge_protocol_version=PROTOCOL_VERSION,
    )
