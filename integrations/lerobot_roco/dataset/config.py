"""Configuration loading for Phase 2 collection."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from integrations.lerobot_roco.roco_runtime.config import default_camera_aliases

from .episode_sampler import DEFAULT_OBJECTS, DEFAULT_TARGETS


@dataclass
class DatasetCollectionConfig:
    task_id: str = "pack"
    skill_id: str = "PUT_OBJECT_IN_CONTAINER"
    active_agent: str = "Alice"
    passive_agents: Tuple[str, ...] = ("Bob",)
    object_names: Tuple[str, ...] = DEFAULT_OBJECTS
    target_names: Tuple[str, ...] = DEFAULT_TARGETS
    image_height: int = 256
    image_width: int = 256
    camera_aliases: Mapping[str, str] = field(default_factory=lambda: default_camera_aliases("Alice"))
    fps: Optional[float] = None
    max_episode_steps: int = 300
    min_episode_frames: int = 1
    max_episode_frames: int = 10000
    expert_backend: str = "rrt"
    expert_place_target_overrides: Mapping[str, str] = field(default_factory=dict)
    action_representation: str = "absolute_joint_position_plus_gripper"
    output_root: str = "artifacts/datasets/pack_put_object_v1"
    repo_id: str = "local/roco-pack-put-object"
    robot_type: str = "roco_pack_alice"
    use_videos: bool = True
    resume: bool = True
    overwrite: bool = False
    randomize_init: bool = True
    render_point_cloud: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DatasetCollectionConfig":
        values: Dict[str, Any] = dict(data)
        for key in ("passive_agents", "object_names", "target_names"):
            if key in values and values[key] is not None:
                values[key] = tuple(str(x) for x in values[key])
        if "camera_aliases" not in values or values["camera_aliases"] is None:
            active_agent = str(values.get("active_agent", "Alice"))
            values["camera_aliases"] = default_camera_aliases(active_agent)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetCollectionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, Mapping):
            raise ValueError("dataset config must be a mapping")
        return cls.from_mapping(data)

    def with_overrides(self, **overrides: Any) -> "DatasetCollectionConfig":
        data = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return self.from_mapping(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "skill_id": self.skill_id,
            "active_agent": self.active_agent,
            "passive_agents": list(self.passive_agents),
            "object_names": list(self.object_names),
            "target_names": list(self.target_names),
            "image_height": int(self.image_height),
            "image_width": int(self.image_width),
            "camera_aliases": dict(self.camera_aliases),
            "fps": self.fps,
            "max_episode_steps": int(self.max_episode_steps),
            "min_episode_frames": int(self.min_episode_frames),
            "max_episode_frames": int(self.max_episode_frames),
            "expert_backend": self.expert_backend,
            "expert_place_target_overrides": dict(self.expert_place_target_overrides),
            "action_representation": self.action_representation,
            "output_root": self.output_root,
            "repo_id": self.repo_id,
            "robot_type": self.robot_type,
            "use_videos": bool(self.use_videos),
            "resume": bool(self.resume),
            "overwrite": bool(self.overwrite),
            "randomize_init": bool(self.randomize_init),
            "render_point_cloud": bool(self.render_point_cloud),
        }

    def resolved_output_root(self, output_root: Optional[str] = None) -> str:
        return os.path.abspath(output_root or self.output_root)
