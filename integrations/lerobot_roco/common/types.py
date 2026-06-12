"""Protocol dataclasses shared by server and client."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None
    field_names: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        data["field_names"] = list(self.field_names)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArraySpec":
        return cls(
            name=str(data["name"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
            low=list(data["low"]) if data.get("low") is not None else None,
            high=list(data["high"]) if data.get("high") is not None else None,
            field_names=tuple(str(x) for x in data.get("field_names", ())),
        )


@dataclass(frozen=True)
class CameraSpec:
    name: str
    height: int
    width: int
    channels: int
    dtype: str = "uint8"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraSpec":
        return cls(
            name=str(data["name"]),
            height=int(data["height"]),
            width=int(data["width"]),
            channels=int(data["channels"]),
            dtype=str(data.get("dtype", "uint8")),
        )


@dataclass(frozen=True)
class RoCoEnvSpec:
    protocol_version: str
    task: str
    task_description: str
    active_agent: str
    passive_agents: Tuple[str, ...]
    max_episode_steps: int
    effective_fps: float
    cameras: Tuple[CameraSpec, ...]
    observation_state: ArraySpec
    action: ArraySpec
    action_mode: str
    success_semantics: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task": self.task,
            "task_description": self.task_description,
            "active_agent": self.active_agent,
            "passive_agents": list(self.passive_agents),
            "max_episode_steps": self.max_episode_steps,
            "effective_fps": self.effective_fps,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "observation_state": self.observation_state.to_dict(),
            "action": self.action.to_dict(),
            "action_mode": self.action_mode,
            "success_semantics": self.success_semantics,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoCoEnvSpec":
        return cls(
            protocol_version=str(data["protocol_version"]),
            task=str(data["task"]),
            task_description=str(data["task_description"]),
            active_agent=str(data["active_agent"]),
            passive_agents=tuple(str(x) for x in data.get("passive_agents", ())),
            max_episode_steps=int(data["max_episode_steps"]),
            effective_fps=float(data["effective_fps"]),
            cameras=tuple(CameraSpec.from_dict(x) for x in data.get("cameras", ())),
            observation_state=ArraySpec.from_dict(data["observation_state"]),
            action=ArraySpec.from_dict(data["action"]),
            action_mode=str(data["action_mode"]),
            success_semantics=str(data["success_semantics"]),
            metadata=dict(data.get("metadata", {})),
        )
