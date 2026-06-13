"""Artifact writing for Phase 4 rollouts."""

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from integrations.lerobot_roco.dataset.manifest import atomic_write_json


class ArtifactError(RuntimeError):
    """Raised when rollout artifacts cannot be written safely."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), sort_keys=True, separators=(",", ":")))
        f.write("\n")


def state_digest_summary(state_digest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not state_digest:
        return {}
    summary: Dict[str, Any] = {}
    for key, value in state_digest.items():
        if isinstance(value, np.ndarray):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            summary[key] = _jsonable(value)
    return summary


@dataclass
class EpisodeArtifactWriter:
    root_dir: str
    episode_name: str
    overwrite: bool = False
    artifact_dir: str = field(init=False)
    _actions: List[np.ndarray] = field(default_factory=list, init=False)
    _states: List[np.ndarray] = field(default_factory=list, init=False)
    _chunk_ids: List[int] = field(default_factory=list, init=False)
    _chunk_offsets: List[int] = field(default_factory=list, init=False)
    _execution_offsets: List[int] = field(default_factory=list, init=False)
    _rewards: List[float] = field(default_factory=list, init=False)
    _frames: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.artifact_dir = os.path.abspath(os.path.join(self.root_dir, self.episode_name))
        if os.path.exists(self.artifact_dir):
            if not self.overwrite:
                raise ArtifactError("artifact directory already exists: {}".format(self.artifact_dir))
            shutil.rmtree(self.artifact_dir)
        os.makedirs(self.artifact_dir, exist_ok=True)

    @property
    def events_path(self) -> str:
        return os.path.join(self.artifact_dir, "events.jsonl")

    @property
    def chunk_trace_path(self) -> str:
        return os.path.join(self.artifact_dir, "policy_chunk_trace.jsonl")

    def write_episode_config(self, config: Mapping[str, Any]) -> None:
        atomic_write_json(os.path.join(self.artifact_dir, "episode_config.json"), _jsonable(config))

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        atomic_write_json(os.path.join(self.artifact_dir, "rollout_manifest.json"), _jsonable(manifest))

    def append_event(self, event: Mapping[str, Any]) -> None:
        append_jsonl(self.events_path, event)

    def append_chunk_trace(self, trace: Mapping[str, Any]) -> None:
        append_jsonl(self.chunk_trace_path, trace)

    def record_transition(
        self,
        env_step: int,
        observation: Mapping[str, Any],
        action: Sequence[float],
        queued_action: Any,
        reward: float,
        info: Mapping[str, Any],
        rendered_frame: Optional[Any] = None,
    ) -> None:
        state = np.asarray(observation.get("agent_pos"), dtype=np.float32)
        self._states.append(np.ascontiguousarray(state, dtype=np.float32))
        self._actions.append(np.ascontiguousarray(action, dtype=np.float32))
        self._chunk_ids.append(int(getattr(queued_action, "chunk_id", -1)))
        self._chunk_offsets.append(int(getattr(queued_action, "chunk_offset", -1)))
        self._execution_offsets.append(int(getattr(queued_action, "execution_offset", -1)))
        self._rewards.append(float(reward))
        if rendered_frame is not None:
            frame = np.asarray(rendered_frame, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                self._frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
        self.append_event(
            {
                "event": "STEP",
                "env_step": int(env_step),
                "reward": float(reward),
                "is_success": bool(dict(info).get("is_success", False)),
                "chunk_id": int(getattr(queued_action, "chunk_id", -1)),
                "chunk_offset": int(getattr(queued_action, "chunk_offset", -1)),
            }
        )

    def _write_video(self, fps: float) -> Optional[str]:
        if not self._frames:
            return None
        path = os.path.join(self.artifact_dir, "video.mp4")
        try:
            import imageio.v2 as imageio
        except Exception:
            self.append_event({"event": "VIDEO_SKIPPED", "reason": "imageio is not installed"})
            return None
        try:
            with imageio.get_writer(path, fps=max(1.0, float(fps))) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
        except Exception as exc:
            self.append_event({"event": "VIDEO_SKIPPED", "reason": str(exc)})
            return None
        return path

    def finalize(
        self,
        result: Mapping[str, Any],
        final_state_digest: Optional[Mapping[str, Any]] = None,
        fps: float = 20.0,
    ) -> Dict[str, Any]:
        actions = np.asarray(self._actions, dtype=np.float32)
        states = np.asarray(self._states, dtype=np.float32)
        np.savez_compressed(
            os.path.join(self.artifact_dir, "actions.npz"),
            actions=actions,
            chunk_ids=np.asarray(self._chunk_ids, dtype=np.int32),
            chunk_offsets=np.asarray(self._chunk_offsets, dtype=np.int32),
            execution_offsets=np.asarray(self._execution_offsets, dtype=np.int32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
        )
        np.savez_compressed(os.path.join(self.artifact_dir, "states.npz"), agent_pos=states)
        final_state = state_digest_summary(final_state_digest)
        atomic_write_json(os.path.join(self.artifact_dir, "final_state.json"), final_state)
        result_json = dict(_jsonable(result))
        result_json["artifact_dir"] = self.artifact_dir
        video_path = self._write_video(fps=fps)
        if video_path is not None:
            result_json["video_path"] = video_path
        atomic_write_json(os.path.join(self.artifact_dir, "result.json"), result_json)
        return result_json

