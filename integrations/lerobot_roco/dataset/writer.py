"""Atomic episode storage and LeRobot export boundary."""

import json
import os
import shutil
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .manifest import DatasetManifest, atomic_write_json, read_json
from .recorder import EpisodeRecord
from .schema import SkillDataSchema


class EpisodeWriteError(RuntimeError):
    pass


NATIVE_ACTION_KEYS = (
    "ctrl_idxs",
    "ctrl_vals",
    "qpos_idxs",
    "qpos_target",
    "eq_active_idxs",
    "eq_active_vals",
)

LEROBOT_MANAGED_FRAME_KEYS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _episode_dir(root: str, episode_id: str) -> str:
    return os.path.join(root, "episodes", episode_id)


def _quarantine_dir(root: str, episode_id: str) -> str:
    return os.path.join(root, "quarantine", episode_id)


def _stage_dir(root: str, episode_id: str) -> str:
    return os.path.join(root, "_staging", "{}.{}".format(episode_id, uuid.uuid4().hex))


def _image_npz_key(feature_key: str) -> str:
    return feature_key.replace("observation.images.", "images__")


def _feature_from_image_npz_key(npz_key: str) -> str:
    return npz_key.replace("images__", "observation.images.")


def episode_to_arrays(record: EpisodeRecord) -> Dict[str, np.ndarray]:
    if not record.frames:
        raise EpisodeWriteError("cannot serialize an empty episode")
    frame_features = [frame.to_feature_dict() for frame in record.frames]
    arrays: Dict[str, np.ndarray] = {
        "observation.state": np.stack([f["observation.state"] for f in frame_features]).astype(np.float32),
        "action": np.stack([f["action"] for f in frame_features]).astype(np.float32),
        "timestamp": np.asarray([f["timestamp"] for f in frame_features], dtype=np.float64),
        "frame_index": np.asarray([f["frame_index"] for f in frame_features], dtype=np.int64),
        "episode_index": np.asarray([f["episode_index"] for f in frame_features], dtype=np.int64),
        "env_step_index": np.asarray([record.frames[i].env_step_index for i in range(record.frame_count)], dtype=np.int64),
        "task_index": np.asarray([f["task_index"] for f in frame_features], dtype=np.int64),
    }
    for key in sorted(frame_features[0].keys()):
        if key.startswith("observation.images."):
            arrays[_image_npz_key(key)] = np.stack([f[key] for f in frame_features]).astype(np.uint8)
    stages = [frame.subtask_stage or "" for frame in record.frames]
    if any(stages):
        arrays["subtask_stage"] = np.asarray(stages, dtype="U64")
    return arrays


def arrays_to_frame_features(arrays: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    frame_count = int(np.asarray(arrays["action"]).shape[0])
    frames: List[Dict[str, Any]] = []
    image_keys = sorted(key for key in arrays if key.startswith("images__"))
    for i in range(frame_count):
        frame: Dict[str, Any] = {
            "observation.state": np.asarray(arrays["observation.state"][i], dtype=np.float32),
            "action": np.asarray(arrays["action"][i], dtype=np.float32),
            "timestamp": float(arrays["timestamp"][i]),
            "frame_index": int(arrays["frame_index"][i]),
            "episode_index": int(arrays["episode_index"][i]),
            "task_index": int(arrays.get("task_index", np.zeros(frame_count, dtype=np.int64))[i]),
        }
        for image_key in image_keys:
            frame[_feature_from_image_npz_key(image_key)] = np.asarray(arrays[image_key][i], dtype=np.uint8)
        if "subtask_stage" in arrays:
            frame["subtask_stage"] = str(arrays["subtask_stage"][i])
        frames.append(frame)
    return frames


def load_episode_arrays(episode_path: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(episode_path, "frames.npz")) as data:
        return {key: data[key] for key in data.files}


def native_actions_path(episode_path: str) -> str:
    return os.path.join(episode_path, "native_actions.npz")


def _write_native_actions(stage_path: str, record: EpisodeRecord) -> None:
    if not any(frame.native_action for frame in record.frames):
        return
    arrays: Dict[str, np.ndarray] = {
        "num_actions": np.asarray([record.frame_count], dtype=np.int32),
    }
    for index, frame in enumerate(record.frames):
        payload = dict(frame.native_action or {})
        for key in NATIVE_ACTION_KEYS:
            value = payload.get(key)
            if value is None:
                dtype = np.float32 if key.endswith("vals") or key == "qpos_target" else np.int32
                value = np.asarray([], dtype=dtype)
            arrays["action_{:06d}_{}".format(index, key)] = np.ascontiguousarray(value)
    np.savez_compressed(native_actions_path(stage_path), **arrays)


def load_native_action_payloads(episode_path: str) -> List[Dict[str, np.ndarray]]:
    path = native_actions_path(episode_path)
    if not os.path.exists(path):
        return []
    with np.load(path) as data:
        count = int(np.asarray(data["num_actions"])[0])
        payloads = []
        for index in range(count):
            payload = {}
            for key in NATIVE_ACTION_KEYS:
                name = "action_{:06d}_{}".format(index, key)
                if name in data:
                    payload[key] = data[name]
            payloads.append(payload)
    return payloads


def load_episode_metadata(episode_path: str) -> Dict[str, Any]:
    return read_json(os.path.join(episode_path, "metadata.json"))


def _write_episode_payload(stage_path: str, record: EpisodeRecord) -> None:
    os.makedirs(stage_path, exist_ok=False)
    arrays = episode_to_arrays(record)
    np.savez_compressed(os.path.join(stage_path, "frames.npz"), **arrays)
    _write_native_actions(stage_path, record)
    metadata = dict(record.metadata)
    metadata.update(
        {
            "episode_id": record.episode_id,
            "episode_index": record.episode_index,
            "variation_id": record.variation_id,
            "variation": record.variation.to_dict(),
            "schema_hash": record.schema_hash,
            "frame_count": record.frame_count,
        }
    )
    atomic_write_json(os.path.join(stage_path, "metadata.json"), metadata)
    atomic_write_json(os.path.join(stage_path, "variation.json"), record.variation.to_dict())


class AtomicEpisodeWriter:
    """Stage, validate, and atomically commit or quarantine episodes."""

    def __init__(
        self,
        dataset_root: str,
        schema: SkillDataSchema,
        resume: bool = True,
        overwrite: bool = False,
        repo_root: Optional[str] = None,
    ) -> None:
        self.dataset_root = os.path.abspath(dataset_root)
        self.schema = schema
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.repo_root = repo_root
        os.makedirs(os.path.join(self.dataset_root, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_root, "quarantine"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_root, "_staging"), exist_ok=True)
        self._write_schema_if_needed()
        self._write_manifest_if_needed()

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.dataset_root, "dataset_manifest.json")

    @property
    def schema_path(self) -> str:
        return os.path.join(self.dataset_root, "schema.json")

    def _write_schema_if_needed(self) -> None:
        if os.path.exists(self.schema_path):
            existing = SkillDataSchema.from_dict(read_json(self.schema_path))
            if existing.schema_hash != self.schema.schema_hash:
                raise EpisodeWriteError("existing schema hash does not match requested schema")
            return
        atomic_write_json(self.schema_path, self.schema.to_dict())

    def _write_manifest_if_needed(self) -> None:
        if os.path.exists(self.manifest_path):
            manifest = DatasetManifest.from_dict(read_json(self.manifest_path))
            if manifest.schema_hash != self.schema.schema_hash:
                raise EpisodeWriteError("existing manifest schema hash does not match requested schema")
            if self.overwrite:
                manifest = DatasetManifest.create(self.schema, repo_root=self.repo_root)
                atomic_write_json(self.manifest_path, manifest.to_dict())
            return
        manifest = DatasetManifest.create(self.schema, repo_root=self.repo_root)
        atomic_write_json(self.manifest_path, manifest.to_dict())

    def _load_manifest(self) -> DatasetManifest:
        return DatasetManifest.from_dict(read_json(self.manifest_path))

    def _save_manifest(self, manifest: DatasetManifest) -> None:
        atomic_write_json(self.manifest_path, manifest.to_dict())

    def committed_variation_ids(self) -> Dict[str, str]:
        committed: Dict[str, str] = {}
        episodes_path = os.path.join(self.dataset_root, "episodes")
        for episode_id in sorted(os.listdir(episodes_path)):
            meta_path = os.path.join(episodes_path, episode_id, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            metadata = read_json(meta_path)
            variation_id = metadata.get("variation_id")
            if variation_id:
                committed[str(variation_id)] = episode_id
        return committed

    def episode_ids(self) -> List[str]:
        path = os.path.join(self.dataset_root, "episodes")
        return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))

    def is_variation_committed(self, variation_id: str) -> bool:
        return variation_id in self.committed_variation_ids()

    def write_episode(self, record: EpisodeRecord) -> Tuple[str, str]:
        committed = self.committed_variation_ids()
        if record.variation_id in committed and self.resume and not self.overwrite:
            return "skipped", committed[record.variation_id]

        final_path = _episode_dir(self.dataset_root, record.episode_id)
        if os.path.exists(final_path):
            if not self.overwrite:
                raise EpisodeWriteError("episode already exists: {}".format(record.episode_id))
            shutil.rmtree(final_path)

        stage_path = _stage_dir(self.dataset_root, record.episode_id)
        try:
            _write_episode_payload(stage_path, record)
            os.replace(stage_path, final_path)
        except Exception:
            if os.path.exists(stage_path):
                shutil.rmtree(stage_path)
            raise

        manifest = self._load_manifest()
        manifest.episodes_attempted += 1
        manifest.episodes_committed += 1
        self._save_manifest(manifest)
        return "committed", record.episode_id

    def quarantine_episode(
        self,
        episode_id: str,
        code: str,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
        record: Optional[EpisodeRecord] = None,
    ) -> str:
        path = _quarantine_dir(self.dataset_root, episode_id)
        if os.path.exists(path):
            if not self.overwrite:
                return path
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=False)
        payload = {
            "episode_id": episode_id,
            "quarantine_code": code,
            "reason": reason,
            "metadata": dict(metadata or {}),
        }
        atomic_write_json(os.path.join(path, "quarantine.json"), payload)
        if record is not None and record.frames:
            _write_episode_payload(os.path.join(path, "payload"), record)

        manifest = self._load_manifest()
        manifest.episodes_attempted += 1
        manifest.episodes_quarantined += 1
        self._save_manifest(manifest)
        return path


class LeRobotDatasetUnavailable(RuntimeError):
    pass


def _import_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as first_exc:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as second_exc:
            raise LeRobotDatasetUnavailable("LeRobotDataset is not importable") from second_exc
        return LeRobotDataset
    return LeRobotDataset


def _add_lerobot_frame(dataset: Any, frame: Mapping[str, Any], task: Optional[str]) -> None:
    """Add one frame across LeRobot versions with different task APIs."""
    payload = dict(frame)
    try:
        dataset.add_frame(payload, task=task)
        return
    except TypeError:
        pass
    if task is not None and "task" not in payload:
        payload["task"] = task
    try:
        dataset.add_frame(payload)
        return
    except ValueError as exc:
        if "Extra features" not in str(exc):
            raise
    trimmed = {key: value for key, value in payload.items() if key not in LEROBOT_MANAGED_FRAME_KEYS}
    dataset.add_frame(trimmed)


def export_records_to_lerobot(
    records: Iterable[EpisodeRecord],
    schema: SkillDataSchema,
    repo_id: str,
    root: str,
    robot_type: Optional[str] = None,
    use_videos: bool = True,
    image_writer_threads: int = 4,
) -> Any:
    """Write records using the public LeRobotDataset recording API.

    The function intentionally does not write Parquet or MP4 files directly.
    """
    LeRobotDataset = _import_lerobot_dataset()
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "root": root,
        "fps": schema.fps,
        "features": schema.to_lerobot_features(),
        "use_videos": use_videos,
        "image_writer_threads": int(image_writer_threads),
    }
    if robot_type is not None:
        kwargs["robot_type"] = robot_type
    dataset = LeRobotDataset.create(**kwargs)
    for record in records:
        task = record.metadata.get("natural_language_instruction")
        for frame in record.frames:
            _add_lerobot_frame(dataset, frame.to_feature_dict(), task=task)
        dataset.save_episode()
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    return dataset


def _frame_for_lerobot(frame: Mapping[str, Any], schema: SkillDataSchema) -> Dict[str, Any]:
    allowed = set(schema.observation_features.keys())
    allowed.add("action")
    allowed.add("timestamp")
    allowed.add("frame_index")
    allowed.add("episode_index")
    allowed.add("task_index")
    return {key: value for key, value in frame.items() if key in allowed}


def export_local_dataset_to_lerobot(
    dataset_root: str,
    lerobot_root: str,
    repo_id: str,
    robot_type: Optional[str] = None,
    use_videos: bool = True,
    image_writer_threads: int = 4,
) -> Any:
    """Export committed local records with the public LeRobotDataset API."""
    schema = SkillDataSchema.from_dict(read_json(os.path.join(dataset_root, "schema.json")))
    LeRobotDataset = _import_lerobot_dataset()
    create_kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "root": lerobot_root,
        "fps": schema.fps,
        "features": schema.to_lerobot_features(),
        "use_videos": use_videos,
        "image_writer_threads": int(image_writer_threads),
    }
    if robot_type is not None:
        create_kwargs["robot_type"] = robot_type
    try:
        dataset = LeRobotDataset.create(**create_kwargs)
    except TypeError:
        for key in ("use_videos", "image_writer_threads", "robot_type"):
            create_kwargs.pop(key, None)
        dataset = LeRobotDataset.create(**create_kwargs)

    episodes_root = os.path.join(dataset_root, "episodes")
    for episode_id in sorted(os.listdir(episodes_root)):
        episode_path = os.path.join(episodes_root, episode_id)
        if not os.path.isdir(episode_path):
            continue
        metadata = load_episode_metadata(episode_path)
        arrays = load_episode_arrays(episode_path)
        task = metadata.get("natural_language_instruction")
        for frame in arrays_to_frame_features(arrays):
            payload = _frame_for_lerobot(frame, schema)
            _add_lerobot_frame(dataset, payload, task=task)
        dataset.save_episode()
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    atomic_write_json(
        os.path.join(dataset_root, "lerobot_export.json"),
        {
            "repo_id": repo_id,
            "lerobot_root": lerobot_root,
            "robot_type": robot_type,
            "use_videos": bool(use_videos),
        },
    )
    return dataset


def load_committed_episode_records(dataset_root: str) -> List[Dict[str, Any]]:
    episodes = []
    episodes_root = os.path.join(dataset_root, "episodes")
    if not os.path.exists(episodes_root):
        return episodes
    for episode_id in sorted(os.listdir(episodes_root)):
        episode_path = os.path.join(episodes_root, episode_id)
        if not os.path.isdir(episode_path):
            continue
        metadata = load_episode_metadata(episode_path)
        arrays = load_episode_arrays(episode_path)
        episodes.append({"episode_id": episode_id, "metadata": metadata, "arrays": arrays})
    return episodes


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = "{}.tmp".format(path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True))
            f.write("\n")
    os.replace(tmp_path, path)
