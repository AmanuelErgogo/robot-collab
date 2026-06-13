"""Validation for Phase 2 local records and split manifests."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .manifest import atomic_write_json, read_json
from .schema import SkillDataSchema
from .splitter import detect_split_leakage
from .writer import load_episode_arrays, load_episode_metadata


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    episode_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "episode_id": self.episode_id,
        }


@dataclass
class ValidationReport:
    dataset_root: str
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, severity: str = "error", episode_id: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, severity=severity, episode_id=episode_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "ok": self.ok,
            "summary": dict(self.summary),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_markdown(self) -> str:
        lines = [
            "# RoCo Dataset Validation Report",
            "",
            "- dataset_root: `{}`".format(self.dataset_root),
            "- ok: `{}`".format(self.ok),
            "- errors: `{}`".format(len(self.errors)),
            "- warnings: `{}`".format(len(self.warnings)),
            "",
            "## Summary",
        ]
        for key, value in sorted(self.summary.items()):
            lines.append("- {}: `{}`".format(key, value))
        lines.extend(["", "## Issues"])
        if not self.issues:
            lines.append("- none")
        else:
            for issue in self.issues:
                suffix = " episode={}".format(issue.episode_id) if issue.episode_id else ""
                lines.append("- [{}] {}{}: {}".format(issue.severity, issue.code, suffix, issue.message))
        lines.append("")
        return "\n".join(lines)


def _check_shape_dtype(
    report: ValidationReport,
    arrays: Mapping[str, np.ndarray],
    key: str,
    expected_shape_tail: Any,
    expected_dtype: str,
    episode_id: str,
) -> None:
    if key not in arrays:
        report.add("MISSING_FEATURE", "missing {}".format(key), episode_id=episode_id)
        return
    arr = np.asarray(arrays[key])
    if tuple(arr.shape[1:]) != tuple(expected_shape_tail):
        report.add(
            "SCHEMA_ERROR",
            "{} shape tail expected {}, got {}".format(key, list(expected_shape_tail), list(arr.shape[1:])),
            episode_id=episode_id,
        )
    if str(arr.dtype) != str(np.dtype(expected_dtype)):
        report.add(
            "SCHEMA_ERROR",
            "{} dtype expected {}, got {}".format(key, expected_dtype, arr.dtype),
            episode_id=episode_id,
        )


def _check_episode(
    report: ValidationReport,
    schema: SkillDataSchema,
    episode_id: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    min_frames: int,
    max_frames: int,
) -> None:
    frame_count = int(np.asarray(arrays.get("action", [])).shape[0]) if "action" in arrays else 0
    if frame_count < min_frames:
        report.add("EXCESSIVE_SHORT_EPISODE", "episode has {} frames".format(frame_count), episode_id=episode_id)
    if frame_count > max_frames:
        report.add("EXCESSIVE_EPISODE_LENGTH", "episode has {} frames".format(frame_count), episode_id=episode_id)
    if metadata.get("schema_hash") != schema.schema_hash:
        report.add("SCHEMA_ERROR", "metadata schema hash mismatch", episode_id=episode_id)
    if int(metadata.get("frame_count", -1)) != frame_count:
        report.add("VIDEO_ALIGNMENT_ERROR", "metadata frame count does not match arrays", episode_id=episode_id)

    _check_shape_dtype(report, arrays, "observation.state", schema.observation_features["observation.state"].shape, "float32", episode_id)
    _check_shape_dtype(report, arrays, "action", schema.action_feature.shape, "float32", episode_id)
    for feature_key, spec in schema.observation_features.items():
        if not feature_key.startswith("observation.images."):
            continue
        npz_key = feature_key.replace("observation.images.", "images__")
        _check_shape_dtype(report, arrays, npz_key, spec.shape, "uint8", episode_id)
        if npz_key in arrays:
            images = np.asarray(arrays[npz_key])
            if images.size == 0:
                continue
            if np.all(images == images.reshape(-1)[0]):
                report.add("BLANK_CAMERA", "{} appears blank".format(feature_key), severity="warning", episode_id=episode_id)
            if images.shape[0] > 1 and np.all(images[0] == images[-1]):
                report.add("FROZEN_CAMERA", "{} first/last frames match".format(feature_key), severity="warning", episode_id=episode_id)

    for key in ("observation.state", "action"):
        if key in arrays and not np.all(np.isfinite(arrays[key])):
            report.add("NONFINITE_ARRAY", "{} contains NaN or Inf".format(key), episode_id=episode_id)

    if "action" in arrays:
        actions = np.asarray(arrays["action"], dtype=np.float32)
        if schema.action_feature.low is not None:
            low = np.asarray(schema.action_feature.low, dtype=np.float32)
            if np.any(actions < low):
                report.add("ACTION_OUT_OF_BOUNDS", "action below low bound", episode_id=episode_id)
        if schema.action_feature.high is not None:
            high = np.asarray(schema.action_feature.high, dtype=np.float32)
            if np.any(actions > high):
                report.add("ACTION_OUT_OF_BOUNDS", "action above high bound", episode_id=episode_id)

    if "timestamp" in arrays:
        timestamps = np.asarray(arrays["timestamp"], dtype=np.float64)
        if len(timestamps) != frame_count:
            report.add("TIMESTAMP_ERROR", "timestamp count mismatch", episode_id=episode_id)
        if len(timestamps) > 1:
            deltas = np.diff(timestamps)
            if np.any(deltas < -1e-9):
                report.add("TIMESTAMP_ERROR", "timestamps are not monotonic", episode_id=episode_id)
            expected = 1.0 / float(schema.fps)
            if expected > 0 and np.median(deltas) > 0 and abs(float(np.median(deltas)) - expected) > expected * 0.25:
                report.add("CADENCE_WARNING", "median cadence differs from schema fps", severity="warning", episode_id=episode_id)
    else:
        report.add("TIMESTAMP_ERROR", "missing timestamps", episode_id=episode_id)

    if "frame_index" in arrays:
        expected_indices = np.arange(frame_count, dtype=np.int64)
        if not np.array_equal(np.asarray(arrays["frame_index"], dtype=np.int64), expected_indices):
            report.add("FRAME_INDEX_ERROR", "frame_index is not contiguous from zero", episode_id=episode_id)
    else:
        report.add("FRAME_INDEX_ERROR", "missing frame_index", episode_id=episode_id)

    required_meta = [
        "episode_id",
        "seed",
        "variation_id",
        "task_id",
        "skill_id",
        "canonical_skill_call",
        "natural_language_instruction",
        "agent_name",
        "robot_name",
        "object_name",
        "target_name",
        "success",
        "termination_reason",
        "expert_backend",
        "expert_plan_id",
        "schema_hash",
    ]
    for key in required_meta:
        if key not in metadata:
            report.add("METADATA_INCOMPLETE", "missing metadata {}".format(key), episode_id=episode_id)


def _check_lerobot_load(report: ValidationReport, dataset_root: str, require_lerobot: bool) -> None:
    has_lerobot_layout = os.path.exists(os.path.join(dataset_root, "meta", "info.json"))
    if not has_lerobot_layout:
        severity = "error" if require_lerobot else "warning"
        report.add("LEROBOT_LOAD_SKIPPED", "no LeRobot meta/info.json found", severity=severity)
        return
    try:
        try:
            from lerobot.datasets import LeRobotDataset
        except Exception:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(root=dataset_root)
        report.summary["lerobot_len"] = len(dataset)
    except Exception as exc:
        severity = "error" if require_lerobot else "warning"
        report.add("LEROBOT_LOAD_FAILED", str(exc), severity=severity)


def validate_dataset(
    dataset_root: str,
    report_dir: Optional[str] = None,
    require_lerobot: bool = False,
    min_frames: int = 1,
    max_frames: int = 10000,
) -> ValidationReport:
    dataset_root = os.path.abspath(dataset_root)
    report = ValidationReport(dataset_root=dataset_root)
    schema_path = os.path.join(dataset_root, "schema.json")
    manifest_path = os.path.join(dataset_root, "dataset_manifest.json")
    if not os.path.exists(schema_path):
        report.add("SCHEMA_ERROR", "missing schema.json")
        return report
    schema = SkillDataSchema.from_dict(read_json(schema_path))
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        if manifest.get("schema_hash") != schema.schema_hash:
            report.add("SCHEMA_ERROR", "manifest schema hash mismatch")
    else:
        report.add("MANIFEST_ERROR", "missing dataset_manifest.json")

    episodes_root = os.path.join(dataset_root, "episodes")
    episode_ids = sorted(name for name in os.listdir(episodes_root)) if os.path.exists(episodes_root) else []
    seen_variations: Dict[str, str] = {}
    total_frames = 0
    for episode_id in episode_ids:
        episode_path = os.path.join(episodes_root, episode_id)
        if not os.path.isdir(episode_path):
            continue
        try:
            metadata = load_episode_metadata(episode_path)
            arrays = load_episode_arrays(episode_path)
        except Exception as exc:
            report.add("EPISODE_LOAD_ERROR", str(exc), episode_id=episode_id)
            continue
        variation_id = str(metadata.get("variation_id", ""))
        if variation_id in seen_variations:
            report.add("DUPLICATE_VARIATION", "also used by {}".format(seen_variations[variation_id]), episode_id=episode_id)
        seen_variations[variation_id] = episode_id
        if "action" in arrays:
            total_frames += int(arrays["action"].shape[0])
        _check_episode(report, schema, episode_id, metadata, arrays, min_frames, max_frames)

    split_path = os.path.join(dataset_root, "splits.json")
    if os.path.exists(split_path):
        leaks = detect_split_leakage(read_json(split_path))
        for variation_id in leaks:
            report.add("SPLIT_LEAKAGE", "variation appears in multiple splits: {}".format(variation_id))
    else:
        report.add("SPLIT_MISSING", "splits.json not found", severity="warning")

    _check_lerobot_load(report, dataset_root, require_lerobot=require_lerobot)

    report.summary.update(
        {
            "episode_count": len(episode_ids),
            "frame_count": total_frames,
            "schema_hash": schema.schema_hash,
            "variation_count": len(seen_variations),
        }
    )
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        atomic_write_json(os.path.join(report_dir, "validation_report.json"), report.to_dict())
        with open(os.path.join(report_dir, "validation_report.md"), "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
    return report
