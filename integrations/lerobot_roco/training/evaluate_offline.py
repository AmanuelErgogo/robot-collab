"""Offline diagnostic metrics for Phase 3 ACT outputs."""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from integrations.lerobot_roco.dataset.manifest import atomic_write_json, read_json
from integrations.lerobot_roco.dataset.schema import SkillDataSchema


@dataclass
class OfflineMetrics:
    sample_count: int
    action_dim: int
    mae_per_action_dim: List[float]
    mse: float
    chunk_mae_by_horizon: List[float] = field(default_factory=list)
    native_bound_violations: Mapping[str, int] = field(default_factory=dict)
    action_smoothness_l2: Optional[float] = None
    gripper_accuracy: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "action_dim": int(self.action_dim),
            "mae_per_action_dim": [float(x) for x in self.mae_per_action_dim],
            "mse": float(self.mse),
            "chunk_mae_by_horizon": [float(x) for x in self.chunk_mae_by_horizon],
            "native_bound_violations": dict(self.native_bound_violations),
            "action_smoothness_l2": None if self.action_smoothness_l2 is None else float(self.action_smoothness_l2),
            "gripper_accuracy": None if self.gripper_accuracy is None else float(self.gripper_accuracy),
            "notes": list(self.notes),
        }


def _as_action_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("actions must have at least sample and action dimensions")
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions contain NaN or Inf")
    return arr


def _bound_violations(predictions: np.ndarray, low: Optional[List[float]], high: Optional[List[float]]) -> Dict[str, int]:
    flat = predictions.reshape((-1, predictions.shape[-1]))
    violations = {"below_low": 0, "above_high": 0}
    if low is not None:
        low_arr = np.asarray(low, dtype=np.float32)
        violations["below_low"] = int(np.sum(flat < low_arr))
    if high is not None:
        high_arr = np.asarray(high, dtype=np.float32)
        violations["above_high"] = int(np.sum(flat > high_arr))
    return violations


def compute_offline_action_metrics(
    predictions: Any,
    targets: Any,
    action_low: Optional[List[float]] = None,
    action_high: Optional[List[float]] = None,
    gripper_index: Optional[int] = None,
    gripper_threshold: float = 0.5,
) -> OfflineMetrics:
    pred = _as_action_array(predictions)
    target = _as_action_array(targets)
    if pred.shape != target.shape:
        raise ValueError("prediction/target shape mismatch: {} vs {}".format(pred.shape, target.shape))
    action_dim = int(pred.shape[-1])
    flat_pred = pred.reshape((-1, action_dim))
    flat_target = target.reshape((-1, action_dim))
    diff = flat_pred - flat_target
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    mse = float(np.mean(diff ** 2))

    chunk_mae: List[float] = []
    if pred.ndim == 3:
        for horizon in range(pred.shape[1]):
            chunk_mae.append(float(np.mean(np.abs(pred[:, horizon, :] - target[:, horizon, :]))))

    smoothness = None
    if pred.shape[0] > 1:
        smooth_source = pred[:, 0, :] if pred.ndim == 3 else pred
        smoothness = float(np.mean(np.linalg.norm(np.diff(smooth_source, axis=0), axis=-1)))

    gripper_accuracy = None
    if gripper_index is not None:
        pred_gripper = flat_pred[:, int(gripper_index)] >= float(gripper_threshold)
        target_gripper = flat_target[:, int(gripper_index)] >= float(gripper_threshold)
        gripper_accuracy = float(np.mean(pred_gripper == target_gripper))

    return OfflineMetrics(
        sample_count=int(pred.shape[0]),
        action_dim=action_dim,
        mae_per_action_dim=[float(x) for x in mae_per_dim],
        mse=mse,
        chunk_mae_by_horizon=chunk_mae,
        native_bound_violations=_bound_violations(pred, action_low, action_high),
        action_smoothness_l2=smoothness,
        gripper_accuracy=gripper_accuracy,
        notes=["Offline error is diagnostic only; it is not manipulation success."],
    )


def metrics_to_markdown(metrics: OfflineMetrics) -> str:
    lines = [
        "# Phase 3 Offline Evaluation",
        "",
        "- sample_count: `{}`".format(metrics.sample_count),
        "- action_dim: `{}`".format(metrics.action_dim),
        "- mse: `{:.8f}`".format(metrics.mse),
        "- action_smoothness_l2: `{}`".format(metrics.action_smoothness_l2),
        "- gripper_accuracy: `{}`".format(metrics.gripper_accuracy),
        "",
        "## MAE Per Action Dimension",
    ]
    for idx, value in enumerate(metrics.mae_per_action_dim):
        lines.append("- dim_{}: `{:.8f}`".format(idx, value))
    if metrics.chunk_mae_by_horizon:
        lines.extend(["", "## Chunk MAE By Horizon"])
        for idx, value in enumerate(metrics.chunk_mae_by_horizon):
            lines.append("- horizon_{}: `{:.8f}`".format(idx, value))
    lines.extend(["", "## Bound Violations"])
    for key, value in sorted(metrics.native_bound_violations.items()):
        lines.append("- {}: `{}`".format(key, value))
    lines.extend(["", "## Notes"])
    for note in metrics.notes:
        lines.append("- {}".format(note))
    lines.append("")
    return "\n".join(lines)


def evaluate_npz(predictions_npz: str, dataset_root: str, output_dir: str) -> OfflineMetrics:
    with np.load(predictions_npz) as data:
        predictions = data["predictions"]
        targets = data["targets"]
    schema = SkillDataSchema.from_dict(read_json(os.path.join(dataset_root, "schema.json")))
    action = schema.action_feature
    gripper_index = None
    for idx, name in enumerate(action.field_names):
        if "gripper" in name or "finger" in name:
            gripper_index = idx
            break
    metrics = compute_offline_action_metrics(
        predictions,
        targets,
        action_low=list(action.low) if action.low is not None else None,
        action_high=list(action.high) if action.high is not None else None,
        gripper_index=gripper_index,
        gripper_threshold=0.5,
    )
    os.makedirs(output_dir, exist_ok=True)
    atomic_write_json(os.path.join(output_dir, "offline_metrics.json"), metrics.to_dict())
    with open(os.path.join(output_dir, "offline_report.md"), "w", encoding="utf-8") as f:
        f.write(metrics_to_markdown(metrics))
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved ACT predictions against target actions.")
    parser.add_argument("--predictions-npz", required=True, help="NPZ with predictions and targets arrays.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    evaluate_npz(args.predictions_npz, args.dataset_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
