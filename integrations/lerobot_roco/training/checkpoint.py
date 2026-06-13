"""Checkpoint inspection and native action postprocessing for Phase 3."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from integrations.lerobot_roco.dataset.manifest import read_json
from integrations.lerobot_roco.dataset.schema import FeatureSpec, SkillDataSchema

from .feature_contract import PolicyFeatureContract


class CheckpointValidationError(RuntimeError):
    pass


@dataclass
class CheckpointInspection:
    checkpoint_dir: str
    pretrained_model_dir: Optional[str]
    files_present: Mapping[str, bool]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    issues: List[Mapping[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not [issue for issue in self.issues if issue.get("severity") == "error"]

    def add(self, code: str, message: str, severity: str = "error") -> None:
        self.issues.append({"code": code, "message": message, "severity": severity})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "pretrained_model_dir": self.pretrained_model_dir,
            "files_present": dict(self.files_present),
            "metadata": dict(self.metadata),
            "ok": self.ok,
            "issues": list(self.issues),
        }


def find_pretrained_model_dir(path: str) -> Optional[str]:
    candidates = [
        path,
        os.path.join(path, "pretrained_model"),
        os.path.join(path, "checkpoints", "last", "pretrained_model"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return os.path.abspath(candidate)
    return None


def inspect_checkpoint(checkpoint_dir: str, dataset_root: Optional[str] = None) -> CheckpointInspection:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    model_dir = find_pretrained_model_dir(checkpoint_dir)
    files = {
        "config.json": bool(model_dir and os.path.exists(os.path.join(model_dir, "config.json"))),
        "model.safetensors": bool(model_dir and os.path.exists(os.path.join(model_dir, "model.safetensors"))),
        "train_config.json": bool(model_dir and os.path.exists(os.path.join(model_dir, "train_config.json"))),
    }
    inspection = CheckpointInspection(
        checkpoint_dir=checkpoint_dir,
        pretrained_model_dir=model_dir,
        files_present=files,
    )
    if model_dir is None:
        inspection.add("CHECKPOINT_CONFIG_MISSING", "could not find pretrained_model/config.json")
        return inspection
    for filename, present in files.items():
        if not present:
            inspection.add("CHECKPOINT_FILE_MISSING", "missing {}".format(filename))

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        policy_config = read_json(config_path)
        inspection.metadata["policy_config"] = policy_config
        if policy_config.get("type") != "act":
            inspection.add("POLICY_TYPE_MISMATCH", "expected ACT policy config")

    train_config_path = os.path.join(model_dir, "train_config.json")
    if os.path.exists(train_config_path):
        inspection.metadata["train_config"] = read_json(train_config_path)

    if dataset_root:
        schema_path = os.path.join(dataset_root, "schema.json")
        if os.path.exists(schema_path):
            schema = SkillDataSchema.from_dict(read_json(schema_path))
            contract = PolicyFeatureContract.from_schema(schema)
            _validate_policy_config_features(inspection, contract)
        else:
            inspection.add("DATASET_SCHEMA_MISSING", "dataset schema unavailable for feature comparison")

    return inspection


def _shape_tuple(value: Any) -> tuple:
    return tuple(int(x) for x in value)


def _validate_policy_config_features(inspection: CheckpointInspection, contract: PolicyFeatureContract) -> None:
    policy_config = dict(inspection.metadata.get("policy_config", {}))
    input_features = policy_config.get("input_features")
    output_features = policy_config.get("output_features")
    if not input_features or not output_features:
        inspection.add(
            "CHECKPOINT_FEATURE_METADATA_MISSING",
            "policy config has no input_features/output_features",
            severity="warning",
        )
        return
    expected_inputs = set(contract.input_features.keys())
    expected_outputs = set(contract.output_features.keys())
    if set(input_features.keys()) != expected_inputs:
        inspection.add(
            "CHECKPOINT_INPUT_FEATURE_KEYS_MISMATCH",
            "expected {}, got {}".format(sorted(expected_inputs), sorted(input_features.keys())),
        )
    if set(output_features.keys()) != expected_outputs:
        inspection.add(
            "CHECKPOINT_OUTPUT_FEATURE_KEYS_MISMATCH",
            "expected {}, got {}".format(sorted(expected_outputs), sorted(output_features.keys())),
        )
    for key, spec in contract.input_features.items():
        if key not in input_features:
            continue
        shape = _shape_tuple(input_features[key].get("shape", ()))
        expected = tuple(spec.shape)
        if key.startswith("observation.images.") and len(expected) == 3:
            expected = (expected[2], expected[0], expected[1])
        if shape != expected:
            inspection.add("CHECKPOINT_INPUT_SHAPE_MISMATCH", "{} expected {}, got {}".format(key, expected, shape))
    for key, spec in contract.output_features.items():
        if key in output_features and _shape_tuple(output_features[key].get("shape", ())) != tuple(spec.shape):
            inspection.add(
                "CHECKPOINT_OUTPUT_SHAPE_MISMATCH",
                "{} expected {}, got {}".format(key, spec.shape, output_features[key].get("shape")),
            )


def postprocess_action_to_native(
    action: Any,
    action_feature: FeatureSpec,
    normalization_mode: str = "identity",
    stats: Optional[Mapping[str, Any]] = None,
    validate_bounds: bool = True,
) -> np.ndarray:
    """Convert model output to native simulator action units.

    This helper is intentionally independent from LeRobot.  It is used by tests
    and checkpoint inspection to make normalized-vs-native assumptions explicit.
    """

    arr = np.asarray(action, dtype=np.float32)
    expected_tail = tuple(action_feature.shape)
    if expected_tail and tuple(arr.shape[-len(expected_tail) :]) != expected_tail:
        raise CheckpointValidationError(
            "action shape tail expected {}, got {}".format(expected_tail, tuple(arr.shape))
        )
    mode = str(normalization_mode or "identity").lower()
    if mode in ("identity", "none"):
        native = arr.astype(np.float32, copy=True)
    elif mode in ("mean_std", "mean-std"):
        if not stats or "mean" not in stats or "std" not in stats:
            raise CheckpointValidationError("mean_std postprocessing requires mean and std stats")
        native = arr * np.asarray(stats["std"], dtype=np.float32) + np.asarray(stats["mean"], dtype=np.float32)
    elif mode in ("min_max", "min-max"):
        if not stats or "min" not in stats or "max" not in stats:
            raise CheckpointValidationError("min_max postprocessing requires min and max stats")
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
        native = ((arr + 1.0) * 0.5) * (high - low) + low
    else:
        raise CheckpointValidationError("unsupported normalization mode: {}".format(normalization_mode))

    if not np.all(np.isfinite(native)):
        raise CheckpointValidationError("native action contains NaN or Inf")
    if validate_bounds:
        if action_feature.low is not None:
            low_bounds = np.asarray(action_feature.low, dtype=np.float32)
            if np.any(native < low_bounds - 1e-6):
                raise CheckpointValidationError("native action is below schema low bounds")
        if action_feature.high is not None:
            high_bounds = np.asarray(action_feature.high, dtype=np.float32)
            if np.any(native > high_bounds + 1e-6):
                raise CheckpointValidationError("native action is above schema high bounds")
    return native.astype(np.float32, copy=False)


def build_clean_reload_command(checkpoint_dir: str, dataset_root: str) -> List[str]:
    return [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "scripts", "inspect_roco_checkpoint.py"),
        "--checkpoint-dir",
        checkpoint_dir,
        "--dataset-root",
        dataset_root,
        "--metadata-only",
    ]


def run_clean_metadata_reload(checkpoint_dir: str, dataset_root: str) -> int:
    command = build_clean_reload_command(checkpoint_dir, dataset_root)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode)
