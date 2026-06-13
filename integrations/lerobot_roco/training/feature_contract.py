"""Feature-contract preflight for ACT training."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from integrations.lerobot_roco.common.types import RoCoEnvSpec
from integrations.lerobot_roco.dataset.manifest import read_json
from integrations.lerobot_roco.dataset.schema import FeatureSpec, SkillDataSchema, build_schema_from_env_spec


LEROBOT_MANAGED_FEATURE_KEYS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


class FeatureContractError(RuntimeError):
    """Raised when required feature preflight checks fail."""

    def __init__(self, issues: List["PreflightIssue"]) -> None:
        self.issues = list(issues)
        message = "; ".join("{}: {}".format(issue.code, issue.message) for issue in self.issues)
        RuntimeError.__init__(self, message)


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class PreflightResult:
    dataset_root: str
    lerobot_dataset_root: Optional[str]
    contract: "PolicyFeatureContract"
    issues: List[PreflightIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, severity: str = "error") -> None:
        self.issues.append(PreflightIssue(code=code, message=message, severity=severity))

    def assert_ok(self) -> None:
        if not self.ok:
            raise FeatureContractError(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "lerobot_dataset_root": self.lerobot_dataset_root,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
            "contract": self.contract.to_dict(),
        }


@dataclass(frozen=True)
class PolicyFeatureContract:
    input_features: Mapping[str, FeatureSpec]
    output_features: Mapping[str, FeatureSpec]
    action_representation: str
    fps: float
    dataset_schema_hash: str
    state_field_names: tuple
    action_field_names: tuple
    camera_order: tuple

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_features": {
                key: value.to_dict() for key, value in sorted(self.input_features.items())
            },
            "output_features": {
                key: value.to_dict() for key, value in sorted(self.output_features.items())
            },
            "action_representation": self.action_representation,
            "fps": float(self.fps),
            "dataset_schema_hash": self.dataset_schema_hash,
            "state_field_names": list(self.state_field_names),
            "action_field_names": list(self.action_field_names),
            "camera_order": list(self.camera_order),
        }

    @classmethod
    def from_schema(cls, schema: SkillDataSchema) -> "PolicyFeatureContract":
        input_features = dict(schema.observation_features)
        output_features = {"action": schema.action_feature}
        camera_order = tuple(
            key for key in sorted(input_features.keys()) if key.startswith("observation.images.")
        )
        return cls(
            input_features=input_features,
            output_features=output_features,
            action_representation=schema.action_representation,
            fps=float(schema.fps),
            dataset_schema_hash=schema.schema_hash,
            state_field_names=tuple(schema.state_field_names),
            action_field_names=tuple(schema.action_field_names),
            camera_order=camera_order,
        )


def _load_schema(dataset_root: str) -> SkillDataSchema:
    schema_path = os.path.join(dataset_root, "schema.json")
    if not os.path.exists(schema_path):
        raise FeatureContractError([PreflightIssue("SCHEMA_MISSING", "missing schema.json")])
    return SkillDataSchema.from_dict(read_json(schema_path))


def _shape_tuple(value: Any) -> tuple:
    return tuple(int(x) for x in value)


def _compare_feature_dicts(
    result: PreflightResult,
    expected: Mapping[str, Mapping[str, Any]],
    actual: Mapping[str, Mapping[str, Any]],
    source: str,
    ignored_actual_keys: Optional[tuple] = None,
) -> None:
    ignored_keys = set(ignored_actual_keys or ())
    actual = {key: value for key, value in actual.items() if key not in ignored_keys}
    expected_keys = sorted(expected.keys())
    actual_keys = sorted(actual.keys())
    if expected_keys != actual_keys:
        result.add(
            "LEROBOT_FEATURE_KEYS_MISMATCH",
            "{} feature keys expected {}, got {}".format(source, expected_keys, actual_keys),
        )
        return
    for key in expected_keys:
        exp = expected[key]
        got = actual[key]
        if str(exp.get("dtype")) != str(got.get("dtype")):
            result.add(
                "LEROBOT_FEATURE_DTYPE_MISMATCH",
                "{} dtype expected {}, got {}".format(key, exp.get("dtype"), got.get("dtype")),
            )
        if _shape_tuple(exp.get("shape", ())) != _shape_tuple(got.get("shape", ())):
            result.add(
                "LEROBOT_FEATURE_SHAPE_MISMATCH",
                "{} shape expected {}, got {}".format(key, list(exp.get("shape", ())), list(got.get("shape", ()))),
            )
        exp_names = exp.get("names")
        got_names = got.get("names")
        if exp_names is not None and got_names is not None and list(exp_names) != list(got_names):
            result.add(
                "LEROBOT_FEATURE_NAMES_MISMATCH",
                "{} names expected {}, got {}".format(key, exp_names, got_names),
            )


def _read_lerobot_info(root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_lerobot_stats(root: str) -> bool:
    return os.path.exists(os.path.join(root, "meta", "stats.json")) or os.path.exists(
        os.path.join(root, "meta", "episodes_stats.jsonl")
    )


def _compare_runtime_schema(result: PreflightResult, schema: SkillDataSchema, runtime_env_spec_path: str) -> None:
    if not os.path.exists(runtime_env_spec_path):
        result.add("RUNTIME_SPEC_MISSING", "runtime env spec not found: {}".format(runtime_env_spec_path))
        return
    runtime_spec = RoCoEnvSpec.from_dict(read_json(runtime_env_spec_path))
    runtime_schema = build_schema_from_env_spec(runtime_spec, skill_id=schema.skill_id)
    runtime_contract = PolicyFeatureContract.from_schema(runtime_schema)
    dataset_contract = result.contract
    if runtime_contract.action_representation != dataset_contract.action_representation:
        result.add(
            "RUNTIME_ACTION_REPRESENTATION_MISMATCH",
            "runtime action representation {} does not match dataset {}".format(
                runtime_contract.action_representation,
                dataset_contract.action_representation,
            ),
        )
    if abs(float(runtime_contract.fps) - float(dataset_contract.fps)) > 1e-6:
        result.add(
            "RUNTIME_FPS_MISMATCH",
            "runtime fps {} does not match dataset {}".format(runtime_contract.fps, dataset_contract.fps),
        )
    _compare_feature_dicts(
        result,
        schema.to_lerobot_features(),
        runtime_schema.to_lerobot_features(),
        "runtime",
    )


def run_feature_preflight(
    dataset_root: str,
    lerobot_dataset_root: Optional[str] = None,
    compatibility_lock_path: Optional[str] = None,
    runtime_env_spec_path: Optional[str] = None,
    expected_action_representation: Optional[str] = None,
    require_lerobot_dataset: bool = True,
) -> PreflightResult:
    """Validate dataset, LeRobot export, and optional runtime schema alignment."""

    dataset_root = os.path.abspath(dataset_root)
    lerobot_root_abs = os.path.abspath(lerobot_dataset_root) if lerobot_dataset_root else None
    schema = _load_schema(dataset_root)
    contract = PolicyFeatureContract.from_schema(schema)
    result = PreflightResult(
        dataset_root=dataset_root,
        lerobot_dataset_root=lerobot_root_abs,
        contract=contract,
    )

    manifest_path = os.path.join(dataset_root, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        result.metadata["dataset_manifest"] = manifest
        if manifest.get("schema_hash") != schema.schema_hash:
            result.add("DATASET_SCHEMA_HASH_MISMATCH", "manifest schema_hash does not match schema.json")
        if abs(float(manifest.get("fps", schema.fps)) - float(schema.fps)) > 1e-6:
            result.add("DATASET_FPS_MISMATCH", "manifest fps does not match schema fps")
    else:
        result.add("DATASET_MANIFEST_MISSING", "dataset_manifest.json is required")

    if expected_action_representation and expected_action_representation != schema.action_representation:
        result.add(
            "ACTION_REPRESENTATION_MISMATCH",
            "expected {}, got {}".format(expected_action_representation, schema.action_representation),
        )

    if compatibility_lock_path:
        lock_path = os.path.abspath(compatibility_lock_path)
        if os.path.exists(lock_path):
            lock = read_json(lock_path)
            result.metadata["compatibility_lock"] = lock
            if lock.get("schema_hash") and lock.get("schema_hash") != schema.schema_hash:
                result.add("LOCK_SCHEMA_HASH_MISMATCH", "compatibility lock schema_hash differs from dataset")
            if lock.get("bridge_protocol") and lock.get("bridge_protocol") != schema.bridge_protocol_version:
                result.add("LOCK_PROTOCOL_MISMATCH", "compatibility lock bridge_protocol differs from dataset")
            if not lock.get("lerobot_version") or not lock.get("lerobot_commit"):
                result.add(
                    "LEROBOT_PIN_INCOMPLETE",
                    "compatibility lock has no concrete lerobot_version/lerobot_commit",
                    severity="warning",
                )
        else:
            result.add("COMPATIBILITY_LOCK_MISSING", "missing compatibility lock: {}".format(lock_path), severity="warning")

    if runtime_env_spec_path:
        _compare_runtime_schema(result, schema, os.path.abspath(runtime_env_spec_path))

    if lerobot_root_abs:
        info = _read_lerobot_info(lerobot_root_abs)
        if info is None:
            severity = "error" if require_lerobot_dataset else "warning"
            result.add(
                "LEROBOT_DATASET_MISSING",
                "no public LeRobotDataset meta/info.json found under {}".format(lerobot_root_abs),
                severity=severity,
            )
        else:
            result.metadata["lerobot_info"] = info
            _compare_feature_dicts(
                result,
                schema.to_lerobot_features(),
                info.get("features", {}),
                "lerobot",
                ignored_actual_keys=LEROBOT_MANAGED_FEATURE_KEYS,
            )
            if abs(float(info.get("fps", schema.fps)) - float(schema.fps)) > 1e-6:
                result.add("LEROBOT_FPS_MISMATCH", "LeRobot fps does not match schema fps")
            if not _has_lerobot_stats(lerobot_root_abs):
                result.add("LEROBOT_STATS_MISSING", "LeRobot stats/normalization metadata is missing")
    elif require_lerobot_dataset:
        result.add("LEROBOT_DATASET_ROOT_MISSING", "lerobot_dataset_root is required for training")

    return result
