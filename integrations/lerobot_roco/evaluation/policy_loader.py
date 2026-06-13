"""Version-isolated LeRobot policy loading for Phase 4."""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.lerobot_roco.training.checkpoint import find_pretrained_model_dir, inspect_checkpoint


class PolicyLoadError(RuntimeError):
    """Raised when the pinned LeRobot policy cannot be loaded."""


class PolicyInferenceError(RuntimeError):
    """Raised when policy inference violates the Phase 4 contract."""


@dataclass(frozen=True)
class ActionValidation:
    finite: bool
    shape_ok: bool
    below_low: int
    above_high: int
    min_value: float
    max_value: float

    @property
    def violation_count(self) -> int:
        return int(self.below_low) + int(self.above_high)

    @property
    def ok(self) -> bool:
        return bool(self.finite and self.shape_ok and self.violation_count == 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finite": bool(self.finite),
            "shape_ok": bool(self.shape_ok),
            "below_low": int(self.below_low),
            "above_high": int(self.above_high),
            "violation_count": int(self.violation_count),
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
        }


@dataclass(frozen=True)
class PolicyInferenceTrace:
    latency_ms: float
    raw_chunk_shape: Sequence[int]
    native_chunk_shape: Sequence[int]
    raw_chunk_dtype: str
    native_chunk_dtype: str
    device: str
    validation: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": float(self.latency_ms),
            "raw_chunk_shape": list(self.raw_chunk_shape),
            "native_chunk_shape": list(self.native_chunk_shape),
            "raw_chunk_dtype": self.raw_chunk_dtype,
            "native_chunk_dtype": self.native_chunk_dtype,
            "device": self.device,
            "validation": dict(self.validation),
        }


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _feature_to_dict(feature: Any) -> Dict[str, Any]:
    if isinstance(feature, Mapping):
        data = dict(feature)
    else:
        data = {
            "type": str(getattr(feature, "type", "")),
            "shape": list(getattr(feature, "shape", ())),
        }
    if "shape" in data:
        data["shape"] = [int(x) for x in data["shape"]]
    if "type" in data:
        data["type"] = str(data["type"]).split(".")[-1]
    return data


def _features_to_dict(features: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(key): _feature_to_dict(value) for key, value in sorted((features or {}).items())}


def validate_native_action_chunk(
    chunk: Any,
    action_low: Sequence[float],
    action_high: Sequence[float],
    expected_action_dim: int,
    tolerance: float = 0.0,
) -> ActionValidation:
    arr = np.asarray(chunk, dtype=np.float32)
    finite = bool(np.all(np.isfinite(arr)))
    shape_ok = bool(arr.ndim == 2 and arr.shape[-1] == int(expected_action_dim))
    if arr.size:
        min_value = float(np.nanmin(arr))
        max_value = float(np.nanmax(arr))
    else:
        min_value = 0.0
        max_value = 0.0
    if not shape_ok or not finite:
        return ActionValidation(finite, shape_ok, 0, 0, min_value, max_value)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    below = int(np.sum(arr < (low - float(tolerance))))
    above = int(np.sum(arr > (high + float(tolerance))))
    return ActionValidation(finite, shape_ok, below, above, min_value, max_value)


@dataclass
class LoadedPolicy:
    checkpoint_dir: str
    pretrained_model_dir: str
    policy: Any
    preprocessor: Any
    postprocessor: Any
    policy_config: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_size(self) -> int:
        return int(getattr(self.policy_config, "chunk_size"))

    @property
    def action_dim(self) -> int:
        output = getattr(self.policy_config, "output_features", {}) or {}
        feature = output.get("action")
        if feature is None:
            raise PolicyInferenceError("policy config has no action output feature")
        shape = tuple(int(x) for x in getattr(feature, "shape", ()))
        if len(shape) != 1:
            raise PolicyInferenceError("action output feature must be rank 1, got {}".format(shape))
        return int(shape[0])

    @property
    def input_features(self) -> Mapping[str, Any]:
        return getattr(self.policy_config, "input_features", {}) or {}

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def predict_native_chunk(
        self,
        policy_inputs: Mapping[str, Any],
        action_low: Sequence[float],
        action_high: Sequence[float],
        tolerance: float = 0.0,
    ) -> Tuple[np.ndarray, PolicyInferenceTrace]:
        start = time.perf_counter()
        try:
            batch = self.preprocessor(dict(policy_inputs))
            raw_chunk = self.policy.predict_action_chunk(batch)
            native_chunk = self.postprocessor(raw_chunk)
        except Exception as exc:
            raise PolicyInferenceError("policy inference failed: {}".format(exc)) from exc
        latency = (time.perf_counter() - start) * 1000.0

        raw_shape = tuple(int(x) for x in getattr(raw_chunk, "shape", ()))
        native_shape = tuple(int(x) for x in getattr(native_chunk, "shape", ()))
        raw_dtype = str(getattr(raw_chunk, "dtype", ""))
        native_dtype = str(getattr(native_chunk, "dtype", ""))
        device = str(getattr(native_chunk, "device", getattr(raw_chunk, "device", "")))

        if hasattr(native_chunk, "detach"):
            native_np = native_chunk.detach().cpu().numpy()
        else:
            native_np = np.asarray(native_chunk)
        native_np = np.asarray(native_np, dtype=np.float32)
        if native_np.ndim == 3:
            if native_np.shape[0] != 1:
                raise PolicyInferenceError("expected batch size 1 native chunk, got {}".format(native_np.shape))
            native_np = native_np[0]
        if native_np.ndim != 2:
            raise PolicyInferenceError("native action chunk must be [chunk, action_dim], got {}".format(native_np.shape))
        validation = validate_native_action_chunk(
            native_np,
            action_low=action_low,
            action_high=action_high,
            expected_action_dim=self.action_dim,
            tolerance=tolerance,
        )
        trace = PolicyInferenceTrace(
            latency_ms=latency,
            raw_chunk_shape=raw_shape,
            native_chunk_shape=native_shape,
            raw_chunk_dtype=raw_dtype,
            native_chunk_dtype=native_dtype,
            device=device,
            validation=validation.to_dict(),
        )
        return np.ascontiguousarray(native_np, dtype=np.float32), trace


def load_lerobot_policy(config: Any) -> LoadedPolicy:
    """Load the pinned LeRobot ACT checkpoint inside the client runtime."""

    checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    model_dir = find_pretrained_model_dir(checkpoint_dir)
    if model_dir is None:
        raise PolicyLoadError("could not locate pretrained_model under {}".format(checkpoint_dir))
    inspection = inspect_checkpoint(model_dir, dataset_root=os.path.abspath(config.dataset_root))
    if not inspection.ok:
        raise PolicyLoadError("checkpoint inspection failed: {}".format(inspection.issues))

    try:
        import importlib.metadata
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    except Exception as exc:
        raise PolicyLoadError("LeRobot is required in the Phase 4 client runtime: {}".format(exc)) from exc

    try:
        policy_cfg = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
        if getattr(config, "device", None):
            policy_cfg.device = str(config.device)
        ds_meta = LeRobotDatasetMetadata(
            config.dataset_repo_id,
            root=os.path.abspath(config.lerobot_dataset_root),
            revision=config.dataset_revision,
        )
        policy = make_policy(policy_cfg, ds_meta=ds_meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=model_dir,
            dataset_stats=getattr(ds_meta, "stats", None),
        )
    except Exception as exc:
        raise PolicyLoadError("failed to load LeRobot policy/processors: {}".format(exc)) from exc

    metadata = {
        "lerobot_version": importlib.metadata.version("lerobot"),
        "checkpoint_dir": checkpoint_dir,
        "pretrained_model_dir": model_dir,
        "policy_type": str(getattr(policy_cfg, "type", "")),
        "chunk_size": int(getattr(policy_cfg, "chunk_size", 0)),
        "n_action_steps": int(getattr(policy_cfg, "n_action_steps", 0)),
        "temporal_ensemble_coeff": getattr(policy_cfg, "temporal_ensemble_coeff", None),
        "normalization_mapping": {
            str(key): str(value) for key, value in dict(getattr(policy_cfg, "normalization_mapping", {}) or {}).items()
        },
        "input_features": _features_to_dict(getattr(policy_cfg, "input_features", {})),
        "output_features": _features_to_dict(getattr(policy_cfg, "output_features", {})),
        "processor_files": {
            "policy_preprocessor.json": _sha256_file(os.path.join(model_dir, "policy_preprocessor.json")),
            "policy_postprocessor.json": _sha256_file(os.path.join(model_dir, "policy_postprocessor.json")),
            "model.safetensors": _sha256_file(os.path.join(model_dir, "model.safetensors")),
            "config.json": _sha256_file(os.path.join(model_dir, "config.json")),
        },
        "checkpoint_inspection": inspection.to_dict(),
        "postprocessor_contract": "policy_postprocessor unnormalizes action to native simulator units before env.step",
    }
    if metadata["policy_type"] != "act":
        raise PolicyLoadError("Phase 4 direct rollout supports ACT only, got {}".format(metadata["policy_type"]))
    if metadata["temporal_ensemble_coeff"] is not None:
        raise PolicyLoadError("temporal ensembling must be explicitly implemented in Phase 4 before use")
    return LoadedPolicy(
        checkpoint_dir=checkpoint_dir,
        pretrained_model_dir=model_dir,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_config=policy_cfg,
        metadata=metadata,
    )
