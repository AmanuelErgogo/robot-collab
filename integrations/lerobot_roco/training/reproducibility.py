"""Run manifest and provenance helpers for Phase 3."""

import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
from typing import Any, Dict, List, Mapping, Optional

from integrations.lerobot_roco.dataset.manifest import atomic_write_json, git_commit, git_dirty, package_version, read_json

from .config import Phase3TrainingConfig, config_hash


def sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _torch_provenance() -> Dict[str, Any]:
    data = {
        "torch_version": package_version("torch"),
        "cuda_available": None,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_names": [],
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch  # type: ignore

            data["cuda_available"] = bool(torch.cuda.is_available())
            data["cuda_version"] = getattr(torch.version, "cuda", None)
            data["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            if data["cuda_available"]:
                data["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return data


@dataclass
class TrainingRunManifest:
    run_id: str
    phase: str
    config: Mapping[str, Any]
    resolved_config_hash: str
    command: List[str]
    dataset: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    preflight: Mapping[str, Any]
    provenance: Mapping[str, Any]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    status: str = "prepared"
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "config": dict(self.config),
            "resolved_config_hash": self.resolved_config_hash,
            "command": list(self.command),
            "dataset": dict(self.dataset),
            "compatibility": dict(self.compatibility),
            "preflight": dict(self.preflight),
            "provenance": dict(self.provenance),
            "artifacts": dict(self.artifacts),
            "status": self.status,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
        }


def dataset_manifest_summary(dataset_root: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "dataset_root": os.path.abspath(dataset_root),
        "schema_hash": None,
        "schema_json_sha256": sha256_file(os.path.join(dataset_root, "schema.json")),
        "splits_json_sha256": sha256_file(os.path.join(dataset_root, "splits.json")),
        "dataset_manifest_json_sha256": sha256_file(os.path.join(dataset_root, "dataset_manifest.json")),
    }
    manifest_path = os.path.join(dataset_root, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        summary.update(
            {
                "dataset_revision": manifest.get("dataset_revision"),
                "schema_hash": manifest.get("schema_hash"),
                "bridge_protocol": manifest.get("bridge_protocol"),
                "action_representation": manifest.get("action_representation"),
                "fps": manifest.get("fps"),
                "episodes_committed": manifest.get("episodes_committed"),
            }
        )
    return summary


def build_run_manifest(
    config: Phase3TrainingConfig,
    command: List[str],
    preflight: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    repo_root: Optional[str] = None,
) -> TrainingRunManifest:
    provenance = {
        "created_utc": _utc_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "roco_commit": git_commit(repo_root),
        "roco_dirty": git_dirty(repo_root),
        "numpy_version": package_version("numpy"),
        "gymnasium_version": package_version("gymnasium"),
        "datasets_version": package_version("datasets"),
        "lerobot_version": package_version("lerobot"),
        "transformers_version": package_version("transformers"),
    }
    provenance.update(_torch_provenance())
    return TrainingRunManifest(
        run_id=config.name,
        phase="phase3_act_training",
        config=config.to_dict(),
        resolved_config_hash=config_hash(config),
        command=list(command),
        dataset=dataset_manifest_summary(config.dataset_root),
        compatibility=dict(compatibility),
        preflight=dict(preflight),
        provenance=provenance,
    )


def write_run_manifest(path: str, manifest: TrainingRunManifest) -> None:
    atomic_write_json(path, manifest.to_dict())
