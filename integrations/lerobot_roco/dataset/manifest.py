"""Dataset manifest and provenance helpers."""

import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python 3.7 fallback for completeness.
    import importlib_metadata  # type: ignore


def atomic_write_json(path: str, data: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = "{}.tmp".format(path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit(repo_root: Optional[str] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def git_dirty(repo_root: Optional[str] = None) -> Optional[bool]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return bool(proc.stdout.strip())


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def package_commit(name: str) -> Optional[str]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    package_path = spec.origin
    if os.path.basename(package_path) == "__init__.py":
        package_path = os.path.dirname(package_path)
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def build_provenance(repo_root: Optional[str] = None) -> Dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "roco_commit": git_commit(repo_root),
        "roco_dirty": git_dirty(repo_root),
        "numpy_version": package_version("numpy"),
        "lerobot_version": package_version("lerobot"),
        "lerobot_commit": package_commit("lerobot"),
        "gymnasium_version": package_version("gymnasium"),
        "torch_version": package_version("torch"),
        "datasets_version": package_version("datasets"),
    }


@dataclass
class DatasetManifest:
    schema_version: str
    schema_hash: str
    dataset_revision: str
    roco_commit: Optional[str]
    lerobot_version: Optional[str]
    lerobot_commit: Optional[str]
    bridge_protocol: str
    task: str
    skill: str
    action_representation: str
    fps: float
    episodes_attempted: int = 0
    episodes_committed: int = 0
    episodes_quarantined: int = 0
    split_manifest: str = "splits.json"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "dataset_revision": self.dataset_revision,
            "roco_commit": self.roco_commit,
            "lerobot_version": self.lerobot_version,
            "lerobot_commit": self.lerobot_commit,
            "bridge_protocol": self.bridge_protocol,
            "task": self.task,
            "skill": self.skill,
            "action_representation": self.action_representation,
            "fps": float(self.fps),
            "episodes_attempted": int(self.episodes_attempted),
            "episodes_committed": int(self.episodes_committed),
            "episodes_quarantined": int(self.episodes_quarantined),
            "split_manifest": self.split_manifest,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        return cls(
            schema_version=str(data["schema_version"]),
            schema_hash=str(data["schema_hash"]),
            dataset_revision=str(data["dataset_revision"]),
            roco_commit=data.get("roco_commit"),
            lerobot_version=data.get("lerobot_version"),
            lerobot_commit=data.get("lerobot_commit"),
            bridge_protocol=str(data["bridge_protocol"]),
            task=str(data["task"]),
            skill=str(data["skill"]),
            action_representation=str(data["action_representation"]),
            fps=float(data["fps"]),
            episodes_attempted=int(data.get("episodes_attempted", 0)),
            episodes_committed=int(data.get("episodes_committed", 0)),
            episodes_quarantined=int(data.get("episodes_quarantined", 0)),
            split_manifest=str(data.get("split_manifest", "splits.json")),
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def create(cls, schema: Any, repo_root: Optional[str] = None) -> "DatasetManifest":
        provenance = build_provenance(repo_root)
        return cls(
            schema_version=schema.schema_version,
            schema_hash=schema.schema_hash,
            dataset_revision=schema.schema_hash[:12],
            roco_commit=provenance.get("roco_commit"),
            lerobot_version=provenance.get("lerobot_version"),
            lerobot_commit=provenance.get("lerobot_commit"),
            bridge_protocol=schema.bridge_protocol_version,
            task=schema.task_id,
            skill=schema.skill_id,
            action_representation=schema.action_representation,
            fps=schema.fps,
            provenance=provenance,
        )


def write_compatibility_lock(path: str, schema: Any, repo_root: Optional[str] = None) -> Dict[str, Any]:
    provenance = build_provenance(repo_root)
    lock = {
        "schema_version": schema.schema_version,
        "schema_hash": schema.schema_hash,
        "bridge_protocol": schema.bridge_protocol_version,
        "action_representation": schema.action_representation,
        "python": provenance.get("python"),
        "pytorch": provenance.get("torch_version"),
        "cuda": None,
        "gymnasium": provenance.get("gymnasium_version"),
        "transformers": package_version("transformers"),
        "datasets": provenance.get("datasets_version"),
        "lerobot_version": provenance.get("lerobot_version"),
        "lerobot_commit": provenance.get("lerobot_commit"),
        "roco_commit": provenance.get("roco_commit"),
        "roco_dirty": provenance.get("roco_dirty"),
        "created_utc": provenance.get("created_utc"),
    }
    atomic_write_json(path, lock)
    return lock
