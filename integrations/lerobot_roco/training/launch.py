"""Phase 3 ACT launch orchestration."""

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from integrations.lerobot_roco.dataset.manifest import atomic_write_json, read_json

from .compatibility import (
    CommandResult,
    build_lerobot_train_command,
    inspect_lerobot_environment,
    run_lerobot_train,
)
from .config import Phase3TrainingConfig, load_training_config, write_resolved_config
from .feature_contract import run_feature_preflight
from .reproducibility import TrainingRunManifest, build_run_manifest, write_run_manifest


class TrainingLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchResult:
    run_dir: str
    manifest_path: str
    command: List[str]
    status: str
    command_result: Optional[CommandResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "command": list(self.command),
            "status": self.status,
            "command_result": self.command_result.to_dict() if self.command_result else None,
        }


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _prepare_run_dir(config: Phase3TrainingConfig) -> str:
    run_dir = os.path.abspath(config.output_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(manifest_path) and not config.overwrite:
        existing = read_json(manifest_path)
        status = existing.get("status")
        raise TrainingLaunchError(
            "run directory already has a manifest with status {!r}: {}".format(status, run_dir)
        )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def prepare_launch(config: Phase3TrainingConfig) -> TrainingRunManifest:
    """Run preflight checks and write the initial immutable run metadata."""

    run_dir = _prepare_run_dir(config)
    resolved_config_path = os.path.join(run_dir, "resolved_config.json")
    write_resolved_config(resolved_config_path, config)

    preflight = run_feature_preflight(
        dataset_root=config.dataset_root,
        lerobot_dataset_root=config.lerobot_dataset_root,
        compatibility_lock_path=config.compatibility_lock,
        runtime_env_spec_path=config.runtime_env_spec_path,
        expected_action_representation=config.action_representation,
        require_lerobot_dataset=config.require_lerobot_dataset,
    )
    lock = preflight.metadata.get("compatibility_lock")
    compatibility = inspect_lerobot_environment(lock if isinstance(lock, dict) else None)
    lerobot_output_dir = os.path.join(run_dir, "lerobot_output")
    if config.overwrite and os.path.exists(lerobot_output_dir):
        shutil.rmtree(lerobot_output_dir)
    command = build_lerobot_train_command(config, output_dir=lerobot_output_dir)
    manifest = build_run_manifest(
        config=config,
        command=command,
        preflight=preflight.to_dict(),
        compatibility=compatibility.to_dict(),
        repo_root=_repo_root(),
    )
    manifest.artifacts = {"resolved_config": resolved_config_path, "lerobot_output": lerobot_output_dir}
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    write_run_manifest(manifest_path, manifest)
    atomic_write_json(os.path.join(run_dir, "preflight.json"), preflight.to_dict())
    atomic_write_json(os.path.join(run_dir, "lerobot_compatibility.json"), compatibility.to_dict())
    return manifest


def launch_training(config: Phase3TrainingConfig) -> LaunchResult:
    run_dir = os.path.abspath(config.output_dir)
    manifest = prepare_launch(config)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    preflight_errors = list(manifest.preflight.get("issues", []))
    preflight_errors = [issue for issue in preflight_errors if issue.get("severity") == "error"]
    compat_errors = list(manifest.compatibility.get("errors", []))

    if config.dry_run:
        manifest.status = "dry_run"
        manifest.end_utc = manifest.provenance.get("created_utc")
        write_run_manifest(manifest_path, manifest)
        return LaunchResult(run_dir=run_dir, manifest_path=manifest_path, command=manifest.command, status="dry_run")

    if preflight_errors:
        manifest.status = "blocked_preflight"
        write_run_manifest(manifest_path, manifest)
        raise TrainingLaunchError("feature preflight failed; see {}".format(os.path.join(run_dir, "preflight.json")))
    if config.require_lerobot_package and compat_errors:
        manifest.status = "blocked_lerobot"
        write_run_manifest(manifest_path, manifest)
        raise TrainingLaunchError(
            "LeRobot environment is not usable; see {}".format(os.path.join(run_dir, "lerobot_compatibility.json"))
        )

    manifest.status = "running"
    from integrations.lerobot_roco.training.reproducibility import _utc_now

    manifest.start_utc = _utc_now()
    write_run_manifest(manifest_path, manifest)
    command_result = run_lerobot_train(manifest.command, os.path.join(run_dir, "logs"))
    manifest.end_utc = _utc_now()
    manifest.status = "completed" if command_result.returncode == 0 else "failed"
    manifest.artifacts = dict(manifest.artifacts)
    manifest.artifacts["logs"] = os.path.join(run_dir, "logs")
    manifest.artifacts["command_result"] = command_result.to_dict()
    write_run_manifest(manifest_path, manifest)
    if command_result.returncode != 0:
        raise TrainingLaunchError("lerobot-train failed with exit code {}".format(command_result.returncode))
    return LaunchResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        command=manifest.command,
        status=manifest.status,
        command_result=command_result,
    )


def launch_from_config_path(path: str) -> LaunchResult:
    return launch_training(load_training_config(path))
