"""LeRobot compatibility adapter for Phase 3 training.

All LeRobot-specific assumptions are kept here.  The module does not import
LeRobot at import time, so it remains safe to import from the RoCo runtime.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import warnings as py_warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .config import Phase3TrainingConfig

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python 3.7 fallback.
    import importlib_metadata  # type: ignore


class LeRobotCompatibilityError(RuntimeError):
    """Raised when the LeRobot training environment is not usable."""


@dataclass(frozen=True)
class LeRobotCompatibilityReport:
    lerobot_version: Optional[str]
    lerobot_commit: Optional[str]
    python: str
    executable: str
    lerobot_train: Optional[str]
    torch_version: Optional[str]
    cuda_available: Optional[bool]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lerobot_version": self.lerobot_version,
            "lerobot_commit": self.lerobot_commit,
            "python": self.python,
            "executable": self.executable,
            "lerobot_train": self.lerobot_train,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CommandResult:
    command: List[str]
    returncode: int
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": int(self.returncode),
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _package_commit(name: str) -> Optional[str]:
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


def inspect_lerobot_environment(lock: Optional[Mapping[str, Any]] = None) -> LeRobotCompatibilityReport:
    errors: List[str] = []
    warnings: List[str] = []
    lerobot_version = _package_version("lerobot")
    lerobot_commit = _package_commit("lerobot")
    lerobot_train = shutil.which("lerobot-train")

    if lerobot_version is None:
        errors.append("LeRobot package is not installed in this Python environment.")
    if lerobot_train is None:
        errors.append("lerobot-train entry point is not on PATH.")

    if lock:
        locked_version = lock.get("lerobot_version")
        locked_commit = lock.get("lerobot_commit")
        if locked_version and lerobot_version and str(locked_version) != str(lerobot_version):
            errors.append("Installed LeRobot version {} does not match lock {}.".format(lerobot_version, locked_version))
        if locked_commit and lerobot_commit and str(locked_commit) != str(lerobot_commit):
            errors.append("Installed LeRobot commit does not match compatibility lock.")
        if not locked_version or not locked_commit:
            warnings.append("Compatibility lock does not pin both LeRobot version and commit.")

    torch_version = _package_version("torch")
    cuda_available = None
    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore")
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
    except Exception:
        if torch_version is None:
            errors.append("Torch package is not installed.")
        else:
            warnings.append("Torch is installed but could not be imported.")

    return LeRobotCompatibilityReport(
        lerobot_version=lerobot_version,
        lerobot_commit=lerobot_commit,
        python=sys.version.split()[0],
        executable=sys.executable,
        lerobot_train=lerobot_train,
        torch_version=torch_version,
        cuda_available=cuda_available,
        errors=errors,
        warnings=warnings,
    )


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def build_lerobot_train_command(config: Phase3TrainingConfig, output_dir: Optional[str] = None) -> List[str]:
    """Build the pinned-public LeRobot train CLI command."""

    out = output_dir or config.output_dir
    command = [
        "lerobot-train",
        "--policy.type={}".format(config.policy_type),
        "--dataset.repo_id={}".format(config.dataset_repo_id),
        "--dataset.root={}".format(config.lerobot_dataset_root),
        "--output_dir={}".format(out),
        "--steps={}".format(config.steps),
        "--batch_size={}".format(config.batch_size),
        "--num_workers={}".format(config.num_workers),
        "--eval_freq={}".format(config.eval_freq),
        "--save_freq={}".format(config.save_freq),
        "--log_freq={}".format(config.log_freq),
        "--seed={}".format(config.seed),
        "--policy.device={}".format(config.device),
        "--policy.use_amp={}".format(_bool_arg(config.use_amp)),
        "--policy.push_to_hub=false",
        "--policy.n_obs_steps={}".format(config.n_obs_steps),
        "--policy.chunk_size={}".format(config.chunk_size),
        "--policy.n_action_steps={}".format(config.n_action_steps),
        "--dataset.use_imagenet_stats={}".format(_bool_arg(config.use_imagenet_stats)),
        "--dataset.image_transforms.enable={}".format(_bool_arg(config.image_transforms_enable)),
        "--wandb.enable={}".format(_bool_arg(config.wandb_enable)),
    ]
    if config.dataset_revision:
        command.append("--dataset.revision={}".format(config.dataset_revision))
    if config.episodes:
        command.append("--dataset.episodes={}".format(json.dumps(list(config.episodes), separators=(",", ":"))))
    if config.optimizer_lr is not None:
        command.append("--policy.optimizer_lr={}".format(config.optimizer_lr))
    command.extend(config.command_extra_args)
    return command


def run_lerobot_train(command: List[str], log_dir: str) -> CommandResult:
    """Run `lerobot-train` without shell expansion and capture logs."""

    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    return CommandResult(
        command=list(command),
        returncode=int(proc.returncode),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
