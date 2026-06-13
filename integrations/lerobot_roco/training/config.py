"""Immutable Phase 3 ACT training configuration."""

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional in minimal client envs.
    yaml = None  # type: ignore


DEFAULT_DATASET_ROOT = "artifacts/datasets/pack_put_object_debug"
DEFAULT_LEROBOT_ROOT = "artifacts/datasets/pack_put_object_debug_lerobot"
DEFAULT_DATASET_REPO_ID = "local/roco-pack-put-object-debug"
DEFAULT_OUTPUT_ROOT = "artifacts/training/phase3"
DEFAULT_COMPATIBILITY_LOCK = "integrations/lerobot_roco/compatibility.lock.json"


class TrainingConfigError(ValueError):
    """Raised when a Phase 3 training config is malformed."""


def _tuple_str(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _tuple_int(values: Optional[Iterable[Any]]) -> Tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class Phase3TrainingConfig:
    """Resolved training-run contract for the ACT Phase 3 launcher."""

    name: str
    dataset_root: str = DEFAULT_DATASET_ROOT
    lerobot_dataset_root: str = DEFAULT_LEROBOT_ROOT
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID
    dataset_revision: Optional[str] = None
    output_root: str = DEFAULT_OUTPUT_ROOT
    compatibility_lock: str = DEFAULT_COMPATIBILITY_LOCK
    runtime_env_spec_path: Optional[str] = None
    split: str = "train"
    episodes: Tuple[int, ...] = ()
    policy_type: str = "act"
    task_id: str = "pack"
    skill_id: str = "PUT_OBJECT_IN_CONTAINER"
    active_agent: str = "Alice"
    action_representation: str = "absolute_joint_position_plus_gripper"
    steps: int = 50
    batch_size: int = 2
    num_workers: int = 0
    seed: int = 1000
    save_freq: int = 25
    log_freq: int = 10
    eval_freq: int = 0
    device: str = "cpu"
    use_amp: bool = False
    wandb_enable: bool = False
    use_imagenet_stats: bool = False
    image_transforms_enable: bool = False
    chunk_size: int = 20
    n_action_steps: int = 20
    n_obs_steps: int = 1
    optimizer_lr: Optional[float] = None
    require_lerobot_dataset: bool = True
    require_lerobot_package: bool = True
    dry_run: bool = False
    overwrite: bool = False
    command_extra_args: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", _tuple_int(self.episodes))
        object.__setattr__(self, "command_extra_args", _tuple_str(self.command_extra_args))
        if not self.name:
            raise TrainingConfigError("name is required")
        if self.policy_type != "act":
            raise TrainingConfigError("Phase 3 only supports ACT")
        for attr in ("steps", "batch_size", "save_freq", "log_freq", "chunk_size", "n_action_steps", "n_obs_steps"):
            if int(getattr(self, attr)) <= 0:
                raise TrainingConfigError("{} must be positive".format(attr))
        if int(self.num_workers) < 0:
            raise TrainingConfigError("num_workers must be non-negative")
        if int(self.eval_freq) < 0:
            raise TrainingConfigError("eval_freq must be non-negative")
        if self.n_action_steps > self.chunk_size:
            raise TrainingConfigError("n_action_steps cannot exceed chunk_size")
        if self.n_obs_steps != 1:
            raise TrainingConfigError("pinned ACT config only supports n_obs_steps=1")
        if self.optimizer_lr is not None and float(self.optimizer_lr) <= 0:
            raise TrainingConfigError("optimizer_lr must be positive when set")
        if self.split not in ("train", "validation", "test", "overfit"):
            raise TrainingConfigError("split must be train, validation, test, or overfit")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_root, self.name)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["episodes"] = list(self.episodes)
        data["command_extra_args"] = list(self.command_extra_args)
        return data

    def with_overrides(self, **kwargs: Any) -> "Phase3TrainingConfig":
        return replace(self, **kwargs)


def debug_training_config() -> Phase3TrainingConfig:
    return Phase3TrainingConfig(
        name="act_pack_put_debug",
        steps=50,
        batch_size=2,
        num_workers=0,
        save_freq=25,
        log_freq=10,
        eval_freq=0,
        device="cpu",
        image_transforms_enable=False,
        use_imagenet_stats=False,
        chunk_size=20,
        n_action_steps=20,
    )


def overfit_training_config() -> Phase3TrainingConfig:
    return debug_training_config().with_overrides(
        name="act_pack_put_overfit",
        split="overfit",
        episodes=(0,),
        steps=500,
        batch_size=1,
        save_freq=50,
        log_freq=10,
        seed=123,
        image_transforms_enable=False,
        use_imagenet_stats=False,
    )


def baseline_training_config() -> Phase3TrainingConfig:
    return debug_training_config().with_overrides(
        name="act_pack_put_baseline",
        steps=100000,
        batch_size=8,
        num_workers=4,
        save_freq=20000,
        log_freq=200,
        device="cuda",
        use_imagenet_stats=True,
        image_transforms_enable=True,
        chunk_size=100,
        n_action_steps=100,
    )


_DEFAULTS = {
    "debug": debug_training_config,
    "overfit": overfit_training_config,
    "baseline": baseline_training_config,
}


def _load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            return json.load(f)
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TrainingConfigError("config root must be a mapping")
    return dict(data)


def _flatten_known_sections(data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if key in ("dataset", "output", "policy", "runtime", "training", "provenance"):
            if not isinstance(value, Mapping):
                raise TrainingConfigError("{} section must be a mapping".format(key))
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _strip_unknown(data: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = set(Phase3TrainingConfig.__dataclass_fields__.keys())
    return {str(key): value for key, value in data.items() if str(key) in allowed}


def load_training_config(path: str) -> Phase3TrainingConfig:
    raw = _load_mapping(path)
    preset = str(raw.get("preset", "") or "").strip()
    if preset:
        if preset not in _DEFAULTS:
            raise TrainingConfigError("unknown preset: {}".format(preset))
        base = _DEFAULTS[preset]()
    else:
        base = debug_training_config()
    data = _strip_unknown(_flatten_known_sections(raw))
    data.pop("preset", None)
    return base.with_overrides(**data)


def write_resolved_config(path: str, config: Phase3TrainingConfig) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = "{}.tmp".format(path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def config_hash(config: Phase3TrainingConfig) -> str:
    import hashlib

    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
