"""Configuration for Phase 4 direct closed-loop policy evaluation."""

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - yaml is optional in minimal test envs.
    yaml = None  # type: ignore


DEFAULT_CHECKPOINT_DIR = (
    "artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050/pretrained_model"
)
DEFAULT_DATASET_ROOT = "artifacts/datasets/pack_put_object_debug"
DEFAULT_LEROBOT_ROOT = "artifacts/datasets/pack_put_object_debug_lerobot"
DEFAULT_DATASET_REPO_ID = "local/roco-pack-put-object-debug"
DEFAULT_OUTPUT_ROOT = "artifacts/evaluation/phase4"
DEFAULT_COMPATIBILITY_LOCK = "integrations/lerobot_roco/compatibility.lock.json"


class EvaluationConfigError(ValueError):
    """Raised when a Phase 4 evaluation config is malformed."""


def _tuple_int(values: Optional[Iterable[Any]]) -> Tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _tuple_str(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class Phase4EvaluationConfig:
    """Resolved contract for a direct ACT rollout run."""

    name: str
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    dataset_root: str = DEFAULT_DATASET_ROOT
    lerobot_dataset_root: str = DEFAULT_LEROBOT_ROOT
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID
    dataset_revision: Optional[str] = None
    compatibility_lock: str = DEFAULT_COMPATIBILITY_LOCK
    output_root: str = DEFAULT_OUTPUT_ROOT
    endpoint: str = "tcp://127.0.0.1:5557"
    active_agent: str = "Alice"
    split: str = "debug"
    frozen_suite: bool = False
    seeds: Tuple[int, ...] = (1000,)
    episode_ids: Tuple[str, ...] = ()
    max_steps: int = 80
    execution_horizon: Optional[int] = None
    action_bound_tolerance: float = 1e-5
    no_progress_window: int = 20
    no_progress_patience: int = 3
    no_progress_state_epsilon: float = 1e-4
    no_progress_action_epsilon: float = 1e-4
    collision_limit: Optional[int] = None
    record_video: bool = True
    render_every_steps: int = 1
    request_timeout_ms: int = 30000
    device: Optional[str] = None
    task_instruction: Optional[str] = None
    require_lerobot: bool = True
    overwrite: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _tuple_int(self.seeds))
        object.__setattr__(self, "episode_ids", _tuple_str(self.episode_ids))
        object.__setattr__(self, "notes", _tuple_str(self.notes))
        if not self.name:
            raise EvaluationConfigError("name is required")
        if self.active_agent not in ("Alice", "Bob"):
            raise EvaluationConfigError("active_agent must be Alice or Bob")
        if self.split not in ("debug", "train", "validation", "test"):
            raise EvaluationConfigError("split must be debug, train, validation, or test")
        if self.max_steps <= 0:
            raise EvaluationConfigError("max_steps must be positive")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise EvaluationConfigError("execution_horizon must be positive when set")
        if self.action_bound_tolerance < 0:
            raise EvaluationConfigError("action_bound_tolerance cannot be negative")
        if self.no_progress_window <= 0 or self.no_progress_patience <= 0:
            raise EvaluationConfigError("no-progress settings must be positive")
        if self.render_every_steps <= 0:
            raise EvaluationConfigError("render_every_steps must be positive")
        if not self.seeds:
            raise EvaluationConfigError("at least one seed is required")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_root, self.name)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["episode_ids"] = list(self.episode_ids)
        data["notes"] = list(self.notes)
        return data

    def with_overrides(self, **kwargs: Any) -> "Phase4EvaluationConfig":
        return replace(self, **kwargs)


def debug_evaluation_config() -> Phase4EvaluationConfig:
    return Phase4EvaluationConfig(
        name="act_pack_put_debug",
        split="debug",
        frozen_suite=False,
        seeds=(1000,),
        max_steps=80,
        execution_horizon=5,
        record_video=True,
    )


def validation_evaluation_config() -> Phase4EvaluationConfig:
    return debug_evaluation_config().with_overrides(
        name="act_pack_put_validation",
        split="validation",
        frozen_suite=False,
        seeds=(2000, 2001, 2002),
        execution_horizon=5,
    )


def test_evaluation_config() -> Phase4EvaluationConfig:
    return debug_evaluation_config().with_overrides(
        name="act_pack_put_test",
        split="test",
        frozen_suite=True,
        seeds=(3000, 3001, 3002),
        execution_horizon=5,
    )


_DEFAULTS = {
    "debug": debug_evaluation_config,
    "validation": validation_evaluation_config,
    "test": test_evaluation_config,
}


def _load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise EvaluationConfigError("config root must be a mapping")
    return dict(data)


def _flatten_known_sections(data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if key in ("policy", "dataset", "runtime", "suite", "monitors", "artifacts", "output"):
            if not isinstance(value, Mapping):
                raise EvaluationConfigError("{} section must be a mapping".format(key))
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _strip_unknown(data: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = set(Phase4EvaluationConfig.__dataclass_fields__.keys())
    return {str(key): value for key, value in data.items() if str(key) in allowed}


def load_evaluation_config(path: str) -> Phase4EvaluationConfig:
    raw = _load_mapping(path)
    preset = str(raw.get("preset", "") or "").strip()
    if preset:
        if preset not in _DEFAULTS:
            raise EvaluationConfigError("unknown preset: {}".format(preset))
        base = _DEFAULTS[preset]()
    else:
        base = debug_evaluation_config()
    data = _strip_unknown(_flatten_known_sections(raw))
    data.pop("preset", None)
    return base.with_overrides(**data)


def config_hash(config: Phase4EvaluationConfig) -> str:
    import hashlib

    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

