"""Immutable configuration for an Octo fine-tuning run."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Default Octo-small checkpoint on HuggingFace, using upstream Octo's hf:// URI.
DEFAULT_PRETRAINED = "hf://rail-berkeley/octo-small"
DEFAULT_OUTPUT_ROOT = "checkpoints/octo"


class OctoTrainingConfigError(ValueError):
    """Raised when an OctoTrainingConfig is invalid."""


@dataclass(frozen=True)
class OctoTrainingConfig:
    """All parameters needed for a single Octo fine-tuning run.

    These map directly onto the arguments consumed by Octo's
    ``scripts/finetune.py`` (or the equivalent Python API).
    """

    # --- Identity ---
    name: str                               # human-readable run name
    task_id: str = "sandwich"
    skill_id: str = "PICK"
    agent_name: str = "Chad"

    # --- Model ---
    pretrained_path: str = DEFAULT_PRETRAINED
    window_size: int = 2                    # Octo-small default

    # --- Dataset ---
    data_root: str = "data/subtask_demos"   # raw NPZ demo root
    rlds_dir: str = ""                      # converted RLDS dir; auto-set if empty
    val_fraction: float = 0.1
    action_dim: int = 14
    state_dim: int = 14
    language_instruction: str = ""          # overrides auto-generated instruction

    # --- Training ---
    output_dir: str = DEFAULT_OUTPUT_ROOT
    num_steps: int = 20_000
    batch_size: int = 128
    chunk_size: int = 10                    # prediction horizon
    save_interval: int = 2_000
    eval_interval: int = 2_000
    seed: int = 42
    lr: float = 3e-4
    frozen_keys: str = "(?!.*hf_token).*vit.*"  # freeze ViT backbone by default

    # --- Infra ---
    dry_run: bool = False
    overwrite: bool = False
    octo_repo_dir: str = ""                  # path to upstream octo-models/octo checkout
    finetune_script: str = ""                # explicit path to upstream scripts/finetune.py
    finetune_config_mode: str = "full,language_conditioned"
    debug: bool = True                       # pass --debug to upstream script (disables wandb)

    def __post_init__(self) -> None:
        if not self.name:
            raise OctoTrainingConfigError("name is required")
        if self.num_steps <= 0:
            raise OctoTrainingConfigError("num_steps must be positive")
        if self.batch_size <= 0:
            raise OctoTrainingConfigError("batch_size must be positive")
        if not 0.0 < self.val_fraction < 1.0:
            raise OctoTrainingConfigError("val_fraction must be in (0, 1)")

    @property
    def resolved_rlds_dir(self) -> str:
        if self.rlds_dir:
            return self.rlds_dir
        return os.path.join(self.data_root, self.task_id, self.skill_id.upper() + "_rlds")

    @property
    def resolved_data_root(self) -> str:
        return os.path.join(self.data_root, self.task_id, self.skill_id.upper())

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, self.name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: Any) -> "OctoTrainingConfig":
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Preset configs for sandwich task
# ---------------------------------------------------------------------------

def sandwich_pick_chad_debug() -> OctoTrainingConfig:
    """Quick smoke-test config: 100 steps on CPU."""
    return OctoTrainingConfig(
        name="octo_sandwich_pick_chad_debug",
        task_id="sandwich",
        skill_id="PICK",
        agent_name="Chad",
        num_steps=100,
        batch_size=2,
        save_interval=50,
        eval_interval=50,
        action_dim=14,
        state_dim=14,
        dry_run=False,
    )


def sandwich_pick_chad_baseline() -> OctoTrainingConfig:
    """Full fine-tune: 20k steps on GPU."""
    return OctoTrainingConfig(
        name="octo_sandwich_pick_chad",
        task_id="sandwich",
        skill_id="PICK",
        agent_name="Chad",
        num_steps=20_000,
        batch_size=128,
        save_interval=2_000,
        eval_interval=2_000,
        action_dim=14,
        state_dim=14,
    )


def sandwich_stack_on_dave_baseline() -> OctoTrainingConfig:
    return OctoTrainingConfig(
        name="octo_sandwich_stack_on_dave",
        task_id="sandwich",
        skill_id="STACK_ON",
        agent_name="Dave",
        num_steps=20_000,
        batch_size=128,
        save_interval=2_000,
        eval_interval=2_000,
        action_dim=14,
        state_dim=14,
    )


_PRESETS: Dict[str, Any] = {
    "sandwich_pick_chad_debug":    sandwich_pick_chad_debug,
    "sandwich_pick_chad":          sandwich_pick_chad_baseline,
    "sandwich_stack_on_dave":      sandwich_stack_on_dave_baseline,
}


# ---------------------------------------------------------------------------
# YAML / JSON loader
# ---------------------------------------------------------------------------

def _load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json") or yaml is None:
            return json.load(fh)
        data = yaml.safe_load(fh)
    return dict(data) if data else {}


def load_octo_training_config(path: str) -> OctoTrainingConfig:
    raw = _load_mapping(path)
    preset_name = str(raw.pop("preset", "") or "").strip()
    if preset_name:
        if preset_name not in _PRESETS:
            raise OctoTrainingConfigError(f"Unknown preset '{preset_name}'. "
                                          f"Choose from: {sorted(_PRESETS)}")
        base = _PRESETS[preset_name]()
    else:
        if "name" not in raw:
            raise OctoTrainingConfigError("YAML must include 'name' or 'preset'")
        base = OctoTrainingConfig(name=raw.pop("name"))

    allowed = set(OctoTrainingConfig.__dataclass_fields__.keys())
    overrides = {k: v for k, v in raw.items() if k in allowed}
    return base.with_overrides(**overrides)
