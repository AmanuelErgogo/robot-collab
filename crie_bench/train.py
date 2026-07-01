"""Training pipeline dispatcher — routes to ACT (LeRobot) or Octo backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SUPPORTED_MODELS = ("act", "octo")


@dataclass
class TrainConfig:
    task: str
    skill: str
    agent: str
    model: str = "octo"                        # "act" | "octo"
    data_dir: str = "data/subtask_demos"
    output_dir: str = "checkpoints"
    pretrained_path: str = ""                  # HF hub path or local dir
    num_steps: int = 20_000
    batch_size: int = 128
    chunk_size: int = 10
    save_interval: int = 2_000
    eval_interval: int = 2_000
    device: str = "cuda"
    seed: int = 42
    dry_run: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    model: str
    task: str
    skill: str
    agent: str
    checkpoint_dir: str
    status: str                 # "completed" | "dry_run" | "failed"
    num_steps: int = 0
    manifest_path: str = ""


def train(cfg: TrainConfig) -> TrainResult:
    """Dispatch training to the requested backend."""
    if cfg.model not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {SUPPORTED_MODELS}, got '{cfg.model}'")

    if cfg.model == "act":
        return _train_act(cfg)
    return _train_octo(cfg)


# ---------------------------------------------------------------------------
# ACT backend (existing LeRobot integration)
# ---------------------------------------------------------------------------

def _train_act(cfg: TrainConfig) -> TrainResult:
    from integrations.lerobot_roco.training.config import Phase3TrainingConfig
    from integrations.lerobot_roco.training.launch import launch_training

    run_name = f"act_{cfg.task}_{cfg.skill.lower()}_{cfg.agent.lower()}"
    dataset_root = os.path.join(cfg.data_dir, cfg.task, cfg.skill.upper())
    lerobot_root = dataset_root + "_lerobot"
    output_root = cfg.output_dir

    act_cfg = Phase3TrainingConfig(
        name=run_name,
        dataset_root=dataset_root,
        lerobot_dataset_root=lerobot_root,
        dataset_repo_id=f"local/roco-{cfg.task}-{cfg.skill.lower()}",
        output_root=output_root,
        task_id=cfg.task,
        skill_id=cfg.skill,
        active_agent=cfg.agent,
        steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.chunk_size,
        save_freq=cfg.save_interval,
        log_freq=max(1, cfg.save_interval // 10),
        device=cfg.device,
        seed=cfg.seed,
        dry_run=cfg.dry_run,
        overwrite=True,
    )

    result = launch_training(act_cfg)
    return TrainResult(
        model="act",
        task=cfg.task,
        skill=cfg.skill,
        agent=cfg.agent,
        checkpoint_dir=os.path.join(result.run_dir, "lerobot_output"),
        status=result.status,
        manifest_path=result.manifest_path,
    )


# ---------------------------------------------------------------------------
# Octo backend
# ---------------------------------------------------------------------------

def _train_octo(cfg: TrainConfig) -> TrainResult:
    from integrations.octo_roco.training.config import OctoTrainingConfig
    from integrations.octo_roco.training.launch import launch_octo_training

    pretrained = cfg.pretrained_path or "hf://rail-berkeley/octo-small"
    run_name = f"octo_{cfg.task}_{cfg.skill.lower()}_{cfg.agent.lower()}"
    checkpoint_dir = os.path.join(cfg.output_dir, run_name)

    octo_cfg = OctoTrainingConfig(
        name=run_name,
        task_id=cfg.task,
        skill_id=cfg.skill,
        agent_name=cfg.agent,
        pretrained_path=pretrained,
        data_root=cfg.data_dir,    # OctoTrainingConfig.resolved_data_root appends task/skill
        output_dir=cfg.output_dir,
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        chunk_size=cfg.chunk_size,
        save_interval=cfg.save_interval,
        eval_interval=cfg.eval_interval,
        seed=cfg.seed,
        dry_run=cfg.dry_run,
    )

    result = launch_octo_training(octo_cfg)
    return TrainResult(
        model="octo",
        task=cfg.task,
        skill=cfg.skill,
        agent=cfg.agent,
        checkpoint_dir=result.checkpoint_dir,
        status=result.status,
        num_steps=result.num_steps_completed,
        manifest_path=result.manifest_path,
    )
