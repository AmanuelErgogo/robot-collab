"""Demo collection pipeline: wraps collect_subtask_demos with task registry awareness."""

from __future__ import annotations

import importlib
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .registry import TASK_REGISTRY, DEFAULT_SKILLS_PER_TASK


@dataclass
class CollectConfig:
    task: str
    skills: List[str]
    num_episodes: int = 100
    output_dir: str = "data/subtask_demos"
    render: bool = False
    verbose: bool = False
    seed: int = 0


@dataclass
class CollectResult:
    task: str
    skills_collected: Dict[str, int]   # skill → successful episode count
    total_episodes_attempted: int
    output_dir: str


def collect(cfg: CollectConfig) -> CollectResult:
    """Run scripted rollouts and save (obs, action) demos per subtask."""
    if cfg.task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{cfg.task}'. Choices: {sorted(TASK_REGISTRY)}")

    np.random.seed(cfg.seed)

    # Import here to avoid hard dependency at module load time
    from scripts.collect_subtask_demos import (
        SubtaskDemoRecorder,
        run_scripted_episode,
        _load_task_env,
    )

    skills = cfg.skills or DEFAULT_SKILLS_PER_TASK[cfg.task]
    print(f"\n{'='*60}")
    print(f"  CRIE-Bench: collecting demos")
    print(f"  task      : {cfg.task}")
    print(f"  skills    : {skills}")
    print(f"  episodes  : {cfg.num_episodes}")
    print(f"  output    : {cfg.output_dir}")
    print(f"{'='*60}\n")

    env = _load_task_env(cfg.task, render=cfg.render)
    image_cameras = ["teaser"]  # capture teaser cam for VLA training
    recorder = SubtaskDemoRecorder(cfg.output_dir, cfg.task, skills,
                                   env=env, image_cameras=image_cameras,
                                   image_size=(128, 128))

    success_count = 0
    for ep in range(cfg.num_episodes):
        print(f"Episode {ep + 1}/{cfg.num_episodes}")
        try:
            ok = run_scripted_episode(env, recorder, verbose=cfg.verbose)
            if ok:
                success_count += 1
        except Exception as exc:
            print(f"  [ERROR] Episode {ep + 1}: {exc}")
            if cfg.verbose:
                traceback.print_exc()

    # Count saved episodes per skill
    skills_collected: Dict[str, int] = {}
    for skill in skills:
        skill_dir = Path(cfg.output_dir) / cfg.task / skill.upper()
        if skill_dir.exists():
            skills_collected[skill] = len([
                d for d in skill_dir.iterdir() if d.is_dir()
            ])
        else:
            skills_collected[skill] = 0

    print(f"\nDone. {success_count}/{cfg.num_episodes} successful episodes.")
    for skill, count in skills_collected.items():
        print(f"  {skill}: {count} episodes saved")

    return CollectResult(
        task=cfg.task,
        skills_collected=skills_collected,
        total_episodes_attempted=cfg.num_episodes,
        output_dir=str(Path(cfg.output_dir) / cfg.task),
    )


def count_demos(output_dir: str, task: str, skill: str) -> int:
    """Return the number of saved demo episodes for (task, skill)."""
    skill_dir = Path(output_dir) / task / skill.upper()
    if not skill_dir.exists():
        return 0
    return len([d for d in skill_dir.iterdir() if d.is_dir()])
