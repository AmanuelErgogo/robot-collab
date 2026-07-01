"""Evaluation configuration for Octo policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OctoEvalConfig:
    """Parameters for rolling out an Octo policy in a RoCo environment."""

    task_id: str
    skill_id: str
    agent_name: str
    checkpoint_dir: str

    num_episodes: int = 20
    max_steps: int = 300
    chunk_size: int = 10
    uncertainty_mode: str = "heuristic"   # "none" | "heuristic" | "policy_metadata"
    render: bool = False
    artifact_dir: str = "artifacts/octo_eval"
    seed: int = 0

    # Octo-specific
    unnorm_key: Optional[str] = None      # None → no un-normalisation
    image_size_h: int = 256
    image_size_w: int = 256
