"""Evaluation suite planning and split discipline for Phase 4."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from integrations.lerobot_roco.dataset.manifest import read_json


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    seed: int
    split: str
    episode_id: Optional[str] = None
    variation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "seed": int(self.seed),
            "split": self.split,
            "episode_id": self.episode_id,
            "variation_id": self.variation_id,
        }


def _split_episode_ids(dataset_root: str, split: str) -> List[str]:
    path = os.path.join(dataset_root, "splits.json")
    if not os.path.exists(path):
        return []
    data = read_json(path)
    value = data.get(split)
    if isinstance(value, dict):
        for key in ("episode_ids", "episodes"):
            if key in value:
                return [str(x) for x in value[key]]
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _variation_id_for_episode(dataset_root: str, episode_id: str) -> Optional[str]:
    path = os.path.join(dataset_root, "episodes", episode_id, "variation.json")
    if not os.path.exists(path):
        return None
    data = read_json(path)
    for key in ("variation_id", "id", "hash"):
        if key in data:
            return str(data[key])
    return None


def build_episode_suite(config: Any) -> List[EpisodePlan]:
    split = "train" if config.split == "debug" else str(config.split)
    explicit_episode_ids = list(getattr(config, "episode_ids", ()) or ())
    episode_ids = explicit_episode_ids or _split_episode_ids(config.dataset_root, split)
    plans: List[EpisodePlan] = []
    seeds = list(config.seeds)
    count = max(len(seeds), len(episode_ids), 1)
    for index in range(count):
        seed = int(seeds[index % len(seeds)])
        episode_id = episode_ids[index % len(episode_ids)] if episode_ids else None
        plans.append(
            EpisodePlan(
                episode_index=index,
                seed=seed,
                split=str(config.split),
                episode_id=episode_id,
                variation_id=_variation_id_for_episode(config.dataset_root, episode_id) if episode_id else None,
            )
        )
    return plans

