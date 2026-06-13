"""Dataset statistics for local Phase 2 records."""

import os
from typing import Any, Dict

import numpy as np

from .writer import load_episode_arrays, load_episode_metadata


def compute_dataset_statistics(dataset_root: str) -> Dict[str, Any]:
    episodes_root = os.path.join(dataset_root, "episodes")
    stats: Dict[str, Any] = {
        "episode_count": 0,
        "frame_count": 0,
        "state": {},
        "action": {},
        "episodes": [],
    }
    if not os.path.exists(episodes_root):
        return stats
    states = []
    actions = []
    for episode_id in sorted(os.listdir(episodes_root)):
        episode_path = os.path.join(episodes_root, episode_id)
        if not os.path.isdir(episode_path):
            continue
        metadata = load_episode_metadata(episode_path)
        arrays = load_episode_arrays(episode_path)
        stats["episode_count"] += 1
        frame_count = int(arrays["action"].shape[0])
        stats["frame_count"] += frame_count
        stats["episodes"].append(
            {
                "episode_id": episode_id,
                "variation_id": metadata.get("variation_id"),
                "frame_count": frame_count,
            }
        )
        states.append(arrays["observation.state"])
        actions.append(arrays["action"])
    if states:
        state = np.concatenate(states, axis=0)
        stats["state"] = {
            "mean": state.mean(axis=0).astype(float).tolist(),
            "std": state.std(axis=0).astype(float).tolist(),
            "min": state.min(axis=0).astype(float).tolist(),
            "max": state.max(axis=0).astype(float).tolist(),
        }
    if actions:
        action = np.concatenate(actions, axis=0)
        stats["action"] = {
            "mean": action.mean(axis=0).astype(float).tolist(),
            "std": action.std(axis=0).astype(float).tolist(),
            "min": action.min(axis=0).astype(float).tolist(),
            "max": action.max(axis=0).astype(float).tolist(),
        }
    return stats
