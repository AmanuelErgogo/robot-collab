"""Variation-group dataset splits."""

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .manifest import atomic_write_json, read_json


DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class SplitManifest:
    splits: Mapping[str, Tuple[str, ...]]
    variation_ids: Mapping[str, Tuple[str, ...]]
    seed: int
    ratios: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "ratios": {key: float(value) for key, value in self.ratios.items()},
            "splits": {key: list(value) for key, value in sorted(self.splits.items())},
            "variation_ids": {key: list(value) for key, value in sorted(self.variation_ids.items())},
        }


def _episode_metadata(dataset_root: str) -> List[Dict[str, Any]]:
    rows = []
    episodes_root = os.path.join(dataset_root, "episodes")
    if not os.path.exists(episodes_root):
        return rows
    for episode_id in sorted(os.listdir(episodes_root)):
        path = os.path.join(episodes_root, episode_id, "metadata.json")
        if os.path.exists(path):
            row = read_json(path)
            row["episode_id"] = episode_id
            rows.append(row)
    return rows


def _stable_order_key(variation_id: str, seed: int) -> str:
    payload = "{}:{}".format(int(seed), variation_id).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    total = sum(float(x) for x in ratios.values())
    if total <= 0:
        raise ValueError("split ratios must have positive total")
    return {str(key): float(value) / total for key, value in ratios.items()}


def _split_groups(groups: Sequence[str], ratios: Mapping[str, float]) -> Dict[str, List[str]]:
    normalized = _normalize_ratios(ratios)
    names = list(normalized.keys())
    count = len(groups)
    result: Dict[str, List[str]] = {name: [] for name in names}
    if count == 0:
        return result
    cumulative = 0
    for index, name in enumerate(names):
        if index == len(names) - 1:
            end = count
        else:
            cumulative += normalized[name]
            end = int(round(cumulative * count))
        start = sum(len(result[prev]) for prev in names[:index])
        result[name] = list(groups[start:end])
    return result


def create_variation_group_splits(
    dataset_root: str,
    output_path: Optional[str] = None,
    ratios: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> SplitManifest:
    ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    rows = _episode_metadata(dataset_root)
    by_variation: Dict[str, List[str]] = {}
    for row in rows:
        variation_id = str(row["variation_id"])
        by_variation.setdefault(variation_id, []).append(str(row["episode_id"]))
    ordered_variations = sorted(by_variation.keys(), key=lambda item: _stable_order_key(item, seed))
    variation_splits = _split_groups(ordered_variations, ratios)
    episode_splits: Dict[str, Tuple[str, ...]] = {}
    variation_ids: Dict[str, Tuple[str, ...]] = {}
    for split_name, split_variations in variation_splits.items():
        episodes: List[str] = []
        for variation_id in split_variations:
            episodes.extend(sorted(by_variation[variation_id]))
        episode_splits[split_name] = tuple(sorted(episodes))
        variation_ids[split_name] = tuple(split_variations)
    manifest = SplitManifest(
        splits=episode_splits,
        variation_ids=variation_ids,
        seed=int(seed),
        ratios=ratios,
    )
    output_path = output_path or os.path.join(dataset_root, "splits.json")
    atomic_write_json(output_path, manifest.to_dict())
    return manifest


def detect_split_leakage(split_data: Mapping[str, Any]) -> List[str]:
    seen: Dict[str, str] = {}
    leaks: List[str] = []
    variation_ids = split_data.get("variation_ids", {})
    for split_name, values in variation_ids.items():
        for variation_id in values:
            variation_id = str(variation_id)
            if variation_id in seen and seen[variation_id] != split_name:
                leaks.append(variation_id)
            seen[variation_id] = split_name
    return sorted(set(leaks))
