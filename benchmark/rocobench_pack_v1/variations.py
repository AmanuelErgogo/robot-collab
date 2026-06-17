"""Immutable variation manifest helpers."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .version import VARIATION_MANIFEST_PATH, read_json


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_hash(path: str = VARIATION_MANIFEST_PATH) -> str:
    return sha256_json(read_json(path))


@dataclass(frozen=True)
class BenchmarkVariation:
    variation_id: str
    variation_group: str
    seed: int
    active_agents: Tuple[str, ...]
    object_assignments: Mapping[str, str]
    target_slots: Mapping[str, str]
    distractors: Tuple[str, ...]
    pose_set: str
    concurrency_case: str
    expected_skill_plan: Tuple[Mapping[str, str], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkVariation":
        reserved = {
            "variation_id",
            "variation_group",
            "seed",
            "active_agents",
            "object_assignments",
            "target_slots",
            "distractors",
            "pose_set",
            "concurrency_case",
            "expected_skill_plan",
        }
        metadata = {str(k): v for k, v in data.items() if k not in reserved}
        return cls(
            variation_id=str(data["variation_id"]),
            variation_group=str(data["variation_group"]),
            seed=int(data["seed"]),
            active_agents=tuple(str(x) for x in data["active_agents"]),
            object_assignments={str(k): str(v) for k, v in dict(data["object_assignments"]).items()},
            target_slots={str(k): str(v) for k, v in dict(data["target_slots"]).items()},
            distractors=tuple(str(x) for x in data.get("distractors", ())),
            pose_set=str(data["pose_set"]),
            concurrency_case=str(data["concurrency_case"]),
            expected_skill_plan=tuple({str(k): str(v) for k, v in dict(item).items()} for item in data["expected_skill_plan"]),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "variation_id": self.variation_id,
            "variation_group": self.variation_group,
            "seed": int(self.seed),
            "active_agents": list(self.active_agents),
            "object_assignments": dict(self.object_assignments),
            "target_slots": dict(self.target_slots),
            "distractors": list(self.distractors),
            "pose_set": self.pose_set,
            "concurrency_case": self.concurrency_case,
            "expected_skill_plan": [dict(item) for item in self.expected_skill_plan],
        }
        data.update(dict(self.metadata))
        return data

    @property
    def variation_hash(self) -> str:
        return sha256_json(self.to_dict())


def load_variation_manifest(path: str = VARIATION_MANIFEST_PATH) -> Dict[str, Any]:
    return read_json(path)


def load_variations(path: str = VARIATION_MANIFEST_PATH) -> Tuple[BenchmarkVariation, ...]:
    manifest = load_variation_manifest(path)
    variations = tuple(BenchmarkVariation.from_dict(item) for item in manifest.get("variations", ()))
    ids = [variation.variation_id for variation in variations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate variation_id in variation manifest")
    return variations


def variations_for_group(group: str, variations: Iterable[BenchmarkVariation] = None) -> Tuple[BenchmarkVariation, ...]:
    values = tuple(variations) if variations is not None else load_variations()
    return tuple(variation for variation in values if variation.variation_group == group)


def select_variations(group: str, episodes_per_task: int, variations: Sequence[BenchmarkVariation] = None) -> Tuple[BenchmarkVariation, ...]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    group_variations = variations_for_group(group, variations)
    if not group_variations:
        raise ValueError("no variations for group {}".format(group))
    selected = []
    for index in range(int(episodes_per_task)):
        selected.append(group_variations[index % len(group_variations)])
    return tuple(selected)
