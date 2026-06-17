"""Result schema for Phase 8 benchmark episodes."""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping


EPISODE_CSV_FIELDS = [
    "benchmark_id",
    "benchmark_version",
    "schema_version",
    "protocol_version",
    "predicate_version",
    "variation_manifest_hash",
    "task_id",
    "variation_id",
    "variation_hash",
    "track",
    "method",
    "episode_index",
    "seed",
    "success",
    "overall_success",
    "learned_success",
    "fallback_success",
    "termination_reason",
    "num_env_steps",
    "sim_time_s",
    "latency_p50_ms",
    "latency_p95_ms",
    "action_violations",
    "drops",
    "collisions",
    "final_error",
    "plan_valid",
    "replans",
    "fallback_attempts",
    "planner_latency_ms",
    "token_count",
    "makespan_steps",
    "parallel_speedup",
    "resource_violations",
    "central_stops",
    "artifact_dir",
    "extra_json",
]


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class BenchmarkEpisodeResult:
    benchmark_id: str
    benchmark_version: str
    schema_version: str
    protocol_version: str
    predicate_version: str
    variation_manifest_hash: str
    task_id: str
    variation_id: str
    variation_hash: str
    track: str
    method: str
    episode_index: int
    seed: int
    success: bool
    overall_success: bool
    learned_success: bool
    fallback_success: bool
    termination_reason: str
    num_env_steps: int = 0
    sim_time_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    action_violations: int = 0
    drops: int = 0
    collisions: int = 0
    final_error: float = 0.0
    plan_valid: bool = False
    replans: int = 0
    fallback_attempts: int = 0
    planner_latency_ms: float = 0.0
    token_count: int = 0
    makespan_steps: int = 0
    parallel_speedup: float = 0.0
    resource_violations: int = 0
    central_stops: int = 0
    artifact_dir: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "predicate_version": self.predicate_version,
            "variation_manifest_hash": self.variation_manifest_hash,
            "task_id": self.task_id,
            "variation_id": self.variation_id,
            "variation_hash": self.variation_hash,
            "track": self.track,
            "method": self.method,
            "episode_index": int(self.episode_index),
            "seed": int(self.seed),
            "success": bool(self.success),
            "overall_success": bool(self.overall_success),
            "learned_success": bool(self.learned_success),
            "fallback_success": bool(self.fallback_success),
            "termination_reason": self.termination_reason,
            "num_env_steps": int(self.num_env_steps),
            "sim_time_s": float(self.sim_time_s),
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "action_violations": int(self.action_violations),
            "drops": int(self.drops),
            "collisions": int(self.collisions),
            "final_error": float(self.final_error),
            "plan_valid": bool(self.plan_valid),
            "replans": int(self.replans),
            "fallback_attempts": int(self.fallback_attempts),
            "planner_latency_ms": float(self.planner_latency_ms),
            "token_count": int(self.token_count),
            "makespan_steps": int(self.makespan_steps),
            "parallel_speedup": float(self.parallel_speedup),
            "resource_violations": int(self.resource_violations),
            "central_stops": int(self.central_stops),
            "artifact_dir": self.artifact_dir,
            "extra": _json_safe(dict(self.extra)),
        }

    def to_csv_row(self) -> Dict[str, Any]:
        data = self.to_dict()
        data["extra_json"] = json.dumps(data.pop("extra"), sort_keys=True, separators=(",", ":"))
        return {field_name: data.get(field_name, "") for field_name in EPISODE_CSV_FIELDS}


def write_episode_json(path: str, result: BenchmarkEpisodeResult) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)


def write_episodes_csv(path: str, results: Iterable[BenchmarkEpisodeResult]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def read_episodes_csv(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            key = "{}::{}".format(row.get("task_id", ""), row.get("variation_id", ""))
            rows[key] = dict(row)
    return rows

