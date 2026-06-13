"""Metrics helpers for Phase 4 policy rollouts."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    n = float(total)
    phat = float(successes) / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def latency_summary(values_ms: Sequence[float]) -> Dict[str, float]:
    if not values_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


@dataclass
class AggregateMetrics:
    episode_count: int
    successes: int
    success_rate: float
    success_wilson_95: Mapping[str, float]
    termination_distribution: Mapping[str, int]
    latency_ms: Mapping[str, float]
    mean_env_steps: float
    action_bound_violations: int
    no_progress_events: int
    drop_slip_rate: float = 0.0
    final_object_target_error: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_count": int(self.episode_count),
            "successes": int(self.successes),
            "success_rate": float(self.success_rate),
            "success_wilson_95": dict(self.success_wilson_95),
            "termination_distribution": dict(self.termination_distribution),
            "latency_ms": dict(self.latency_ms),
            "mean_env_steps": float(self.mean_env_steps),
            "action_bound_violations": int(self.action_bound_violations),
            "no_progress_events": int(self.no_progress_events),
            "drop_slip_rate": float(self.drop_slip_rate),
            "final_object_target_error": float(self.final_object_target_error),
            "notes": list(self.notes),
        }


def aggregate_rollout_results(results: Iterable[Any]) -> AggregateMetrics:
    results = list(results)
    total = len(results)
    successes = sum(1 for result in results if bool(getattr(result, "success", False)))
    terminations: Dict[str, int] = {}
    latencies: List[float] = []
    steps: List[int] = []
    bound_violations = 0
    no_progress = 0
    object_errors: List[float] = []
    slips = 0
    for result in results:
        reason = str(getattr(result, "termination_reason", "UNKNOWN"))
        terminations[reason] = terminations.get(reason, 0) + 1
        latencies.extend(float(x) for x in getattr(result, "inference_latency_ms", ()))
        steps.append(int(getattr(result, "num_env_steps", 0)))
        bound_violations += int(getattr(result, "action_bound_violations", 0))
        no_progress += int(getattr(result, "no_progress_events", 0))
        info = dict(getattr(result, "final_info", {}) or {})
        if "final_object_target_error" in info:
            object_errors.append(float(info["final_object_target_error"]))
        if info.get("object_lost") or info.get("slip_detected"):
            slips += 1
    return AggregateMetrics(
        episode_count=total,
        successes=successes,
        success_rate=(float(successes) / float(total)) if total else 0.0,
        success_wilson_95=wilson_interval(successes, total),
        termination_distribution=terminations,
        latency_ms=latency_summary(latencies),
        mean_env_steps=float(np.mean(np.asarray(steps, dtype=np.float64))) if steps else 0.0,
        action_bound_violations=bound_violations,
        no_progress_events=no_progress,
        drop_slip_rate=(float(slips) / float(total)) if total else 0.0,
        final_object_target_error=float(np.mean(np.asarray(object_errors, dtype=np.float64))) if object_errors else 0.0,
        notes=["Success is taken from task predicate info['is_success'], not from timeout or done alone."],
    )

