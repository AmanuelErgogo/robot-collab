"""Aggregate metrics and paired statistics for Phase 8 results."""

import math
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .results import BenchmarkEpisodeResult


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    n = float(total)
    phat = float(successes) / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def aggregate_results(results: Iterable[BenchmarkEpisodeResult]) -> Dict[str, Any]:
    rows = list(results)
    total = len(rows)
    successes = sum(1 for row in rows if row.success)
    learned_successes = sum(1 for row in rows if row.learned_success)
    fallback_successes = sum(1 for row in rows if row.fallback_success)
    latencies = [row.latency_p50_ms for row in rows if row.latency_p50_ms > 0]
    reasons = Counter(row.termination_reason for row in rows)
    action_violations = sum(row.action_violations for row in rows)
    collisions = sum(row.collisions for row in rows)
    drops = sum(row.drops for row in rows)
    return {
        "episode_count": total,
        "successes": successes,
        "success_rate": float(successes) / total if total else 0.0,
        "success_wilson_95": wilson_interval(successes, total),
        "learned_successes": learned_successes,
        "fallback_successes": fallback_successes,
        "termination_distribution": dict(sorted(reasons.items())),
        "mean_steps": sum(row.num_env_steps for row in rows) / float(total) if total else 0.0,
        "mean_sim_time_s": sum(row.sim_time_s for row in rows) / float(total) if total else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "action_violations": int(action_violations),
        "collisions": int(collisions),
        "drops": int(drops),
        "planner": {
            "valid_plans": sum(1 for row in rows if row.plan_valid),
            "replans": sum(row.replans for row in rows),
            "fallback_attempts": sum(row.fallback_attempts for row in rows),
        },
        "multi_agent": {
            "mean_makespan_steps": sum(row.makespan_steps for row in rows) / float(total) if total else 0.0,
            "mean_parallel_speedup": sum(row.parallel_speedup for row in rows) / float(total) if total else 0.0,
            "resource_violations": sum(row.resource_violations for row in rows),
            "central_stops": sum(row.central_stops for row in rows),
        },
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def mcnemar_exact(a_success: Sequence[bool], b_success: Sequence[bool]) -> Dict[str, Any]:
    if len(a_success) != len(b_success):
        raise ValueError("paired sequences must have the same length")
    b01 = 0
    b10 = 0
    for a, b in zip(a_success, b_success):
        a = bool(a)
        b = bool(b)
        if (not a) and b:
            b01 += 1
        elif a and (not b):
            b10 += 1
    discordant = b01 + b10
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(b01, b10) + 1)) / (2.0 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"b01": b01, "b10": b10, "discordant": discordant, "p_value": p_value}


def paired_bootstrap_difference(
    a_success: Sequence[bool],
    b_success: Sequence[bool],
    samples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    if len(a_success) != len(b_success):
        raise ValueError("paired sequences must have the same length")
    n = len(a_success)
    if n == 0:
        return {"mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    diffs = [float(_as_bool(a)) - float(_as_bool(b)) for a, b in zip(a_success, b_success)]
    rng = random.Random(seed)
    sampled: List[float] = []
    for _ in range(int(samples)):
        total = 0.0
        for _idx in range(n):
            total += diffs[rng.randrange(n)]
        sampled.append(total / float(n))
    sampled.sort()
    lo = sampled[int(0.025 * (len(sampled) - 1))]
    hi = sampled[int(0.975 * (len(sampled) - 1))]
    return {"mean_difference": sum(diffs) / float(n), "ci_low": lo, "ci_high": hi}


def paired_comparison(a_rows: Mapping[str, Mapping[str, Any]], b_rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(a_rows).intersection(b_rows))
    a_success = [_as_bool(a_rows[key].get("success", False)) for key in keys]
    b_success = [_as_bool(b_rows[key].get("success", False)) for key in keys]
    return {
        "paired_n": len(keys),
        "paired_keys": keys,
        "a_success_rate": sum(float(x) for x in a_success) / len(keys) if keys else 0.0,
        "b_success_rate": sum(float(x) for x in b_success) / len(keys) if keys else 0.0,
        "mcnemar": mcnemar_exact(a_success, b_success),
        "risk_difference_bootstrap": paired_bootstrap_difference(a_success, b_success),
    }

