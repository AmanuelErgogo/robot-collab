#!/usr/bin/env python
"""Compare Phase 4 ACT metrics against an explicit RRT/expert metrics file."""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.dataset.manifest import atomic_write_json


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _success_rate(metrics):
    if "success_rate" in metrics:
        return float(metrics["success_rate"])
    total = float(metrics.get("episode_count", metrics.get("episodes", 0)) or 0)
    successes = float(metrics.get("successes", metrics.get("successful_episodes", 0)) or 0)
    return successes / total if total else 0.0


def compare_metrics(act_metrics_path, rrt_metrics_path, output_dir):
    act = _read_json(act_metrics_path)
    rrt = _read_json(rrt_metrics_path)
    comparison = {
        "act_metrics_path": os.path.abspath(act_metrics_path),
        "rrt_metrics_path": os.path.abspath(rrt_metrics_path),
        "act_success_rate": _success_rate(act),
        "rrt_success_rate": _success_rate(rrt),
        "success_rate_delta_act_minus_rrt": _success_rate(act) - _success_rate(rrt),
        "act_termination_distribution": act.get("termination_distribution", {}),
        "rrt_termination_distribution": rrt.get("termination_distribution", {}),
        "fallback_used": False,
        "note": "This is an explicit metrics comparison; the ACT rollout did not call RRT fallback.",
    }
    os.makedirs(output_dir, exist_ok=True)
    atomic_write_json(os.path.join(output_dir, "rrt_comparison.json"), comparison)
    lines = [
        "# ACT vs RRT Comparison",
        "",
        "- ACT success_rate: `{:.4f}`".format(comparison["act_success_rate"]),
        "- RRT success_rate: `{:.4f}`".format(comparison["rrt_success_rate"]),
        "- delta ACT-RRT: `{:.4f}`".format(comparison["success_rate_delta_act_minus_rrt"]),
        "- fallback_used: `False`",
        "",
    ]
    with open(os.path.join(output_dir, "rrt_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare direct ACT metrics with explicit RRT metrics.")
    parser.add_argument("--act-metrics", required=True, help="Phase 4 metrics.json")
    parser.add_argument("--rrt-metrics", required=True, help="RRT/expert metrics JSON to compare against.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    comparison = compare_metrics(args.act_metrics, args.rrt_metrics, args.output_dir)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

