#!/usr/bin/env python
"""Evaluate RoCoBench-Pack-Skills-v1 baselines."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.rocobench_pack_v1.config import METHODS, TRACKS, EvaluationRunConfig
from benchmark.rocobench_pack_v1.evaluator import BenchmarkEvaluator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="rocobench_pack_skills_v1")
    parser.add_argument("--track", choices=TRACKS, default="skill_policy")
    parser.add_argument("--method", "--baseline", dest="method", choices=METHODS, default="hold")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="artifacts/benchmark/run_001")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--request-timeout-ms", type=int, default=30000)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = EvaluationRunConfig(
        suite=args.suite,
        track=args.track,
        method=args.method,
        output_dir=args.output_dir,
        episodes_per_task=args.episodes_per_task,
        task_ids=tuple(args.task_id),
        policy_path=args.policy_path,
        endpoint=args.endpoint,
        request_timeout_ms=args.request_timeout_ms,
        execution_horizon=args.execution_horizon,
        record_video=args.record_video,
        overwrite=args.overwrite,
    )
    run_manifest, _results, metrics = BenchmarkEvaluator(config).evaluate()
    summary = {
        "output_dir": os.path.abspath(args.output_dir),
        "run_manifest": os.path.abspath(os.path.join(args.output_dir, "run_manifest.json")),
        "episodes_csv": os.path.abspath(os.path.join(args.output_dir, "episodes.csv")),
        "metrics_json": os.path.abspath(os.path.join(args.output_dir, "metrics.json")),
        "report_md": os.path.abspath(os.path.join(args.output_dir, "report.md")),
        "variation_manifest_hash": run_manifest["variation_manifest_hash"],
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

