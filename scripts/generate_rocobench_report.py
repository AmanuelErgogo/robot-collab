#!/usr/bin/env python
"""Regenerate metrics and report.md from a RoCoBench episodes.csv file."""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.rocobench_pack_v1.report import write_markdown_report, write_json
from benchmark.rocobench_pack_v1.metrics import aggregate_results
from benchmark.rocobench_pack_v1.results import BenchmarkEpisodeResult


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _result_from_row(row):
    extra = {}
    if row.get("extra_json"):
        extra = json.loads(row["extra_json"])
    return BenchmarkEpisodeResult(
        benchmark_id=row["benchmark_id"],
        benchmark_version=row["benchmark_version"],
        schema_version=row["schema_version"],
        protocol_version=row["protocol_version"],
        predicate_version=row["predicate_version"],
        variation_manifest_hash=row["variation_manifest_hash"],
        task_id=row["task_id"],
        variation_id=row["variation_id"],
        variation_hash=row["variation_hash"],
        track=row["track"],
        method=row["method"],
        episode_index=int(row["episode_index"]),
        seed=int(row["seed"]),
        success=_bool(row["success"]),
        overall_success=_bool(row["overall_success"]),
        learned_success=_bool(row["learned_success"]),
        fallback_success=_bool(row["fallback_success"]),
        termination_reason=row["termination_reason"],
        num_env_steps=int(row["num_env_steps"] or 0),
        sim_time_s=float(row["sim_time_s"] or 0.0),
        latency_p50_ms=float(row["latency_p50_ms"] or 0.0),
        latency_p95_ms=float(row["latency_p95_ms"] or 0.0),
        action_violations=int(row["action_violations"] or 0),
        drops=int(row["drops"] or 0),
        collisions=int(row["collisions"] or 0),
        final_error=float(row["final_error"] or 0.0),
        plan_valid=_bool(row["plan_valid"]),
        replans=int(row["replans"] or 0),
        fallback_attempts=int(row["fallback_attempts"] or 0),
        planner_latency_ms=float(row["planner_latency_ms"] or 0.0),
        token_count=int(row["token_count"] or 0),
        makespan_steps=int(row["makespan_steps"] or 0),
        parallel_speedup=float(row["parallel_speedup"] or 0.0),
        resource_violations=int(row["resource_violations"] or 0),
        central_stops=int(row["central_stops"] or 0),
        artifact_dir=row.get("artifact_dir", ""),
        extra=extra,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "run_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(run_dir, "episodes.csv"), "r", encoding="utf-8") as f:
        results = [_result_from_row(row) for row in csv.DictReader(f)]
    metrics = aggregate_results(results)
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    write_markdown_report(os.path.join(run_dir, "report.md"), manifest, metrics)
    print(json.dumps({"run_dir": run_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

