"""Report writers for Phase 8 benchmark runs."""

import json
import os
from typing import Any, Dict, Iterable, Mapping

from .metrics import aggregate_results
from .results import BenchmarkEpisodeResult, write_episodes_csv


def write_json(path: str, data: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, sort_keys=True)


def write_markdown_report(
    path: str,
    run_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    lines = [
        "# RoCoBench-Pack-Skills-v1 Report",
        "",
        "- benchmark: `{}`".format(run_manifest.get("benchmark_id", "")),
        "- version: `{}`".format(run_manifest.get("benchmark_version", "")),
        "- track: `{}`".format(run_manifest.get("track", "")),
        "- method: `{}`".format(run_manifest.get("method", "")),
        "- variation_manifest_hash: `{}`".format(run_manifest.get("variation_manifest_hash", "")),
        "- episodes: `{}`".format(metrics.get("episode_count", 0)),
        "- success_rate: `{:.4f}`".format(float(metrics.get("success_rate", 0.0))),
        "- wilson_95: `[{:.4f}, {:.4f}]`".format(
            float(metrics.get("success_wilson_95", {}).get("low", 0.0)),
            float(metrics.get("success_wilson_95", {}).get("high", 0.0)),
        ),
        "- learned_successes: `{}`".format(metrics.get("learned_successes", 0)),
        "- fallback_successes: `{}`".format(metrics.get("fallback_successes", 0)),
        "",
        "## Termination Reasons",
    ]
    for reason, count in sorted(metrics.get("termination_distribution", {}).items()):
        lines.append("- {}: `{}`".format(reason, count))
    lines.extend(
        [
            "",
            "## Discipline",
            "- Skill-policy and planner+skill tracks are reported separately.",
            "- Variation IDs and hashes are fixed before method execution.",
            "- Success uses task predicate evidence from the runtime path.",
            "- Learned, fallback, and overall success are separate fields.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_outputs(
    output_dir: str,
    run_manifest: Mapping[str, Any],
    results: Iterable[BenchmarkEpisodeResult],
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = list(results)
    metrics = aggregate_results(results)
    write_json(os.path.join(output_dir, "run_manifest.json"), run_manifest)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_episodes_csv(os.path.join(output_dir, "episodes.csv"), results)
    write_markdown_report(os.path.join(output_dir, "report.md"), run_manifest, metrics)
    return metrics

