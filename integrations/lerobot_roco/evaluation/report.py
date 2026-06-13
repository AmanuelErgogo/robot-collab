"""Markdown and CSV reporting for Phase 4 evaluations."""

import csv
import os
from typing import Any, Iterable, Mapping

from integrations.lerobot_roco.dataset.manifest import atomic_write_json

from .metrics import aggregate_rollout_results


def write_episodes_csv(path: str, results: Iterable[Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = []
    for idx, result in enumerate(results):
        rows.append(
            {
                "episode_index": idx,
                "success": bool(getattr(result, "success", False)),
                "termination_reason": str(getattr(result, "termination_reason", "")),
                "num_env_steps": int(getattr(result, "num_env_steps", 0)),
                "sim_time": float(getattr(result, "sim_time", 0.0)),
                "artifact_dir": str(getattr(result, "artifact_dir", "")),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_index", "success", "termination_reason", "num_env_steps", "sim_time", "artifact_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_distribution_csv(path: str, distribution: Mapping[str, int]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["termination_reason", "count"])
        writer.writeheader()
        for key, value in sorted(distribution.items()):
            writer.writerow({"termination_reason": key, "count": int(value)})


def write_latency_csv(path: str, results: Iterable[Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "chunk_index", "latency_ms"])
        writer.writeheader()
        for episode_index, result in enumerate(results):
            for chunk_index, value in enumerate(getattr(result, "inference_latency_ms", ())):
                writer.writerow(
                    {
                        "episode_index": int(episode_index),
                        "chunk_index": int(chunk_index),
                        "latency_ms": float(value),
                    }
                )


def write_report(output_dir: str, config: Any, results: Iterable[Any], extra: Mapping[str, Any] = None) -> Mapping[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = list(results)
    aggregate = aggregate_rollout_results(results)
    metrics = aggregate.to_dict()
    atomic_write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_episodes_csv(os.path.join(output_dir, "episodes.csv"), results)
    write_distribution_csv(os.path.join(output_dir, "termination_reasons.csv"), aggregate.termination_distribution)
    write_latency_csv(os.path.join(output_dir, "latency.csv"), results)

    lines = [
        "# Phase 4 Direct ACT Evaluation",
        "",
        "- run: `{}`".format(getattr(config, "name", "")),
        "- split: `{}`".format(getattr(config, "split", "")),
        "- frozen_suite: `{}`".format(getattr(config, "frozen_suite", False)),
        "- checkpoint: `{}`".format(getattr(config, "checkpoint_dir", "")),
        "- execution_horizon: `{}`".format(getattr(config, "execution_horizon", None)),
        "- episodes: `{}`".format(metrics["episode_count"]),
        "- success_rate: `{:.4f}`".format(metrics["success_rate"]),
        "- wilson_95: `[{:.4f}, {:.4f}]`".format(
            metrics["success_wilson_95"]["low"], metrics["success_wilson_95"]["high"]
        ),
        "- latency_p50_ms: `{:.3f}`".format(metrics["latency_ms"]["p50"]),
        "- latency_p95_ms: `{:.3f}`".format(metrics["latency_ms"]["p95"]),
        "",
        "## Termination Reasons",
    ]
    for key, value in sorted(metrics["termination_distribution"].items()):
        lines.append("- {}: `{}`".format(key, value))
    lines.extend(
        [
            "",
            "## Discipline",
            "- Direct policy rollout only; planner and fallback are disconnected.",
            "- Success comes from `info['is_success']` task predicate evidence.",
            "- Test suites marked frozen should not be changed after viewing results.",
            "",
        ]
    )
    if extra:
        lines.extend(["## Extra", ""])
        for key, value in sorted(extra.items()):
            lines.append("- {}: `{}`".format(key, value))
        lines.append("")
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return metrics

