#!/usr/bin/env python
"""Aggregate CRIE-BT JSONL evaluation logs."""

import argparse
import csv
import json
import os
from collections import defaultdict
from math import sqrt


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mean(values):
    values = list(values)
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0 if values else None
    avg = _mean(values)
    return sqrt(sum((float(value) - avg) ** 2 for value in values) / float(len(values)))


def _numeric_values(items, key):
    values = []
    for item in items:
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _nested_numeric_values(items, key):
    values = []
    for item in items:
        for value in item.get(key, []) or []:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return values


def _optional_rate(numerator, denominator):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _group_value(row, group_by):
    if group_by == "task_mode":
        return "{}:{}".format(row.get("task_id", ""), row.get("mode", ""))
    if group_by == "task_method":
        return "{}:{}".format(row.get("task_id", ""), row.get("paper_method", row.get("mode", "")))
    return row.get("mode", "")


def summarize(path):
    return summarize_grouped(path, group_by="mode")


def summarize_grouped(path, group_by="mode"):
    rows = list(_read_jsonl(path))
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[_group_value(row, group_by)].append(row)
    summaries = []
    for mode, items in sorted(by_mode.items()):
        n = len(items)
        if n == 0:
            continue
        success = sum(1 for item in items if item.get("success"))
        planner_calls = sum(int(item.get("planner_calls", 0)) for item in items)
        replans = sum(int(item.get("replans", 0)) for item in items)
        retries = sum(int(item.get("local_retries", 0)) for item in items)
        explanations = sum(len(item.get("explanations", [])) for item in items)
        steps = sum(int(item.get("steps", 0)) for item in items)
        sim_success_items = [item for item in items if "sim_success" in item]
        sim_success = sum(1 for item in sim_success_items if item.get("sim_success"))
        wall_times = _numeric_values(items, "wall_time_s")
        llm_latencies = _nested_numeric_values(items, "llm_call_latencies_s")
        prompt_tokens = _numeric_values(items, "llm_prompt_tokens")
        completion_tokens = _numeric_values(items, "llm_completion_tokens")
        total_tokens = _numeric_values(items, "llm_total_tokens")
        reactivity = _numeric_values(items, "reactivity_s")
        hallucination_count = sum(int(item.get("hallucination_count", 0) or 0) for item in items)
        hallucination_annotations = sum(int(item.get("hallucination_annotation_count", 0) or 0) for item in items)
        failure_counts = defaultdict(int)
        for item in items:
            for key, value in dict(item.get("failure_counts", {})).items():
                failure_counts[key] += int(value)
        summaries.append({
            "mode": mode,
            "episodes": n,
            "success_rate": float(success) / float(n),
            "task_success_rate": _optional_rate(sim_success, len(sim_success_items)),
            "avg_steps": float(steps) / float(n),
            "avg_wall_time_s": _mean(wall_times),
            "std_wall_time_s": _std(wall_times),
            "avg_llm_latency_s": _mean(llm_latencies),
            "avg_llm_prompt_tokens": _mean(prompt_tokens),
            "avg_llm_completion_tokens": _mean(completion_tokens),
            "avg_llm_total_tokens": _mean(total_tokens),
            "avg_reactivity_s": _mean(reactivity),
            "hallucination_rate": _optional_rate(hallucination_count, hallucination_annotations),
            "avg_planner_calls": float(planner_calls) / float(n),
            "avg_replans": float(replans) / float(n),
            "avg_local_retries": float(retries) / float(n),
            "failure_counts": dict(failure_counts),
            "recovery_rate": float(success) / float(max(1, sum(failure_counts.values()))),
            "explanation_count": explanations,
            "unnecessary_replanning_proxy": replans,
        })
    return summaries


def write_outputs(summaries, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "summary.csv")
    md_path = os.path.join(output_dir, "summary.md")
    fields = [
        "mode",
        "episodes",
        "success_rate",
        "task_success_rate",
        "avg_steps",
        "avg_wall_time_s",
        "std_wall_time_s",
        "avg_llm_latency_s",
        "avg_llm_prompt_tokens",
        "avg_llm_completion_tokens",
        "avg_llm_total_tokens",
        "avg_reactivity_s",
        "hallucination_rate",
        "avg_planner_calls",
        "avg_replans",
        "avg_local_retries",
        "recovery_rate",
        "explanation_count",
        "unnecessary_replanning_proxy",
        "failure_counts",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            copied = dict(row)
            copied["failure_counts"] = json.dumps(copied["failure_counts"], sort_keys=True)
            writer.writerow(copied)
    lines = ["# CRIE-BT Evaluation Summary", ""]
    for row in summaries:
        task_success = row["task_success_rate"]
        wall_time = row["avg_wall_time_s"]
        tokens = row["avg_llm_total_tokens"]
        extras = []
        if task_success is not None:
            extras.append("task_success_rate=`{:.3f}`".format(task_success))
        if wall_time is not None:
            extras.append("avg_wall_time_s=`{:.2f}`".format(wall_time))
        if tokens is not None:
            extras.append("avg_llm_total_tokens=`{:.0f}`".format(tokens))
        if not extras:
            extras.append("avg_steps=`{:.2f}`".format(row["avg_steps"]))
        lines.append("- `{}`: success_rate=`{:.3f}`, {}, avg_replans=`{:.2f}`, avg_local_retries=`{:.2f}`".format(
            row["mode"],
            row["success_rate"],
            ", ".join(extras),
            row["avg_replans"],
            row["avg_local_retries"],
        ))
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return csv_path, md_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL file produced by run_crie_bt_eval.py")
    parser.add_argument("--output-dir", default="results/crie_bt/analysis")
    parser.add_argument("--group-by", choices=["mode", "task_mode", "task_method"], default="mode")
    args = parser.parse_args(argv)
    summaries = summarize_grouped(args.input, group_by=args.group_by)
    csv_path, md_path = write_outputs(summaries, args.output_dir)
    print(json.dumps({"summary_csv": csv_path, "summary_md": md_path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
