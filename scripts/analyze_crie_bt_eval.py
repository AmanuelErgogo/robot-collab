#!/usr/bin/env python
"""Aggregate CRIE-BT JSONL evaluation logs."""

import argparse
import csv
import json
import os
from collections import defaultdict


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(path):
    rows = list(_read_jsonl(path))
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row.get("mode", "")].append(row)
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
        failure_counts = defaultdict(int)
        for item in items:
            for key, value in dict(item.get("failure_counts", {})).items():
                failure_counts[key] += int(value)
        summaries.append({
            "mode": mode,
            "episodes": n,
            "success_rate": float(success) / float(n),
            "avg_steps": float(steps) / float(n),
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
        "avg_steps",
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
        lines.append("- `{}`: success_rate=`{:.3f}`, avg_steps=`{:.2f}`, avg_replans=`{:.2f}`, avg_local_retries=`{:.2f}`".format(
            row["mode"],
            row["success_rate"],
            row["avg_steps"],
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
    args = parser.parse_args(argv)
    summaries = summarize(args.input)
    csv_path, md_path = write_outputs(summaries, args.output_dir)
    print(json.dumps({"summary_csv": csv_path, "summary_md": md_path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
