#!/usr/bin/env python
"""Aggregate condition-matrix results into paper-ready tables.

Reads one or more ``results.jsonl`` files produced by ``run_condition_matrix.py``
and emits (see ``docs/crie_next_stage_plan/05_implementation_plan.md``, Milestone 5):

* a success-rate + efficiency table per condition;
* a replanning / retry table;
* a monitor-metric table (CRIE-BT conditions only);
* a dialogue-burden table (human-robot conditions);
* a per-task success breakdown.

Each table is written as ``.csv`` and as a LaTeX ``tabular`` snippet.  Pure
stdlib -- no pandas required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def _group(rows: List[Dict[str, Any]], key: Callable[[Dict[str, Any]], Any]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def summary_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group(rows, lambda r: r["condition_name"])
    out = []
    for condition, group in sorted(grouped.items()):
        out.append({
            "condition_name": condition,
            "controller_family": group[0]["controller_family"],
            "team_type": group[0]["team_type"],
            "coordination_mode": group[0]["coordination_mode"],
            "monitor_backend": group[0]["monitor_backend"],
            "n": len(group),
            "success_rate": _mean([1.0 if r.get("success") else 0.0 for r in group]),
            "mean_steps": _mean([r.get("num_steps") for r in group]),
            "mean_planner_calls": _mean([r.get("planner_calls") for r in group]),
            "mean_wall_time_s": _mean([r.get("wall_time_s") for r in group]),
        })
    return out


def replanning_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group(rows, lambda r: r["condition_name"])
    out = []
    for condition, group in sorted(grouped.items()):
        out.append({
            "condition_name": condition,
            "mean_replans": _mean([r.get("replans") for r in group]),
            "mean_local_retries": _mean([r.get("local_retries") for r in group]),
            "mean_failed_subtasks": _mean([r.get("failed_subtasks") for r in group]),
            "recovery_rate": _mean([
                1.0 if (r.get("failed_subtasks", 0) and r.get("success")) else 0.0
                for r in group if r.get("failed_subtasks", 0)
            ]),
        })
    return out


def monitor_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Explicit monitor metrics apply to CRIE-BT conditions only.
    monitored = [r for r in rows if r.get("controller_family") == "CRIE-BT"]
    grouped = _group(monitored, lambda r: r["condition_name"])
    out = []
    for condition, group in sorted(grouped.items()):
        out.append({
            "condition_name": condition,
            "monitor_backend": group[0]["monitor_backend"],
            "monitor_privileged": group[0]["monitor_privileged"],
            "mean_monitor_updates": _mean([r.get("monitor_updates") for r in group]),
        })
    return out


def dialogue_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hr = [r for r in rows if r.get("team_type") == "HR"]
    grouped = _group(hr, lambda r: r["condition_name"])
    out = []
    for condition, group in sorted(grouped.items()):
        out.append({
            "condition_name": condition,
            "mean_dialogue_turns": _mean([r.get("dialogue_turns") for r in group]),
            "mean_human_interventions": _mean([r.get("human_interventions") for r in group]),
        })
    return out


def per_task_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group(rows, lambda r: (r["condition_name"], r["task_id"]))
    out = []
    for (condition, task), group in sorted(grouped.items()):
        out.append({
            "condition_name": condition,
            "task_id": task,
            "n": len(group),
            "success_rate": _mean([1.0 if r.get("success") else 0.0 for r in group]),
        })
    return out


def write_csv(path: str, table: List[Dict[str, Any]]) -> None:
    if not table:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def write_latex(path: str, table: List[Dict[str, Any]], caption: str) -> None:
    if not table:
        return
    columns = list(table[0].keys())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_tex(c) for c in columns) + " \\\\",
        "\\hline",
    ]
    for row in table:
        lines.append(" & ".join(_tex(row[c]) for c in columns) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\caption{" + caption + "}", "\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


TABLES = {
    "summary": (summary_table, "Task success and efficiency by condition."),
    "replanning": (replanning_table, "Replanning, local retry, and recovery by condition."),
    "monitor": (monitor_table, "Progress-monitor metrics (CRIE-BT conditions)."),
    "dialogue": (dialogue_table, "Dialogue burden and human interventions (human-robot conditions)."),
    "per_task": (per_task_table, "Per-task success rate by condition."),
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="One or more results.jsonl files.")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Also print tables to stdout.")
    args = parser.parse_args(argv)

    rows = load_rows(args.results)
    os.makedirs(args.output_dir, exist_ok=True)

    for name, (builder, caption) in TABLES.items():
        table = builder(rows)
        write_csv(os.path.join(args.output_dir, name + ".csv"), table)
        write_latex(os.path.join(args.output_dir, name + ".tex"), table, caption)
        if args.do_print:
            print("\n== {} ==".format(name))
            for entry in table:
                print(json.dumps(entry, sort_keys=True))

    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir,
                      "tables": sorted(TABLES.keys())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
