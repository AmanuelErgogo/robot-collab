#!/usr/bin/env python
"""Validate the pipeline against the real RoCoBench simulator (Step 1).

Runs the new collaboration controllers over a real ``rocobench.envs`` task with
the legacy RRT skill stack and the coded simulator monitor. By default it issues
a single WAIT action per agent (valid, no motion planning) so it exercises the
whole integration path -- real env -> ObservationBundle/oracle_state -> RRT
executor -> coded monitor -> episode logger -- quickly and deterministically.

Requires an offscreen GL backend:

    MUJOCO_GL=egl python scripts/validate_roco_step1.py --task pack

To drive real manipulation, pass RoCoBench EXECUTE blocks via --response (repeat)
or swap ``RoCoScriptedPlanner`` for ``LLMPlannerAdapter`` wrapping a RoCo prompter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONBREAKPOINT", "0")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="pack", help="RoCoBench task id (pack, sandwich, cabinet, sweep, sort).")
    parser.add_argument("--conditions", nargs="*", default=["VLM-RR-Cent", "CRIE-BT-RR-Cent"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--response", action="append", default=None,
                        help="Raw RoCoBench EXECUTE block; repeat for multiple steps. Default: WAIT.")
    parser.add_argument("--output", default=None, help="Optional results.jsonl path.")
    args = parser.parse_args(argv)

    from rocobench.crie_bt.pipeline import build_collaboration_controller, build_condition
    from rocobench.crie_bt.pipeline.roco_backend import build_roco_step1

    rows = []
    for condition_name in args.conditions:
        condition = build_condition(condition_name, "step1")
        env_adapter, executor, planner_factory = build_roco_step1(
            args.task, seed=args.seed, responses=args.response)
        controller = build_collaboration_controller(
            condition, planner_factory=planner_factory, executor=executor)
        started = time.perf_counter()
        row = controller.run_episode(
            condition_name, env_adapter, args.task, args.task, args.max_steps, seed=args.seed)
        row["wall_time_s"] = round(time.perf_counter() - started, 3)
        row.pop("events", None)
        rows.append(row)
        print("{:18s} success={} steps={} planner_calls={} monitor_updates={} "
              "backend={} env={} [{}s]".format(
                  condition_name, row["success"], row["num_steps"], row["planner_calls"],
                  row["monitor_updates"], row["monitor_backend"], row["environment"], row["wall_time_s"]))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
