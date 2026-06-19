#!/usr/bin/env python
"""Run scripted CRIE-BT ablation evaluations."""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rocobench.crie_bt.controllers import build_controller
from rocobench.crie_bt.executor import LearnedSkillExecutor, RRTSkillExecutor, ScriptedSkillExecutor
from rocobench.crie_bt.failure_injection import FailureInjectionConfig, build_scripted_executor_for_scenario
from rocobench.crie_bt.planner import LLMPlannerAdapter, ScriptedPlanner
from rocobench.crie_bt.status import ExecutionMode


class SyntheticEnv(object):
    def __init__(self, task: str, seed: int = 0) -> None:
        self.task = task
        self.seed = int(seed)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return {
            "objects": {
                "apple": {"position": [0.2, 0.0, 0.1]},
                "red_block": {"position": [0.2, 0.0, 0.1]},
                "object": {"position": [0.2, 0.0, 0.1]},
            },
            "targets": {
                "bin_front_left": {"position": [0.8, 0.0, 0.1]},
                "bin_front_right": {"position": [0.8, 0.2, 0.1]},
                "red_bin": {"position": [0.8, 0.0, 0.1]},
                "target": {"position": [0.8, 0.0, 0.1]},
            },
            "agents": {
                "Alice": {"gripper_position": [0.2, 0.0, 0.2], "held_object": None},
                "Bob": {"gripper_position": [0.2, 0.2, 0.2], "held_object": None},
            },
            "packed": {},
        }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _modes(value: str) -> List[str]:
    if value == "all":
        return [ExecutionMode.OPEN_LOOP.value, ExecutionMode.DIRECT_FEEDBACK.value, ExecutionMode.BT_MEDIATED.value]
    return [value]


def _build_planner(name: str):
    if name == "scripted":
        return ScriptedPlanner()
    if name == "llm":
        return LLMPlannerAdapter()
    raise ValueError("unsupported planner: {}".format(name))


def _build_executor(name: str, scenario: str):
    if name == "scripted":
        return build_scripted_executor_for_scenario(FailureInjectionConfig(scenario=scenario))
    if name == "rrt":
        return RRTSkillExecutor()
    if name == "learned":
        return LearnedSkillExecutor()
    raise ValueError("unsupported executor: {}".format(name))


def run(args) -> List[Dict[str, Any]]:
    random.seed(int(args.seed))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = []
    with open(args.output, "w", encoding="utf-8") as f:
        for episode in range(int(args.episodes)):
            for mode in _modes(args.mode):
                planner = _build_planner(args.planner)
                executor = _build_executor(args.executor, args.failure_scenario)
                controller = build_controller(
                    mode,
                    planner,
                    executor,
                    uncertainty_mode=args.uncertainty,
                    max_retries=int(args.max_retries),
                )
                env = SyntheticEnv(args.task, seed=int(args.seed) + episode)
                task_goal = "{} task: put apple into bin_front_left".format(args.task)
                try:
                    row = controller.run_episode(env, task_goal, int(args.max_steps))
                    row["episode"] = episode
                    row["task_name"] = args.task
                    row["skipped"] = False
                except NotImplementedError as exc:
                    row = {
                        "episode": episode,
                        "task_name": args.task,
                        "mode": mode,
                        "task": task_goal,
                        "success": False,
                        "steps": 0,
                        "completed_subtasks": 0,
                        "failed_subtasks": 0,
                        "planner_calls": 0,
                        "replans": 0,
                        "local_retries": 0,
                        "failure_counts": {},
                        "events": [],
                        "subtask_results": [],
                        "explanations": [],
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
                rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["sort", "cabinet", "rope", "sweep", "sandwich", "pack"], default="pack")
    parser.add_argument("--mode", choices=["open_loop", "direct_feedback", "bt_mediated", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/crie_bt/eval.jsonl")
    parser.add_argument("--executor", choices=["scripted", "rrt", "learned"], default="scripted")
    parser.add_argument("--planner", choices=["scripted", "llm"], default="scripted")
    parser.add_argument("--uncertainty", choices=["none", "heuristic", "ensemble_variance", "policy_metadata"], default="heuristic")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--failure-scenario",
        choices=["none", "missed_grasp", "slippage", "no_progress", "target_occupied", "human_interrupt"],
        default="none",
    )
    args = parser.parse_args(argv)
    rows = run(args)
    print(json.dumps({"output": args.output, "episodes": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
