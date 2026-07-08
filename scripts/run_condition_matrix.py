#!/usr/bin/env python
"""Run the CRIE-BT / SARM condition matrix.

This is the single entry point that replaces the ad hoc ``--mode`` / ``--planner-mode``
combinations of the legacy runner.  Every run resolves ``(condition, stage)`` through
the condition registry, so every episode row records exactly which condition it
executed (see ``docs/crie_next_stage_plan/05_implementation_plan.md``, Milestone 0).

Examples
--------
Dry-run (no execution), just emit fully-populated condition metadata rows::

    python scripts/run_condition_matrix.py --stage step1 --dry-run --episodes 1

Step 1 robot-robot simulation on the synthetic backend::

    python scripts/run_condition_matrix.py --stage step1 \
        --tasks sandwich pack cabinet sort \
        --conditions VLM-RR-Cent VLM-RR-Dialog CRIE-BT-RR-Cent CRIE-BT-RR-Dialog \
        --episodes 10 --seeds 0 1 2 \
        --output results/step1_rr_sim/results.jsonl

Step 2 human-robot simulation with a scripted terminal human::

    python scripts/run_condition_matrix.py --stage step2 \
        --tasks medication_sim cooking_sim \
        --conditions VLM-HR-Cent CRIE-BT-HR-Cent \
        --human-interface scripted --episodes 5 \
        --output results/step2_hr_terminal/results.jsonl

The synthetic backend needs no MuJoCo/LLM.  Real RoCoBench (RRT) and learned-skill
(Step 3) backends attach through the documented adapters in
``rocobench.crie_bt.pipeline``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rocobench.crie_bt.pipeline import (
    ScriptedHumanCommunicationInterface,
    SyntheticEnvironmentAdapter,
    TerminalCommunicationInterface,
    build_collaboration_controller,
    build_condition,
    get_registry,
)
from rocobench.crie_bt.pipeline.conditions import ConditionConfig


def _stage_conditions(stage: str) -> List[str]:
    """Default set of condition names for a stage, from the registry."""
    registry = get_registry()
    stage_spec = registry.stage_spec(stage)
    teams = set(stage_spec.get("team_types", []))
    names = []
    for name in registry.condition_names():
        cfg = registry.build_condition(name, stage)
        if not teams or cfg.team_type in teams:
            names.append(name)
    return names


def _make_communication(kind: str, condition: ConditionConfig):
    if condition.team_type != "HR":
        return None
    if kind == "terminal":
        return TerminalCommunicationInterface()
    # scripted: auto-accept so batch runs never block on input().
    return ScriptedHumanCommunicationInterface(default_action="accept")


def _run_one(
    condition: ConditionConfig,
    task_id: str,
    seed: int,
    episode_index: int,
    max_steps: int,
    human_interface: str,
    events_dir: Optional[str],
    backend: str = "synthetic",
    llm_source: str = "gpt-4",
    api_key_path: str = "",
    record_video: bool = False,
) -> Dict[str, Any]:
    events_path = None
    tag = "{}_{}_seed{}_ep{:03d}".format(condition.code_name, task_id, seed, episode_index)
    if events_dir:
        events_path = os.path.join(events_dir, tag + "_events.jsonl")
    if backend == "roco":
        # Real RoCoBench MuJoCo + RRT + real LLM through the registry/pipeline.
        # Requires MUJOCO_GL (e.g. egl) and LLM credentials (see docs).
        from rocobench.crie_bt.pipeline.roco_backend import build_roco_condition

        video_dir = None
        if record_video and events_dir:
            video_dir = os.path.join(events_dir, "video", tag)  # -> execute.mp4
        controller, env = build_roco_condition(
            condition, task_id, seed=seed, llm_source=llm_source, api_key_path=api_key_path,
            prompt_save_dir=(os.path.join(events_dir, "prompts", tag) if events_dir else None),
            artifact_dir=video_dir,
        )
    else:
        env = SyntheticEnvironmentAdapter()
        communication = _make_communication(human_interface, condition)
        controller = build_collaboration_controller(condition, communication=communication)

    started = time.perf_counter()
    row = controller.run_episode(
        condition.condition_name, env, task_id, task_id, max_steps,
        seed=seed, episode_index=episode_index, events_path=events_path,
    )
    row["wall_time_s"] = round(time.perf_counter() - started, 6)
    # Keep the episode row itself compact; events already live in events.jsonl.
    row.pop("events", None)
    return row


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    conditions = args.conditions or _stage_conditions(args.stage)
    stage_spec = get_registry().stage_spec(args.stage)
    tasks = args.tasks or list(stage_spec.get("tasks", []))
    seeds = args.seeds or [args.seed]

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events_dir = None
    if not args.dry_run and not args.no_events:
        events_dir = os.path.join(os.path.dirname(output_path), "events")
        os.makedirs(events_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(output_path, "w", encoding="utf-8") as handle:
        for condition_name in conditions:
            condition = build_condition(condition_name, args.stage)
            for task_id in tasks:
                for seed in seeds:
                    for episode_index in range(int(args.episodes)):
                        if args.dry_run:
                            row = _dry_row(condition, task_id, seed, episode_index)
                        else:
                            row = _run_one(
                                condition, task_id, seed, episode_index,
                                int(args.max_steps), args.human_interface, events_dir,
                                backend=args.backend, llm_source=args.llm_source,
                                api_key_path=args.api_key_path, record_video=args.record_video,
                            )
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        rows.append(row)
    return rows


def _dry_row(condition: ConditionConfig, task_id: str, seed: int, episode_index: int) -> Dict[str, Any]:
    """A fully-populated metadata row with zeroed metrics (no execution)."""
    row = {
        "episode_id": "{stage}_{task}_seed{seed}_ep{idx:03d}_{code}".format(
            stage=condition.stage, task=task_id, seed=seed, idx=episode_index, code=condition.code_name),
        "stage": condition.stage,
        "task_id": task_id,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "success": False,
        "task_done": False,
        "completion_time_s": None,
        "wall_time_s": 0.0,
        "dry_run": True,
    }
    row.update(condition.metadata())
    for counter in ("num_steps", "planner_calls", "replans", "local_retries",
                    "failed_subtasks", "dialogue_turns", "human_interventions", "monitor_updates"):
        row[counter] = 0
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["step1", "step2", "step2a", "step3"], default="step1")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Condition names (paper or code names). Defaults to the stage's team.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task ids. Defaults to the stage's tasks.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--skill-backend", choices=["RRT", "LearnedSkill"], default=None,
                        help="Informational; the synthetic backend stands in for RRT in sim.")
    parser.add_argument("--monitor-backend", choices=["VLM-self", "CodedSim", "SARM"], default=None,
                        help="Informational; resolved from the registry per condition.")
    parser.add_argument("--human-interface", choices=["scripted", "terminal"], default="scripted")
    parser.add_argument("--backend", choices=["synthetic", "roco"], default="synthetic",
                        help="synthetic = deterministic stub; roco = real MuJoCo+RRT+LLM (needs MUJOCO_GL + creds).")
    parser.add_argument("--llm-source", default="gpt-4", help="LLM for --backend roco (e.g. gemini-2.5-flash).")
    parser.add_argument("--api-key-path", default="", help="Credential path for --backend roco.")
    parser.add_argument("--record-video", action="store_true",
                        help="Record execute.mp4 per episode under <output-dir>/events/video/ (roco backend only).")
    parser.add_argument("--output", default="results/condition_matrix/results.jsonl")
    parser.add_argument("--no-events", action="store_true", help="Do not write per-episode events.jsonl.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Emit fully-populated condition metadata rows without executing episodes.")
    args = parser.parse_args(argv)

    rows = run(args)
    successes = sum(1 for row in rows if row.get("success"))
    summary = {
        "output": args.output,
        "stage": args.stage,
        "dry_run": bool(args.dry_run),
        "episodes": len(rows),
        "successes": successes,
        "conditions": sorted({row["condition_name"] for row in rows}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
