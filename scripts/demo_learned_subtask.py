"""demo_learned_subtask.py — end-to-end demo of the learned subtask skill pipeline.

Demonstrates the full pipeline without a real neural-network checkpoint:

  LLM plan  →  SkillPlan  →  SubtaskLearnedExecutor  →  MockACTHandle (RRT-backed)
                                       ↓
                            UncertaintyEstimator (policy_metadata mode)
                                       ↓
                            SubtaskSuccessChecker  →  Result + artefacts

Usage
-----
    # Dry-run (no MuJoCo window)
    python scripts/demo_learned_subtask.py --task pack --skill PICK --object apple

    # With renderer + uncertainty mode switch
    python scripts/demo_learned_subtask.py \\
        --task sandwich --skill STACK_ON --object bacon --target cutting_board \\
        --uncertainty_mode heuristic --render

    # Run all skills on pack task
    python scripts/demo_learned_subtask.py --task pack --run_all --render
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rocobench.skills.models import SkillCall, SkillPlan, SkillExecutionStatus
from rocobench.skills.learned.registry import LearnedPolicyRegistry
from rocobench.skills.learned.models import LearnedPolicySpec
from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache
from rocobench.skills.learned.mock_handle import MockACTHandle
from rocobench.skills.learned.subtask_executor import SubtaskLearnedExecutor
from rocobench.skills.learned.subtask_skills import (
    build_registry_for_task,
    SKILL_WAIT,
)


# ---------------------------------------------------------------------------
# Task env factory
# ---------------------------------------------------------------------------

TASK_MAP = {
    "pack":     ("rocobench.envs.task_pack", "PackGroceryTask"),
    "sandwich": ("rocobench.envs.task_sandwich", "MakeSandwichTask"),
    "cabinet":  ("rocobench.envs.task_cabinet", "OpenCabinetTask"),
    "sort":     ("rocobench.envs.task_sort", "SortTask"),
    "rope":     ("rocobench.envs.task_rope", "RopeTask"),
    "sweep":    ("rocobench.envs.task_sweep", "SweepTask"),
}


def load_env(task_name: str, render: bool = False):
    module_path, class_name = TASK_MAP[task_name]
    import importlib
    cls = getattr(importlib.import_module(module_path), class_name)
    # MujocoSimEnv uses render_cameras / has_gui, not a simple 'render' flag
    kwargs = {}
    if not render:
        # Use a single minimal camera to keep render_all_cameras() happy while
        # avoiding an OpenGL display. The concatenation in base_env needs ≥1 camera.
        kwargs["render_cameras"] = ["teaser"]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Policy registry builder (mock — no real checkpoints needed)
# ---------------------------------------------------------------------------

def build_mock_policy_registry(
    env,
    robots: Dict[str, Any],
    skills: List[str],
    task_name: str,
    uncertainty_mode: str,
) -> tuple:
    """
    Build a LearnedPolicyRegistry populated with mock ACT specs for every
    (skill, agent, embodiment) combination. Returns (registry, cache).

    In production: replace MockACTHandle with a real LeRobot ACT handle and
    point checkpoint paths to your trained .pt files.
    """
    mock_handle = MockACTHandle(
        env=env,
        robots=robots,
        chunk_size=10,
        uncertainty_mode=uncertainty_mode,
        base_confidence=0.90,
    )

    registry = LearnedPolicyRegistry()
    agent_names = list(robots.keys())
    # robot_name_map_inv maps agent display name ("Alice") → MuJoCo body ("ur5e_robotiq")
    name_map_inv = getattr(env, "robot_name_map_inv", {})

    for skill_name in skills:
        for agent_name in agent_names:
            # Embodiment is the MuJoCo hardware name (e.g. "ur5e_robotiq")
            embodiment_id = name_map_inv.get(agent_name, agent_name)
            policy_id = f"{task_name}_{skill_name}_{agent_name}_mock_v1"
            spec = LearnedPolicySpec(
                policy_id=policy_id,
                skill_name=skill_name,
                agent_name=agent_name,
                embodiment_id=embodiment_id,
                task_id=task_name,
                checkpoint="mock://no_checkpoint",
                checkpoint_revision="mock-v1",
                policy_type="act",
                schema_hash="mock",
                action_representation="joint_ctrl",
                cameras=(),
                max_steps=300,
                execution_horizon=10,
                success_monitor="stable",
                failure_monitors=("NO_PROGRESS", "TIMEOUT"),
                enabled=True,
            )
            try:
                registry.register(spec)
            except Exception:
                pass  # skip duplicates in --run_all mode

    # Override validate_static so mock path "mock://..." doesn't fail os.path.exists
    _patch_registry_for_mock(registry)

    cache = BoundedPolicyHandleCache(loader=lambda spec: mock_handle, max_size=1)
    return registry, cache


def _patch_registry_for_mock(registry: LearnedPolicyRegistry):
    """Allow 'mock://' checkpoint paths to pass static validation."""
    original = registry.validate_static

    def patched_validate(spec):
        if spec.checkpoint.startswith("mock://"):
            return True
        return original(spec)

    registry.validate_static = patched_validate


# ---------------------------------------------------------------------------
# Single skill demo
# ---------------------------------------------------------------------------

def run_skill_demo(
    env,
    robots: Dict[str, Any],
    task_name: str,
    skill_name: str,
    call_args: Dict[str, str],
    uncertainty_mode: str,
    artifact_dir: str,
    max_steps: int = 300,
) -> Dict[str, Any]:
    """Run one skill with the SubtaskLearnedExecutor and return result dict."""

    skills_to_register = [skill_name, SKILL_WAIT]
    registry, cache = build_mock_policy_registry(
        env, robots, skills_to_register, task_name, uncertainty_mode
    )

    from rocobench.skills.learned.config import LearnedExecutorConfig
    executor = SubtaskLearnedExecutor(
        env=env,
        robots=robots,
        policy_registry=registry,
        policy_cache=cache,
        uncertainty_mode=uncertainty_mode,
        stable_success_checks=2,
        max_steps=max_steps,
        config=LearnedExecutorConfig(task_id=task_name),
    )

    # Build a minimal SkillPlan: primary agent executes skill, others WAIT
    agent_names = list(robots.keys())
    primary = agent_names[0]

    calls = []
    for agent_name in agent_names:
        if agent_name == primary:
            call = SkillCall(
                agent_name=agent_name,
                skill_name=skill_name,
                arguments=call_args,
                raw_action=f"ACTION {skill_name}({', '.join(f'{k}={v}' for k,v in call_args.items())})",
            )
        else:
            call = SkillCall(
                agent_name=agent_name,
                skill_name=SKILL_WAIT,
                arguments={},
                raw_action="ACTION WAIT",
            )
        calls.append(call)

    plan = SkillPlan(
        calls=calls,
        parsed_proposal=f"demo:{skill_name}",
    )

    obs = env.get_obs()
    t0 = time.monotonic()
    result = executor.execute(plan, obs, artifact_dir=artifact_dir)
    elapsed = time.monotonic() - t0

    # Load uncertainty trace from artefact
    unc_summary = {}
    unc_file = Path(artifact_dir) / "uncertainty_summary.json"
    if unc_file.exists():
        with open(unc_file) as f:
            unc_summary = json.load(f)

    return {
        "skill": skill_name,
        "call_args": call_args,
        "success": result.success,
        "status": result.status.value,
        "reason": result.reason,
        "num_sim_steps": result.num_sim_steps,
        "elapsed_s": round(elapsed, 2),
        "uncertainty_summary": unc_summary,
        "artifact_dir": artifact_dir,
    }


# ---------------------------------------------------------------------------
# Predefined demo sequences per task
# ---------------------------------------------------------------------------

DEMO_SEQUENCES = {
    "pack": [
        ("PICK",          {"object": "apple"},       None),
        ("PLACE",         {"target": "bin_inside"},  None),
        ("PICK_AND_PLACE",{"object": "banana", "target": "bin_inside"}, None),
    ],
    "sandwich": [
        ("PICK",     {"object": "bread_slice1"}, None),
        ("STACK_ON", {"object": "bread_slice1", "target": "cutting_board"}, None),
        ("PICK",     {"object": "bacon"},        None),
        ("STACK_ON", {"object": "bacon", "target": "bread_slice1"}, None),
    ],
    "cabinet": [
        ("PICK",         {"object": "cup"},       None),
        ("OPEN_CABINET", {"door": "left_door"},   None),
        ("PLACE",        {"target": "left_shelf"}, None),
    ],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASK_MAP))
    p.add_argument("--skill", default=None,
                   help="Single skill to demo (e.g. PICK). Ignored if --run_all.")
    p.add_argument("--object", default=None, help="Object argument for the skill.")
    p.add_argument("--target", default=None, help="Target argument for the skill.")
    p.add_argument("--door",   default=None, help="Door argument for OPEN_CABINET.")
    p.add_argument("--run_all", action="store_true",
                   help="Run the predefined demo sequence for the task.")
    p.add_argument("--uncertainty_mode",
                   choices=["none", "heuristic", "policy_metadata", "ensemble_variance"],
                   default="policy_metadata")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--artifact_dir", default="artifacts/subtask_demo")
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _print_result(r: Dict[str, Any]):
    status_icon = "✓" if r["success"] else "✗"
    print(f"\n  {status_icon} {r['skill']}({r['call_args']})")
    print(f"     status       : {r['status']}")
    if r["reason"]:
        print(f"     reason       : {r['reason']}")
    print(f"     sim steps    : {r['num_sim_steps']}")
    print(f"     wall time    : {r['elapsed_s']}s")
    unc = r.get("uncertainty_summary", {})
    if unc.get("n_chunks", 0) > 0:
        print(f"     uncertainty  : mean_conf={unc['mean_confidence']:.3f}  "
              f"min_conf={unc['min_confidence']:.3f}  "
              f"high_risk_chunks={unc['high_risk_chunks']}/{unc['n_chunks']}")
    print(f"     artifacts    : {r['artifact_dir']}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Learned Subtask Skill Demo")
    print(f"  Task            : {args.task}")
    print(f"  Uncertainty mode: {args.uncertainty_mode}")
    print(f"{'='*60}")

    env = load_env(args.task, render=args.render)
    env.reset()
    env.sample_initial_scene()
    robots = env.get_sim_robots()

    print(f"\n  Robots available: {list(robots.keys())}")

    results = []
    base_dir = Path(args.artifact_dir) / args.task

    if args.run_all:
        sequence = DEMO_SEQUENCES.get(args.task, [])
        if not sequence:
            print(f"  [WARNING] No predefined demo sequence for task '{args.task}'.")
            print(f"  Use --skill to specify a single skill.")
            return

        print(f"\n  Running {len(sequence)}-step demo sequence ...\n")
        for i, (skill_name, call_args, _) in enumerate(sequence):
            art_dir = str(base_dir / f"step_{i:02d}_{skill_name}")
            r = run_skill_demo(
                env, robots, args.task, skill_name, call_args,
                args.uncertainty_mode, art_dir, args.max_steps,
            )
            results.append(r)
            _print_result(r)
    else:
        if args.skill is None:
            print("  Provide --skill or use --run_all.")
            return

        call_args = {}
        if args.object: call_args["object"] = args.object
        if args.target: call_args["target"] = args.target
        if args.door:   call_args["door"]   = args.door

        art_dir = str(base_dir / f"{args.skill}")
        r = run_skill_demo(
            env, robots, args.task, args.skill, call_args,
            args.uncertainty_mode, art_dir, args.max_steps,
        )
        results.append(r)
        _print_result(r)

    # Summary
    n_ok = sum(r["success"] for r in results)
    print(f"\n{'='*60}")
    print(f"  Summary: {n_ok}/{len(results)} skills succeeded")
    if results:
        total_steps = sum(r["num_sim_steps"] for r in results)
        total_time  = sum(r["elapsed_s"] for r in results)
        print(f"  Total sim steps : {total_steps}")
        print(f"  Total wall time : {total_time:.1f}s")
    print(f"{'='*60}\n")

    # Save summary JSON
    summary_path = base_dir / "demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
