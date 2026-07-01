"""CRIE-Bench command-line interface.

Usage
-----
    python -m crie_bench tasks
    python -m crie_bench agents --task sandwich
    python -m crie_bench collect --task sandwich --num-episodes 200
    python -m crie_bench train   --task sandwich --skill PICK --agent Chad --model octo
    python -m crie_bench test    --task sandwich --skill PICK --agent Chad \\
                                 --checkpoint checkpoints/octo_sandwich_pick_chad
    python -m crie_bench skills
    python -m crie_bench skills  --task sandwich
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_tasks(args: argparse.Namespace) -> int:
    from .registry import TASK_REGISTRY, SANDWICH_RECIPES

    print(f"\n{'='*68}")
    print(f"  CRIE-Bench: available tasks")
    print(f"{'='*68}")
    for task_id, meta in sorted(TASK_REGISTRY.items()):
        agents_str = ", ".join(f"{a.name}({a.embodiment_id})" for a in meta.agents)
        skills_str = ", ".join(meta.skills)
        print(f"\n  {task_id}")
        print(f"    {meta.description}")
        print(f"    agents : {agents_str}")
        print(f"    skills : {skills_str}")
        if meta.recipes:
            print(f"    recipes: {', '.join(meta.recipes)}")
    print()
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    from .registry import TASK_REGISTRY

    task = args.task
    if task not in TASK_REGISTRY:
        print(f"[ERROR] Unknown task '{task}'. Run `crie_bench tasks` for the full list.")
        return 1

    meta = TASK_REGISTRY[task]
    print(f"\n{'='*60}")
    print(f"  Agents for task: {task}")
    print(f"{'='*60}")
    print(f"  {'name':<12} {'embodiment_id':<20} description")
    print(f"  {'-'*12} {'-'*20} {'-'*30}")
    for a in meta.agents:
        print(f"  {a.name:<12} {a.embodiment_id:<20} {a.description}")
    print()
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from .collect import CollectConfig, collect
    from .registry import DEFAULT_SKILLS_PER_TASK

    skills: List[str] = args.skills or DEFAULT_SKILLS_PER_TASK.get(args.task, ["PICK"])
    cfg = CollectConfig(
        task=args.task,
        skills=skills,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        render=args.render,
        verbose=args.verbose,
        seed=args.seed,
    )
    try:
        collect(cfg)
    except Exception as exc:
        print(f"[ERROR] collect failed: {exc}")
        if args.verbose:
            import traceback; traceback.print_exc()
        return 1
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    cfg = TrainConfig(
        task=args.task,
        skill=args.skill,
        agent=args.agent,
        model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained or "",
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    try:
        result = train(cfg)
        print(f"\nTraining {result.status}.")
        print(f"  checkpoint : {result.checkpoint_dir}")
        if result.manifest_path:
            print(f"  manifest   : {result.manifest_path}")
    except Exception as exc:
        print(f"[ERROR] train failed: {exc}")
        if args.verbose:
            import traceback; traceback.print_exc()
        return 1
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    from .evaluate import EvalConfig, evaluate

    cfg = EvalConfig(
        task=args.task,
        skill=args.skill,
        agent=args.agent,
        checkpoint=args.checkpoint,
        model=args.model,
        uncertainty_mode=args.uncertainty_mode,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        chunk_size=args.chunk_size,
        render=args.render,
        record_video=not args.no_video,
        video_fps=args.fps,
        video_cameras=tuple(args.cameras),
        artifact_dir=args.artifact_dir,
        seed=args.seed,
    )
    try:
        result = evaluate(cfg)
        sr = result.success_rate
        print(f"Success rate: {sr*100:.1f}%")
        return 0 if sr > 0 else 1
    except Exception as exc:
        print(f"[ERROR] test failed: {exc}")
        if getattr(args, "verbose", False):
            import traceback; traceback.print_exc()
        return 1


def cmd_skills(args: argparse.Namespace) -> int:
    from .skills_view import print_skills_table
    print_skills_table(task=getattr(args, "task", None))
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="crie_bench",
        description="CRIE-Bench: collect → train → test → inspect learned robot skills.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="command", metavar="COMMAND")

    # tasks -------------------------------------------------------------------
    p_tasks = sub.add_parser("tasks", help="List all available task environments.")

    # agents ------------------------------------------------------------------
    p_agents = sub.add_parser("agents", help="List robots/embodiments for a task.")
    p_agents.add_argument("--task", required=True, help="Task name (e.g. sandwich).")

    # collect -----------------------------------------------------------------
    p_col = sub.add_parser("collect", help="Collect scripted (RRT) demonstration data.")
    p_col.add_argument("--task", required=True)
    p_col.add_argument("--skills", nargs="+", default=None,
                       help="Skills to collect. Defaults to task defaults.")
    p_col.add_argument("--num-episodes", type=int, default=100, dest="num_episodes")
    p_col.add_argument("--output-dir", default="data/subtask_demos", dest="output_dir")
    p_col.add_argument("--render", action="store_true")
    p_col.add_argument("--verbose", action="store_true")
    p_col.add_argument("--seed", type=int, default=0)

    # train -------------------------------------------------------------------
    p_tr = sub.add_parser("train", help="Fine-tune a learned policy.")
    p_tr.add_argument("--task",     required=True)
    p_tr.add_argument("--skill",    required=True, help="Skill to train (e.g. PICK).")
    p_tr.add_argument("--agent",    required=True, help="Agent name (e.g. Chad).")
    p_tr.add_argument("--model",    default="octo", choices=("act", "octo"),
                       help="Policy architecture. Default: octo.")
    p_tr.add_argument("--data-dir", default="data/subtask_demos", dest="data_dir")
    p_tr.add_argument("--output-dir", default="checkpoints/octo", dest="output_dir")
    p_tr.add_argument("--pretrained", default="", help="Pre-trained checkpoint path or HF hub ID.")
    p_tr.add_argument("--num-steps", type=int, default=20_000, dest="num_steps")
    p_tr.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p_tr.add_argument("--chunk-size", type=int, default=10, dest="chunk_size")
    p_tr.add_argument("--save-interval", type=int, default=2_000, dest="save_interval")
    p_tr.add_argument("--eval-interval", type=int, default=2_000, dest="eval_interval")
    p_tr.add_argument("--device", default="cuda")
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="Validate config and print command without running training.")
    p_tr.add_argument("--verbose", action="store_true")

    # test --------------------------------------------------------------------
    p_te = sub.add_parser("test", help="Evaluate a trained policy.")
    p_te.add_argument("--task",       required=True)
    p_te.add_argument("--skill",      required=True)
    p_te.add_argument("--agent",      required=True)
    p_te.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    p_te.add_argument("--model",      default="octo", choices=("act", "octo", "mock"))
    p_te.add_argument("--uncertainty-mode", default="heuristic",
                       choices=("none", "heuristic", "policy_metadata", "ensemble_variance"),
                       dest="uncertainty_mode")
    p_te.add_argument("--num-episodes", type=int, default=20, dest="num_episodes")
    p_te.add_argument("--max-steps",    type=int, default=300, dest="max_steps")
    p_te.add_argument("--chunk-size",   type=int, default=10,  dest="chunk_size")
    p_te.add_argument("--render",       action="store_true")
    p_te.add_argument("--no-video",     action="store_true", dest="no_video",
                       help="Disable mp4 recording (faster).")
    p_te.add_argument("--fps",          type=int, default=20,
                       help="Video frame rate. Default: 20.")
    p_te.add_argument("--cameras",      nargs="+", default=["teaser", "top_cam"],
                       help="Camera names for video. Default: teaser top_cam.")
    p_te.add_argument("--artifact-dir", default="artifacts/eval", dest="artifact_dir")
    p_te.add_argument("--seed", type=int, default=0)

    # skills ------------------------------------------------------------------
    p_sk = sub.add_parser("skills", help="Show inventory of trained skill checkpoints.")
    p_sk.add_argument("--task", default=None,
                       help="Filter by task. Omit to show all tasks.")

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "tasks":   cmd_tasks,
    "agents":  cmd_agents,
    "collect": cmd_collect,
    "train":   cmd_train,
    "test":    cmd_test,
    "skills":  cmd_skills,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        print(f"[ERROR] Unknown command '{args.command}'.")
        return 1

    return handler(args)
