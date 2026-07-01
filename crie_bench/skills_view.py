"""Display the inventory of trained skill checkpoints."""

from __future__ import annotations

from typing import List, Optional

from .registry import TrainedSkillEntry, scan_trained_skills, TASK_REGISTRY


def _check(exists: bool) -> str:
    return "OK" if exists else "MISSING"


def print_skills_table(task: Optional[str] = None, configs_dir: Optional[str] = None) -> None:
    entries = scan_trained_skills(configs_dir=configs_dir, task_filter=task)

    title = f"Trained skill inventory"
    if task:
        title += f" — task: {task}"
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    if not entries:
        print("  (no trained skills found — run `crie_bench train` to add one)")
        print()
        return

    # Group by task
    by_task: dict = {}
    for e in entries:
        by_task.setdefault(e.task_id, []).append(e)

    for task_id, task_entries in sorted(by_task.items()):
        print(f"\n  Task: {task_id}")
        print(f"  {'policy_id':<35} {'skill':<14} {'agent':<8} {'type':<6} {'ckpt':>8}  {'enabled'}")
        print(f"  {'-'*35} {'-'*14} {'-'*8} {'-'*6} {'-'*8}  {'-'*7}")
        for e in sorted(task_entries, key=lambda x: (x.agent_name, x.skill_name)):
            enabled_str = "yes" if e.enabled else "no"
            ckpt_str = _check(e.checkpoint_exists)
            print(f"  {e.policy_id:<35} {e.skill_name:<14} {e.agent_name:<8} "
                  f"{e.policy_type:<6} {ckpt_str:>8}  {enabled_str}")
    print()
