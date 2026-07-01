"""test_all_checkpoints.py — run every enabled skill checkpoint and produce artifacts.

For each enabled policy found in configs/skills/*.yaml this script:
  1. Runs N rollout episodes using the appropriate policy handle (octo / act / mock).
  2. Exports per-episode video (mp4) from the env's render buffers.
  3. Writes a JSON summary with success rate, mean steps, uncertainty stats.
  4. Writes a combined eval_report.json across all skills.

Usage
-----
    python scripts/test_all_checkpoints.py                         # all enabled skills
    python scripts/test_all_checkpoints.py --task sandwich         # filter by task
    python scripts/test_all_checkpoints.py --policy-id mock_sandwich_pick_chad
    python scripts/test_all_checkpoints.py --num-episodes 3 --fps 20
    python scripts/test_all_checkpoints.py --dry-run               # list what would run

Artifacts written to --artifact-dir (default: artifacts/checkpoint_tests/):
    <task>/<skill>/<agent>/<policy_id>/
        ep0000/
            execute.mp4          # episode video
            result.json          # per-episode result
        eval_summary.json        # aggregated success rate, mean steps
    eval_report.json             # top-level report across all skills
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from crie_bench.registry import scan_trained_skills, TrainedSkillEntry, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EpisodeArtifact:
    episode: int
    success: bool
    num_steps: int
    wall_time_s: float
    video_path: str
    mean_confidence: float = 1.0
    high_risk_chunks: int = 0
    total_chunks: int = 1


@dataclass
class SkillTestResult:
    policy_id: str
    task_id: str
    skill_name: str
    agent_name: str
    policy_type: str
    checkpoint: str
    episodes: List[EpisodeArtifact] = field(default_factory=list)
    error: str = ""

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def mean_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.num_steps for e in self.episodes) / len(self.episodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "skill_name": self.skill_name,
            "agent_name": self.agent_name,
            "policy_type": self.policy_type,
            "checkpoint": self.checkpoint,
            "num_episodes": len(self.episodes),
            "success_rate": round(self.success_rate, 3),
            "mean_steps": round(self.mean_steps, 1),
            "error": self.error,
            "episodes": [asdict(e) for e in self.episodes],
        }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _load_env(task_id: str, render_cameras: List[str]):
    meta = TASK_REGISTRY.get(task_id)
    if meta is None:
        raise ValueError(f"Unknown task '{task_id}'")
    module_path, class_name = meta.class_path.split(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(render_cameras=render_cameras)


# ---------------------------------------------------------------------------
# Policy handle factory
# ---------------------------------------------------------------------------

def _build_handle(entry: TrainedSkillEntry, spec, env=None, robots=None):
    ptype = entry.policy_type.lower()
    if ptype == "mock":
        from rocobench.skills.learned.mock_handle import MockACTHandle
        return MockACTHandle(env=env, robots=robots or {}, chunk_size=spec.execution_horizon)
    if ptype == "bc_nn":
        from rocobench.skills.learned.bc_nn_handle import BCNNHandle
        return BCNNHandle(entry.checkpoint, chunk_size=spec.execution_horizon)
    if ptype == "octo":
        from integrations.octo_roco.handle import OctoHandle
        # Use from_finetuned_npz if the checkpoint dir has our npz format
        ckpt = entry.checkpoint
        from pathlib import Path as _Path
        if (_Path(ckpt) / "train_meta.json").exists() or any(
            d.name.startswith("step_") for d in _Path(ckpt).iterdir() if d.is_dir()
        ) if _Path(ckpt).is_dir() else False:
            return OctoHandle.from_finetuned_npz(ckpt, spec)
        return OctoHandle.from_checkpoint(ckpt, spec)
    if ptype in ("act", "lerobot"):
        from integrations.lerobot_roco.evaluation.policy_loader import load_policy
        from integrations.lerobot_roco.evaluation.lerobot_handle import LeRobotACTHandle
        policy = load_policy(entry.checkpoint)
        return LeRobotACTHandle(policy, chunk_size=spec.execution_horizon)
    raise ValueError(f"Unsupported policy_type '{entry.policy_type}'")


# ---------------------------------------------------------------------------
# Per-skill test runner
# ---------------------------------------------------------------------------

def _run_skill(
    entry: TrainedSkillEntry,
    num_episodes: int,
    artifact_root: str,
    fps: int,
    render_cameras: List[str],
    seed: int,
) -> SkillTestResult:
    from rocobench.skills.models import SkillCall, SkillPlan
    from rocobench.skills.learned.registry import LearnedPolicyRegistry
    from rocobench.skills.learned.models import LearnedPolicySpec
    from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache
    from rocobench.skills.learned.subtask_executor import SubtaskLearnedExecutor
    from rocobench.skills.learned.config import LearnedExecutorConfig

    result = SkillTestResult(
        policy_id=entry.policy_id,
        task_id=entry.task_id,
        skill_name=entry.skill_name,
        agent_name=entry.agent_name,
        policy_type=entry.policy_type,
        checkpoint=entry.checkpoint,
    )

    skill_artifact_dir = os.path.join(
        artifact_root, entry.task_id, entry.skill_name, entry.agent_name, entry.policy_id
    )
    Path(skill_artifact_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    try:
        env = _load_env(entry.task_id, render_cameras)
        robots = env.get_sim_robots()

        spec = LearnedPolicySpec(
            policy_id=entry.policy_id,
            skill_name=entry.skill_name,
            agent_name=entry.agent_name,
            embodiment_id=entry.embodiment_id,
            task_id=entry.task_id,
            checkpoint=entry.checkpoint,
            checkpoint_revision=entry.config_file,
            policy_type=entry.policy_type,
            schema_hash=getattr(entry, "schema_hash", ""),
            action_representation=getattr(entry, "action_representation", "joint_ctrl"),
            cameras=tuple(render_cameras),
            max_steps=getattr(entry, "max_steps", 300),
            execution_horizon=getattr(entry, "execution_horizon", 10),
            success_monitor="stable",
            failure_monitors=getattr(entry, "failure_monitors",
                                     ("TIMEOUT", "NO_PROGRESS", "NONFINITE_ACTION")),
            enabled=True,
        )

        registry = LearnedPolicyRegistry()
        registry.register(spec)

        handle = _build_handle(entry, spec, env=env, robots=robots)
        cache = BoundedPolicyHandleCache(loader=lambda s: handle, max_size=4)

        executor = SubtaskLearnedExecutor(
            env=env,
            robots=robots,
            policy_registry=registry,
            policy_cache=cache,
            uncertainty_mode="heuristic",
            config=LearnedExecutorConfig(task_id=entry.task_id),
        )

        default_args = _default_args(entry.task_id, entry.skill_name)

        for ep in range(num_episodes):
            ep_dir = os.path.join(skill_artifact_dir, f"ep{ep:04d}")
            Path(ep_dir).mkdir(parents=True, exist_ok=True)

            # Reset env and clear render buffers
            obs = env.reset()
            if hasattr(env, "sample_initial_scene"):
                env.sample_initial_scene()
            obs = env.get_obs()
            _clear_render_buffers(env)

            skill_call = SkillCall(
                agent_name=entry.agent_name,
                skill_name=entry.skill_name,
                arguments=default_args,
                raw_action=f"NAME {entry.agent_name} ACTION {entry.skill_name}",
            )
            plan = SkillPlan(calls=[skill_call], parsed_proposal="scripted", plan_id=f"ep{ep}")

            t0 = time.perf_counter()
            exec_result = executor.execute(plan, obs, artifact_dir=ep_dir)
            wall_time = time.perf_counter() - t0

            # Export video from buffered render frames
            video_path = os.path.join(ep_dir, "execute.mp4")
            _export_video(env, video_path, fps=fps)

            # Write per-episode result
            result_dict = {
                "episode": ep,
                "success": exec_result.success,
                "num_steps": exec_result.num_sim_steps,
                "wall_time_s": round(wall_time, 2),
                "status": str(exec_result.status),
                "reason": exec_result.reason,
                "video": video_path,
            }
            meta = exec_result.metadata or {}
            unc = meta.get("uncertainty_summary", {})
            result_dict["mean_confidence"] = float(unc.get("mean_confidence", 1.0))
            result_dict["high_risk_chunks"] = int(unc.get("high_risk_chunks", 0))
            result_dict["total_chunks"] = int(unc.get("n_chunks", 1))

            with open(os.path.join(ep_dir, "result.json"), "w") as fh:
                json.dump(result_dict, fh, indent=2)

            ep_artifact = EpisodeArtifact(
                episode=ep,
                success=exec_result.success,
                num_steps=exec_result.num_sim_steps,
                wall_time_s=round(wall_time, 2),
                video_path=video_path,
                mean_confidence=result_dict["mean_confidence"],
                high_risk_chunks=result_dict["high_risk_chunks"],
                total_chunks=result_dict["total_chunks"],
            )
            result.episodes.append(ep_artifact)

            status = "OK" if exec_result.success else "FAIL"
            frames = _count_render_frames(env)
            print(
                f"    ep {ep:03d}  {status}  steps={exec_result.num_sim_steps}  "
                f"conf={result_dict['mean_confidence']:.2f}  "
                f"video={frames}fr → {video_path}"
            )

    except Exception as exc:
        result.error = str(exc)
        traceback.print_exc()

    # Write skill-level summary
    summary_path = os.path.join(skill_artifact_dir, "eval_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    return result


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _clear_render_buffers(env) -> None:
    """Clear stale frames from previous episodes."""
    bufs = getattr(env, "render_buffers", {})
    for buf in bufs.values():
        buf.clear()


def _count_render_frames(env) -> int:
    bufs = getattr(env, "render_buffers", {})
    if not bufs:
        return 0
    return max(len(b) for b in bufs.values())


def _export_video(env, video_path: str, fps: int = 20) -> bool:
    """Export buffered frames to mp4.  Returns True if video was written."""
    try:
        frames = _count_render_frames(env)
        if frames == 0:
            print(f"    [WARN] No render frames buffered — skipping video export.")
            return False
        env.export_render_to_video(video_path, out_type="mp4", fps=fps, concat=True)
        return True
    except Exception as exc:
        print(f"    [WARN] Video export failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Default skill arguments per task
# ---------------------------------------------------------------------------

_DEFAULT_ARGS: Dict[str, Dict[str, Dict[str, str]]] = {
    "sandwich": {
        "PICK":     {"object": "tomato"},
        "STACK_ON": {"target": "bread_slice1"},
    },
    "pack": {
        "PICK":             {"object": "apple"},
        "PLACE":            {"target": "container"},
        "PICK_AND_PLACE":   {"object": "apple", "target": "container"},
        "PUT_OBJECT_IN_CONTAINER": {"object": "apple", "target": "container"},
    },
    "cabinet": {
        "PICK":         {"object": "mug"},
        "PLACE":        {"target": "shelf"},
        "OPEN_CABINET": {"door": "door_handle"},
    },
    "sort":  {"PICK": {"object": "block_red"}, "PLACE": {"target": "bin_red"}},
    "rope":  {"PICK": {"object": "rope_end_left"}, "PLACE": {"target": "target"}},
    "sweep": {"SWEEP": {"target": "debris"}},
}


def _default_args(task_id: str, skill_name: str) -> Dict[str, str]:
    return _DEFAULT_ARGS.get(task_id, {}).get(skill_name, {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task",        default=None, help="Filter by task name.")
    p.add_argument("--policy-id",   default=None, dest="policy_id",
                   help="Run only this policy_id.")
    p.add_argument("--num-episodes",type=int, default=3, dest="num_episodes",
                   help="Episodes per skill. Default: 3.")
    p.add_argument("--artifact-dir",default="artifacts/checkpoint_tests",
                   dest="artifact_dir",
                   help="Root directory for artifacts. Default: artifacts/checkpoint_tests/")
    p.add_argument("--fps",         type=int, default=20,
                   help="Video frame rate. Default: 20.")
    p.add_argument("--cameras",     nargs="+",
                   default=["teaser", "top_cam"],
                   help="Camera names for video. Default: teaser top_cam.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--dry-run",     action="store_true", dest="dry_run",
                   help="List what would run without executing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    all_entries = scan_trained_skills(task_filter=args.task)
    # Only run enabled skills with available checkpoints
    entries = [
        e for e in all_entries
        if e.enabled and e.checkpoint_exists
        and (args.policy_id is None or e.policy_id == args.policy_id)
    ]

    if not entries:
        print("[INFO] No enabled skills with existing checkpoints found.")
        print("  Run `python -m crie_bench skills` to see the inventory.")
        return 0

    print(f"\n{'='*70}")
    print(f"  CRIE-Bench checkpoint tester")
    print(f"  Found {len(entries)} testable skill(s):")
    for e in entries:
        print(f"    {e.policy_id}  ({e.task_id} / {e.skill_name} / {e.agent_name})")
    print(f"  episodes    : {args.num_episodes}")
    print(f"  cameras     : {args.cameras}")
    print(f"  artifact_dir: {args.artifact_dir}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[dry-run] Exiting without running.")
        return 0

    all_results: List[SkillTestResult] = []

    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] Testing: {entry.policy_id}")
        print(f"  task={entry.task_id}  skill={entry.skill_name}  "
              f"agent={entry.agent_name}  type={entry.policy_type}")

        result = _run_skill(
            entry=entry,
            num_episodes=args.num_episodes,
            artifact_root=args.artifact_dir,
            fps=args.fps,
            render_cameras=args.cameras,
            seed=args.seed,
        )
        all_results.append(result)

        if result.error:
            print(f"  [ERROR] {result.error}")
        else:
            print(f"  Success rate: {result.success_rate*100:.1f}%  "
                  f"Mean steps: {result.mean_steps:.1f}")
        print()

    # Write top-level report
    report = {
        "num_skills_tested": len(all_results),
        "skills": [r.to_dict() for r in all_results],
        "overall_success_rate": (
            sum(r.success_rate for r in all_results) / len(all_results)
            if all_results else 0.0
        ),
    }
    report_path = os.path.join(args.artifact_dir, "eval_report.json")
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"{'='*70}")
    print(f"  Report saved: {report_path}")
    print()
    for r in all_results:
        status = "OK" if not r.error else "ERROR"
        sr = f"{r.success_rate*100:.0f}%"
        print(f"  [{status}] {r.policy_id:<40}  success={sr}  steps={r.mean_steps:.0f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
