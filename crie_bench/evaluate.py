"""Evaluation pipeline: run a trained policy and report success metrics."""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import TASK_REGISTRY


@dataclass
class EvalConfig:
    task: str
    skill: str
    agent: str
    checkpoint: str
    model: str = "octo"             # "act" | "octo" | "mock"
    uncertainty_mode: str = "heuristic"
    num_episodes: int = 20
    max_steps: int = 300
    chunk_size: int = 10
    render: bool = False
    record_video: bool = True        # export mp4 per episode
    video_fps: int = 20
    video_cameras: tuple = ("teaser", "top_cam")
    artifact_dir: str = "artifacts/eval"
    seed: int = 0


@dataclass
class EpisodeResult:
    episode: int
    success: bool
    num_steps: int
    wall_time: float
    mean_confidence: float
    high_risk_chunks: int
    total_chunks: int


@dataclass
class EvalResult:
    task: str
    skill: str
    agent: str
    model: str
    checkpoint: str
    episodes: List[EpisodeResult] = field(default_factory=list)

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

    def summary(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "skill": self.skill,
            "agent": self.agent,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "num_episodes": len(self.episodes),
            "success_rate": round(self.success_rate, 3),
            "mean_steps": round(self.mean_steps, 1),
            "episodes": [
                {
                    "episode": e.episode,
                    "success": e.success,
                    "num_steps": e.num_steps,
                    "wall_time_s": round(e.wall_time, 2),
                    "mean_confidence": round(e.mean_confidence, 3),
                    "high_risk_chunks": e.high_risk_chunks,
                    "total_chunks": e.total_chunks,
                }
                for e in self.episodes
            ],
        }


def _load_env(task: str, render: bool = False, video_cameras: tuple = ("teaser",)):
    meta = TASK_REGISTRY[task]
    module_path, class_name = meta.class_path.split(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    cameras = list(video_cameras) if video_cameras else ["teaser"]
    return cls(render_cameras=cameras)


def _build_policy_handle(cfg: EvalConfig, spec, env, robots):
    if cfg.model == "mock":
        from rocobench.skills.learned.mock_handle import MockACTHandle
        return MockACTHandle(
            env=env,
            robots=robots,
            chunk_size=cfg.chunk_size,
            uncertainty_mode=cfg.uncertainty_mode,
        )

    if cfg.model == "octo":
        from integrations.octo_roco.handle import OctoHandle
        return OctoHandle.from_checkpoint(cfg.checkpoint, spec)

    if cfg.model == "act":
        from integrations.lerobot_roco.evaluation.policy_loader import load_policy
        policy = load_policy(cfg.checkpoint)
        from integrations.lerobot_roco.evaluation.lerobot_handle import LeRobotACTHandle
        return LeRobotACTHandle(policy, chunk_size=cfg.chunk_size)

    raise ValueError(f"Unknown model '{cfg.model}'")


def evaluate(cfg: EvalConfig) -> EvalResult:
    """Run rollouts of the trained policy and collect success metrics."""
    if cfg.task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{cfg.task}'")

    np.random.seed(cfg.seed)

    from rocobench.skills.models import SkillCall, SkillPlan
    from rocobench.skills.learned.registry import LearnedPolicyRegistry
    from rocobench.skills.learned.models import LearnedPolicySpec
    from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache
    from rocobench.skills.learned.subtask_executor import SubtaskLearnedExecutor
    from rocobench.skills.learned.config import LearnedExecutorConfig

    task_meta = TASK_REGISTRY[cfg.task]
    agent_meta = next((a for a in task_meta.agents if a.name == cfg.agent), None)
    if agent_meta is None:
        raise ValueError(
            f"Agent '{cfg.agent}' not found in task '{cfg.task}'. "
            f"Valid agents: {task_meta.agent_names}"
        )

    env = _load_env(cfg.task, render=cfg.render, video_cameras=cfg.video_cameras)
    robots = env.get_sim_robots()

    spec = LearnedPolicySpec(
        policy_id=f"{cfg.model}_{cfg.task}_{cfg.skill}_{cfg.agent}",
        skill_name=cfg.skill,
        agent_name=cfg.agent,
        embodiment_id=agent_meta.embodiment_id,
        task_id=cfg.task,
        checkpoint=cfg.checkpoint,
        checkpoint_revision="eval",
        policy_type=cfg.model,
        schema_hash="",
        action_representation="joint_ctrl",
        cameras=("wrist_cam", "overhead_cam"),
        max_steps=cfg.max_steps,
        execution_horizon=cfg.chunk_size,
        success_monitor="stable",
        failure_monitors=("TIMEOUT", "NO_PROGRESS", "NONFINITE_ACTION"),
        enabled=True,
    )

    registry = LearnedPolicyRegistry()
    registry.register(spec)

    handle = _build_policy_handle(cfg, spec, env, robots)
    cache = BoundedPolicyHandleCache(loader=lambda s: handle, max_size=4)

    executor_cfg = LearnedExecutorConfig(
        task_id=cfg.task,
        stable_success_checks=2,
    )
    executor = SubtaskLearnedExecutor(
        env=env,
        robots=robots,
        policy_registry=registry,
        policy_cache=cache,
        uncertainty_mode=cfg.uncertainty_mode,
        config=executor_cfg,
    )

    result = EvalResult(
        task=cfg.task,
        skill=cfg.skill,
        agent=cfg.agent,
        model=cfg.model,
        checkpoint=cfg.checkpoint,
    )

    print(f"\n{'='*60}")
    print(f"  CRIE-Bench: evaluation")
    print(f"  task      : {cfg.task}  skill: {cfg.skill}  agent: {cfg.agent}")
    print(f"  model     : {cfg.model}  checkpoint: {cfg.checkpoint}")
    print(f"  episodes  : {cfg.num_episodes}")
    print(f"{'='*60}\n")

    for ep in range(cfg.num_episodes):
        obs = env.reset()
        if hasattr(env, "sample_initial_scene"):
            env.sample_initial_scene()
        obs = env.get_obs()
        # Clear stale frames from prior episodes
        for buf in getattr(env, "render_buffers", {}).values():
            buf.clear()

        skill_call = SkillCall(
            agent_name=cfg.agent,
            skill_name=cfg.skill,
            arguments={"object": _default_object(cfg.task, cfg.skill)},
            raw_action=f"NAME {cfg.agent} ACTION {cfg.skill}",
        )
        plan = SkillPlan(calls=[skill_call], parsed_proposal="scripted", plan_id=f"ep{ep}")

        artifact_ep = os.path.join(cfg.artifact_dir, cfg.task, cfg.skill, f"ep{ep:04d}")
        Path(artifact_ep).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        exec_result = executor.execute(plan, obs, artifact_dir=artifact_ep)
        wall_time = time.perf_counter() - t0

        # Export video
        video_path = ""
        if cfg.record_video:
            video_path = os.path.join(artifact_ep, "execute.mp4")
            bufs = getattr(env, "render_buffers", {})
            n_frames = max((len(b) for b in bufs.values()), default=0)
            if n_frames > 0:
                try:
                    env.export_render_to_video(video_path, out_type="mp4",
                                               fps=cfg.video_fps, concat=True)
                except Exception as exc:
                    print(f"    [WARN] Video export failed: {exc}")
                    video_path = ""

        meta = exec_result.metadata or {}
        unc = meta.get("uncertainty_summary", {})
        ep_result = EpisodeResult(
            episode=ep,
            success=exec_result.success,
            num_steps=exec_result.num_sim_steps,
            wall_time=wall_time,
            mean_confidence=float(unc.get("mean_confidence", 1.0)),
            high_risk_chunks=int(unc.get("high_risk_chunks", 0)),
            total_chunks=int(unc.get("n_chunks", 1)),
        )
        result.episodes.append(ep_result)

        status = "OK" if ep_result.success else "FAIL"
        print(
            f"  ep {ep:03d}  {status}  steps={ep_result.num_steps}  "
            f"conf={ep_result.mean_confidence:.2f}  "
            f"risk={ep_result.high_risk_chunks}/{ep_result.total_chunks}"
        )

    # Write summary JSON
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(cfg.artifact_dir, "eval_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(result.summary(), fh, indent=2)

    sr = result.success_rate
    print(f"\n  Success rate : {sr*100:.1f}%  ({sum(e.success for e in result.episodes)}/{len(result.episodes)})")
    print(f"  Mean steps   : {result.mean_steps:.1f}")
    print(f"  Summary saved: {summary_path}\n")
    return result


def _default_object(task: str, skill: str) -> str:
    """Return a sensible default object argument for the given task/skill."""
    defaults: Dict[str, Dict[str, str]] = {
        "sandwich": {"PICK": "tomato", "STACK_ON": "cheese"},
        "pack":     {"PICK": "apple",  "PLACE": "container", "PICK_AND_PLACE": "apple"},
        "cabinet":  {"PICK": "mug",    "PLACE": "shelf",     "OPEN_CABINET": "door_handle"},
        "sort":     {"PICK": "block_red", "PLACE": "bin_red"},
        "rope":     {"PICK": "rope_end_left", "PLACE": "target"},
        "sweep":    {"SWEEP": "debris"},
    }
    return defaults.get(task, {}).get(skill, "object")
