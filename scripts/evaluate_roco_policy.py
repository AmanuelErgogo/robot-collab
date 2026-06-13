#!/usr/bin/env python
"""Run a Phase 4 direct ACT evaluation suite."""

import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
CLIENT_SRC = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "client", "src")
if CLIENT_SRC not in sys.path:
    sys.path.insert(0, CLIENT_SRC)

from integrations.lerobot_roco.dataset.manifest import atomic_write_json
from integrations.lerobot_roco.evaluation.artifacts import EpisodeArtifactWriter
from integrations.lerobot_roco.evaluation.config import config_hash, load_evaluation_config
from integrations.lerobot_roco.evaluation.policy_loader import load_lerobot_policy
from integrations.lerobot_roco.evaluation.report import write_report
from integrations.lerobot_roco.evaluation.rollout import run_policy_rollout
from integrations.lerobot_roco.evaluation.variation_suite import build_episode_suite


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct ACT rollout over a configured suite.")
    parser.add_argument("--config", required=True, help="Phase 4 evaluation YAML/JSON config.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = load_evaluation_config(args.config)
    overrides = {}
    if args.endpoint is not None:
        overrides["endpoint"] = args.endpoint
    if args.execution_horizon is not None:
        overrides["execution_horizon"] = int(args.execution_horizon)
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.no_video:
        overrides["record_video"] = False
    if args.overwrite:
        overrides["overwrite"] = True
    if overrides:
        config = config.with_overrides(**overrides)

    plans = build_episode_suite(config)
    if args.limit_episodes is not None:
        plans = plans[: int(args.limit_episodes)]
    os.makedirs(config.output_dir, exist_ok=True)
    atomic_write_json(
        os.path.join(config.output_dir, "suite_manifest.json"),
        {
            "config": config.to_dict(),
            "config_hash": config_hash(config),
            "plans": [plan.to_dict() for plan in plans],
            "planner_connected": False,
            "fallback_connected": False,
        },
    )

    from lerobot_roco_env import RoCoGymEnv

    env = RoCoGymEnv(
        endpoint=config.endpoint,
        active_agent=config.active_agent,
        request_timeout_ms=config.request_timeout_ms,
        max_episode_steps=config.max_steps,
    )
    results = []
    try:
        policy = load_lerobot_policy(config)
        for plan in plans:
            writer = EpisodeArtifactWriter(
                root_dir=os.path.join(config.output_dir, "episodes"),
                episode_name="episode_{:06d}".format(plan.episode_index),
                overwrite=config.overwrite,
            )
            writer.append_event({"event": "EPISODE_PLAN", "plan": plan.to_dict()})
            result = run_policy_rollout(
                env,
                policy,
                config,
                writer,
                seed=plan.seed,
                episode_index=plan.episode_index,
            )
            results.append(result)
    finally:
        env.close()

    metrics = write_report(
        config.output_dir,
        config,
        results,
        extra={"config_hash": config_hash(config), "suite_size": len(plans)},
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

