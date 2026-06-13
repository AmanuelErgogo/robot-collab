#!/usr/bin/env python
"""Run one direct Phase 4 ACT rollout through RoCoGymEnv."""

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

from integrations.lerobot_roco.evaluation.artifacts import EpisodeArtifactWriter
from integrations.lerobot_roco.evaluation.config import load_evaluation_config
from integrations.lerobot_roco.evaluation.policy_loader import load_lerobot_policy
from integrations.lerobot_roco.evaluation.rollout import run_policy_rollout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one direct ACT policy rollout.")
    parser.add_argument("--config", required=True, help="Phase 4 evaluation YAML/JSON config.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = load_evaluation_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["seeds"] = (int(args.seed),)
    if args.execution_horizon is not None:
        overrides["execution_horizon"] = int(args.execution_horizon)
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.endpoint is not None:
        overrides["endpoint"] = args.endpoint
    if args.no_video:
        overrides["record_video"] = False
    if args.overwrite:
        overrides["overwrite"] = True
    if overrides:
        config = config.with_overrides(**overrides)

    from lerobot_roco_env import RoCoGymEnv

    env = RoCoGymEnv(
        endpoint=config.endpoint,
        active_agent=config.active_agent,
        request_timeout_ms=config.request_timeout_ms,
        max_episode_steps=config.max_steps,
    )
    try:
        policy = load_lerobot_policy(config)
        writer = EpisodeArtifactWriter(
            root_dir=os.path.join(config.output_dir, "episodes"),
            episode_name="episode_000000",
            overwrite=config.overwrite,
        )
        result = run_policy_rollout(env, policy, config, writer, seed=int(config.seeds[0]), episode_index=0)
    finally:
        env.close()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.termination_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())

