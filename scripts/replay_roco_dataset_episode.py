#!/usr/bin/env python
"""Replay a committed episode through a live RoCo simulator."""

import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.dataset.config import DatasetCollectionConfig
from integrations.lerobot_roco.dataset.cli import create_env_from_config
from integrations.lerobot_roco.dataset.replay import replay_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--config", default="configs/dataset/pack_put_object_alice.yaml")
    parser.add_argument("--compare", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DatasetCollectionConfig.from_yaml(args.config)
    env, _ = create_env_from_config(config, seed=0)
    result = replay_episode(
        args.dataset_root,
        args.episode_id,
        env,
        active_agent=config.active_agent,
        compare=args.compare,
    )
    print(result.to_dict())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
