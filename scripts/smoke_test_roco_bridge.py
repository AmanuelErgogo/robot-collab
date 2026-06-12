#!/usr/bin/env python
"""Client-side smoke test for the RoCo bridge."""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
CLIENT_SRC = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "client", "src")
if CLIENT_SRC not in sys.path:
    sys.path.insert(0, CLIENT_SRC)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the RoCo bridge.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--active-agent", default="Alice")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--artifacts-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from lerobot_roco_env.client import RemoteRoCoClient

        client = RemoteRoCoClient(endpoint=args.endpoint)
        diagnostics = {
            "ping": client.ping(),
            "hello": client.hello(),
        }
        spec = client.get_spec()
        diagnostics["spec"] = spec.to_dict()
        print(json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True))
        client.close()
        if args.diagnostics_only:
            return 0

        from lerobot_roco_env import RoCoGymEnv

        env = RoCoGymEnv(endpoint=args.endpoint, active_agent=args.active_agent)
        obs, info = env.reset(seed=0)
        transitions = []
        for step in range(args.steps):
            action = env.action_space.sample() if args.random_actions else env.hold_action()
            obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(
                {
                    "step": step,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "is_success": info["is_success"],
                }
            )
            if terminated or truncated:
                break
        image = env.render()
        result = {
            "reset_info": info,
            "transitions": transitions,
            "render_shape": list(image.shape),
            "render_dtype": str(image.dtype),
            "state_shape": list(np.asarray(obs["agent_pos"]).shape),
            "action_shape": list(env.action_space.shape),
        }
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        if args.artifacts_dir:
            os.makedirs(args.artifacts_dir, exist_ok=True)
            with open(os.path.join(args.artifacts_dir, "env_spec.json"), "w", encoding="utf-8") as f:
                json.dump(spec.to_dict(), f, indent=2, sort_keys=True)
            with open(os.path.join(args.artifacts_dir, "reset_info.json"), "w", encoding="utf-8") as f:
                json.dump(_json_safe(info), f, indent=2, sort_keys=True)
            try:
                from PIL import Image

                Image.fromarray(image).save(os.path.join(args.artifacts_dir, "render_000.png"))
            except Exception as exc:
                print("Could not save PNG render artifact: %s" % exc, file=sys.stderr)
        env.close()
        return 0
    except Exception as exc:
        print("smoke test failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
