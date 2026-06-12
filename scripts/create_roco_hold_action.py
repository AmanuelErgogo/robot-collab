#!/usr/bin/env python
"""Create a trusted local hold-action file for bridge replay demos.

Run this in the RoCo Python 3.8 environment. The default output is a safe
NumPy ``.npz`` file containing SimAction arrays, so replay can run in the
client environment without importing rocobench. Pickle output is still
available for trusted local RoCo artifacts, but pickle bytes are never sent over
RPC.
"""

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from integrations.lerobot_roco.roco_runtime.action_adapter import RoCoActionAdapter, encode_sim_action
from integrations.lerobot_roco.roco_runtime.config import RoCoBridgeServerConfig
from integrations.lerobot_roco.roco_runtime.env_factory import create_roco_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a trusted RoCo hold SimAction replay file.")
    parser.add_argument("--output", default="artifacts/phase0/hold_action.npz")
    parser.add_argument("--format", choices=["npz", "pickle"], default=None)
    parser.add_argument("--active-agent", default="Alice", choices=["Alice", "Bob"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    config = RoCoBridgeServerConfig(
        active_agent=args.active_agent,
        seed=args.seed,
        image_height=args.image_height,
        image_width=args.image_width,
        headless=args.headless,
    )
    env = create_roco_env(config)
    env.seed(np_seed=args.seed)
    env.reset(reload=True)

    adapter = RoCoActionAdapter(env, args.active_agent)
    action = adapter.current_active_hold_action()
    sim_action, _ = adapter.to_sim_action(action)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_format = args.format
    if output_format is None:
        output_format = "pickle" if output.endswith(".pkl") else "npz"

    if output_format == "pickle":
        with open(output, "wb") as f:
            pickle.dump([sim_action], f)
        print("wrote trusted hold SimAction pickle: %s" % output)
    else:
        payload = encode_sim_action(sim_action)
        arrays = {"num_actions": np.asarray([1], dtype=np.int32)}
        for key, value in payload.items():
            arrays["action_0_%s" % key] = value
        np.savez(output, **arrays)
        print("wrote trusted hold SimAction array file: %s" % output)
    print("action_dim: %d" % action.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
