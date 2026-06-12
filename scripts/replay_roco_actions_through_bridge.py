#!/usr/bin/env python
"""Replay trusted local RoCo SimAction artifacts through the debug bridge path."""

import argparse
import os
import sys
import pickle
from typing import Iterable, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integrations.lerobot_roco.roco_runtime.action_adapter import encode_sim_action


def _placeholder_error(path: str) -> None:
    if path == "path/to/actions.pkl":
        raise FileNotFoundError(
            "path/to/actions.pkl is a placeholder. Pass a real local action file, "
            "or create one with scripts/create_roco_hold_action.py."
        )


def _load_pickle_actions(path: str) -> List[object]:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not load pickle action file because module %r is missing. "
            "Run pickle replay in the RoCo environment, or create a safe .npz "
            "file with: python scripts/create_roco_hold_action.py "
            "--output artifacts/phase0/hold_action.npz" % exc.name
        ) from exc
    if hasattr(obj, "ctrl_idxs") and hasattr(obj, "ctrl_vals"):
        return [obj]
    if isinstance(obj, Iterable):
        actions = list(obj)
        if all(hasattr(action, "ctrl_idxs") and hasattr(action, "ctrl_vals") for action in actions):
            return actions
    raise TypeError("Pickle must contain a SimAction or an iterable of SimAction-like objects.")


def _load_npz_payloads(path: str) -> List[dict]:
    data = np.load(path)
    num_actions = int(np.asarray(data["num_actions"])[0]) if "num_actions" in data else 1
    payloads = []
    keys = ["ctrl_idxs", "ctrl_vals", "qpos_idxs", "qpos_target", "eq_active_idxs", "eq_active_vals"]
    for idx in range(num_actions):
        payload = {}
        for key in keys:
            name = "action_%d_%s" % (idx, key)
            if name in data:
                payload[key] = np.ascontiguousarray(data[name])
        payloads.append(payload)
    return payloads


def _load_action_payloads(path: str) -> List[dict]:
    _placeholder_error(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Action file does not exist: %s. Create a demo file with: "
            "python scripts/create_roco_hold_action.py --output artifacts/phase0/hold_action.npz" % path
        )
    if path.endswith(".npz"):
        return _load_npz_payloads(path)
    actions = _load_pickle_actions(path)
    return [encode_sim_action(action) for action in actions]


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay trusted local SimAction arrays through bridge debug command.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--active-agent", default="Alice")
    parser.add_argument("--action-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        client_src = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "client", "src")
        if client_src not in sys.path:
            sys.path.insert(0, client_src)
        from lerobot_roco_env.client import RemoteRoCoClient

        action_payloads = _load_action_payloads(args.action_file)
        client = RemoteRoCoClient(endpoint=args.endpoint)
        reset = client.reset(seed=args.seed, active_agent=args.active_agent)
        episode_id = reset["episode_id"]
        step_index = int(reset["step_index"])
        for payload in action_payloads:
            transition = client.step_native_action(episode_id, step_index, payload)
            step_index = int(transition["step_index"])
            if transition.get("terminated") or transition.get("truncated"):
                break
        print("replayed %d trusted local SimAction payloads" % len(action_payloads))
        client.close_episode()
        client.close()
        return 0
    except Exception as exc:
        print("replay failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
