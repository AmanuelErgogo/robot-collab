"""train_bc_nn.py — k-Nearest Neighbor Behavioral Cloning (pure numpy).

Loads collected subtask demonstrations, builds a state→action lookup database,
normalizes it, and saves a .npz checkpoint that BCNNHandle can load at inference.

Usage
-----
    python scripts/train_bc_nn.py \
        --task sandwich --skill PICK --agent Chad \
        --data-dir data/subtask_demos \
        --output checkpoints/bc_nn/sandwich_pick_chad.npz \
        --k 5

The checkpoint is registered in configs/skills/learned_sandwich_bc.yaml and
picked up by the crie_bench test / test_all_checkpoints pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Feature extraction (must match BCNNHandle._obs_to_state)
# ---------------------------------------------------------------------------

# Hardware body names per agent display name
_AGENT_HW = {
    "Chad": "ur5e_suction",
    "Dave": "humanoid",
    "Alice": "ur5e_robotiq",
    "Bob": "panda",
}

# Chad's ctrl indices in the 64-dim ctrl array
_CTRL_IDXS = {
    "Chad": [1, 2, 3, 4, 5, 6, 0],   # ur5e_suction
    "Dave": [8, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # humanoid
    "Alice": [1, 2, 3, 4, 5, 6, 0],
    "Bob": [7, 8, 9, 10, 11, 12, 13],
}


def extract_state(obs_dict: dict, hw_name: str) -> np.ndarray:
    """Build a flat state vector from one timestep's obs dict."""
    parts = []
    # Robot joints + EE
    for key in (f"{hw_name}_qpos", f"{hw_name}_ee_xpos", f"{hw_name}_ee_xquat"):
        if key in obs_dict:
            parts.append(obs_dict[key].astype(np.float32))
    # All object positions (sorted for determinism)
    obj_keys = sorted(k for k in obs_dict if k.startswith("obj_") and k.endswith("_xpos"))
    for k in obj_keys:
        parts.append(obs_dict[k].astype(np.float32))
    return np.concatenate(parts)


def extract_action(ctrl_arr: np.ndarray, ctrl_idxs: List[int]) -> np.ndarray:
    """Extract the agent's ctrl values from the 64-dim ctrl array."""
    return ctrl_arr[ctrl_idxs].astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(data_root: str, task: str, skill: str, agent: str):
    hw_name = _AGENT_HW.get(agent, agent.lower())
    ctrl_idxs = _CTRL_IDXS.get(agent, list(range(7)))

    skill_dir = Path(data_root) / task / skill.upper()
    if not skill_dir.exists():
        raise FileNotFoundError(f"No demos at {skill_dir}")

    ep_dirs = sorted(skill_dir.glob("episode_*"))
    if not ep_dirs:
        raise FileNotFoundError(f"No episode dirs in {skill_dir}")

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    ep_ids: List[int] = []

    for ep_idx, ep_dir in enumerate(ep_dirs):
        obs_npz = np.load(ep_dir / "observations.npz")
        act_npz = np.load(ep_dir / "actions.npz")
        ctrl = act_npz["ctrl"]        # (T, 64)
        T = ctrl.shape[0]

        obs_dict_t = {}
        for key in obs_npz.files:
            obs_dict_t[key] = obs_npz[key]  # (T, d)

        for t in range(T):
            # Build per-timestep obs dict
            obs_t = {k: obs_dict_t[k][t] for k in obs_dict_t}
            s = extract_state(obs_t, hw_name)
            a = extract_action(ctrl[t], ctrl_idxs)
            states.append(s)
            actions.append(a)
            ep_ids.append(ep_idx)

    states_arr = np.stack(states, axis=0)    # (N, d_state)
    actions_arr = np.stack(actions, axis=0)  # (N, d_action)
    ep_ids_arr = np.array(ep_ids, dtype=np.int32)

    print(f"  Loaded {len(ep_dirs)} episodes → {len(states_arr)} state-action pairs")
    print(f"  state dim : {states_arr.shape[1]}")
    print(f"  action dim: {actions_arr.shape[1]}")
    return states_arr, actions_arr, ep_ids_arr, ctrl_idxs


# ---------------------------------------------------------------------------
# Training (normalise + save)
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*60}")
    print(f"  kNN-BC training")
    print(f"  task   : {args.task}   skill: {args.skill}   agent: {args.agent}")
    print(f"  data   : {args.data_dir}")
    print(f"  output : {args.output}")
    print(f"  k      : {args.k}")
    print(f"{'='*60}\n")

    states, actions, ep_ids, ctrl_idxs = load_dataset(
        args.data_dir, args.task, args.skill, args.agent
    )

    # Normalise states
    state_mean = states.mean(axis=0)
    state_std  = states.std(axis=0) + 1e-6
    states_norm = (states - state_mean) / state_std

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        states_norm=states_norm.astype(np.float32),
        actions=actions.astype(np.float32),
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        ctrl_idxs=np.array(ctrl_idxs, dtype=np.int32),
        ep_ids=ep_ids,
        k=np.int32(args.k),
        task=np.array(args.task),
        skill=np.array(args.skill),
        agent=np.array(args.agent),
        hw_name=np.array(_AGENT_HW.get(args.agent, args.agent.lower())),
    )
    print(f"\nCheckpoint saved: {out_path}")
    print(f"  {len(states)} training pairs, state_dim={states.shape[1]}, "
          f"action_dim={actions.shape[1]}, k={args.k}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task",     required=True)
    p.add_argument("--skill",    required=True)
    p.add_argument("--agent",    required=True)
    p.add_argument("--data-dir", default="data/subtask_demos", dest="data_dir")
    p.add_argument("--output",   default="checkpoints/bc_nn/sandwich_pick_chad.npz")
    p.add_argument("--k",        type=int, default=5, help="Neighbors for kNN. Default: 5.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
