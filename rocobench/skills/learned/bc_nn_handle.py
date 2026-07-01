"""BCNNHandle — k-Nearest Neighbor Behavioral Cloning policy handle.

Loads a .npz checkpoint produced by scripts/train_bc_nn.py and serves
action predictions via the LearnedPolicyHandle interface.

At each call to predict_native_chunk():
  1. Extract a flat state vector from the current observation.
  2. Find the k nearest neighbours in the normalised training database.
  3. Return the weighted-average action repeated for chunk_size steps
     (the executor will call us again after each step so the query state
     is always fresh).

This is a real trained model (kNN regression) with no extra dependencies
beyond numpy — the checkpoint stores the full normalised state-action database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .policy_handle import LearnedPolicyHandle, NativeActionChunk

# Hardware body names (must match train_bc_nn.py)
_AGENT_HW = {
    "Chad": "ur5e_suction",
    "Dave": "humanoid",
    "Alice": "ur5e_robotiq",
    "Bob": "panda",
}


class BCNNHandle(LearnedPolicyHandle):
    """k-Nearest Neighbor Behavioral Cloning policy.

    Checkpoint (.npz) fields
    ------------------------
    states_norm : (N, d_state)   normalised training states
    actions     : (N, d_action)  training actions (ctrl values for agent joints)
    state_mean  : (d_state,)     normalisation mean
    state_std   : (d_state,)     normalisation std
    ctrl_idxs   : (d_action,)    which ctrl slots the actions occupy
    k           : scalar         number of nearest neighbours
    hw_name     : str            hardware body name (e.g. "ur5e_suction")
    """

    def __init__(self, checkpoint_path: str, chunk_size: int = 10, k: Optional[int] = None):
        data = np.load(checkpoint_path, allow_pickle=True)
        self._states_norm: np.ndarray = data["states_norm"]           # (N, d)
        self._actions: np.ndarray     = data["actions"]               # (N, d_a)
        self._state_mean: np.ndarray  = data["state_mean"]
        self._state_std: np.ndarray   = data["state_std"]
        self._ctrl_idxs: np.ndarray   = data["ctrl_idxs"].astype(np.int32)
        self._k: int                  = int(k or int(data["k"]))
        self._hw_name: str            = str(data["hw_name"])
        self._task: str               = str(data["task"])
        self._skill: str              = str(data["skill"])
        self._agent: str              = str(data["agent"])
        self.chunk_size: int          = int(chunk_size)

        # Sorted object keys (must match train_bc_nn.py extract_state)
        self._obj_keys: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # LearnedPolicyHandle interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._obj_keys = None

    def predict_native_chunk(
        self,
        observation,
        instruction: Mapping[str, Any],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> NativeActionChunk:
        state = self._obs_to_state(observation)
        action = self._knn_action(state)

        # The action from kNN is in the agent's joint space (d_action dims).
        # The executor expects a full ctrl-space action (n_ctrl dims = len(action_low)).
        n_ctrl = len(action_low)
        full_action = np.zeros(n_ctrl, dtype=np.float32)
        for pos, ctrl_idx in enumerate(self._ctrl_idxs):
            if pos < len(action) and int(ctrl_idx) < n_ctrl:
                full_action[int(ctrl_idx)] = float(action[pos])

        # Clip to bounds
        full_action = np.clip(full_action, action_low, action_high)

        # Repeat for chunk_size steps (executor refreshes each step)
        actions_arr = np.tile(full_action, (self.chunk_size, 1)).astype(np.float32)

        dist = self._query_distances(state)
        confidence = float(np.exp(-dist.mean()))
        metadata = {
            "confidence": confidence,
            "nn_distances": dist.tolist(),
            "backend": "bc_nn",
        }
        return NativeActionChunk(actions=actions_arr, metadata=metadata)

    def health_check(self, spec) -> Dict[str, Any]:
        return {
            "policy_type": spec.policy_type if spec is not None else "bc_nn",
            "chunk_size": self.chunk_size,
            "schema_hash": "",
            "action_representation": "joint_ctrl",
            "backend": "bc_nn",
            "n_training_pairs": len(self._states_norm),
            "k": self._k,
        }

    def unload(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_state(self, obs) -> np.ndarray:
        """Build the same state vector as train_bc_nn.extract_state()."""
        parts = []
        hw = self._hw_name
        agent = getattr(obs, hw, None)
        if agent is not None:
            parts.append(np.array(agent.qpos, dtype=np.float32))
            parts.append(np.array(agent.ee_xpos, dtype=np.float32))
            parts.append(np.array(agent.ee_xquat, dtype=np.float32))

        # All object positions (sorted)
        objects = getattr(obs, "objects", {}) or {}
        if self._obj_keys is None:
            self._obj_keys = sorted(objects.keys())
        for obj_name in self._obj_keys:
            obj_state = objects.get(obj_name)
            if obj_state is not None:
                parts.append(np.array(obj_state.xpos, dtype=np.float32))

        if not parts:
            return np.zeros(self._states_norm.shape[1], dtype=np.float32)

        state = np.concatenate(parts)
        # Pad or trim to match training dimension
        d = self._states_norm.shape[1]
        if len(state) < d:
            state = np.pad(state, (0, d - len(state)))
        elif len(state) > d:
            state = state[:d]
        return state

    def _query_distances(self, state: np.ndarray) -> np.ndarray:
        state_norm = (state - self._state_mean) / self._state_std
        diff = self._states_norm - state_norm[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=1)
        k = min(self._k, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        return dists[idx]

    def _knn_action(self, state: np.ndarray) -> np.ndarray:
        state_norm = (state - self._state_mean) / self._state_std
        diff = self._states_norm - state_norm[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=1)
        k = min(self._k, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        k_dists = dists[idx]
        k_actions = self._actions[idx]   # (k, d_action)

        # Inverse-distance weights (closer → higher weight)
        weights = 1.0 / (k_dists + 1e-6)
        weights /= weights.sum()
        return (weights[:, np.newaxis] * k_actions).sum(axis=0).astype(np.float32)
