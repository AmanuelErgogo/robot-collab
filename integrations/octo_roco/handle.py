"""OctoHandle — LearnedPolicyHandle backed by an Octo VLA checkpoint.

Octo (https://octo-models.github.io) is a JAX-based generalist robot policy.
This handle wraps Octo's inference API to satisfy the RoCo
``LearnedPolicyHandle`` interface used by ``SubtaskLearnedExecutor``.

Install Octo (optional):
    pip install octo           # JAX CPU
    pip install octo[cuda12]   # JAX GPU (CUDA 12)

The handle degrades gracefully: if ``octo`` is not installed it raises an
``ImportError`` with installation instructions at load time only.  The rest of
the RoCo codebase (mock mode, ACT, tests) continues to work unaffected.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rocobench.skills.learned.models import LearnedPolicySpec
from rocobench.skills.learned.policy_handle import LearnedPolicyHandle, NativeActionChunk

# ---------------------------------------------------------------------------
# Optional import
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    from octo.model.octo_model import OctoModel as _OctoModel
    _OCTO_AVAILABLE = True
except ImportError:
    _OctoModel = None  # type: ignore
    _OCTO_AVAILABLE = False

_INSTALL_MSG = (
    "Octo is not installed.  To use OctoHandle, install it:\n"
    "  pip install octo           # CPU / JAX default\n"
    "  pip install octo[cuda12]   # CUDA 12 GPU\n"
    "See https://octo-models.github.io for details."
)

# Default image size expected by Octo-small
_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)

# How many observations Octo uses per prediction (window_size=2 for Octo-small)
_WINDOW_SIZE = 2


class OctoHandle(LearnedPolicyHandle):
    """Wraps an Octo checkpoint for use inside SubtaskLearnedExecutor.

    The handle maintains a rolling observation window of length ``_WINDOW_SIZE``
    and calls ``model.sample_actions()`` to produce an action chunk of shape
    ``(chunk_size, n_ctrl)``.

    Args:
        model: Loaded OctoModel instance.
        spec:  The LearnedPolicySpec for this (skill, agent, task) triple.
        unnorm_key: Dataset statistics key used for action un-normalisation.
            Defaults to the task_id from ``spec``.  Pass ``None`` to skip
            un-normalisation (useful when fine-tuned on a single custom dataset).
        image_size: (H, W) to resize primary camera images before inference.
        rng_seed: JAX random seed.
    """

    def __init__(
        self,
        model: Any,
        spec: LearnedPolicySpec,
        unnorm_key: Optional[str] = None,
        image_size: Tuple[int, int] = _DEFAULT_IMAGE_SIZE,
        rng_seed: int = 0,
    ) -> None:
        if not _OCTO_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        self._model = model
        self._spec = spec
        self._unnorm_key = unnorm_key  # None → no un-normalisation
        self._image_size = image_size
        self._rng = jax.random.PRNGKey(rng_seed)

        # Rolling window buffers; populated in reset() and updated each step
        self._img_window: Optional[np.ndarray] = None   # (W, H, W, 3) uint8
        self._state_window: Optional[np.ndarray] = None  # (W, state_dim)
        self._task = None  # cached task embedding

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        spec: LearnedPolicySpec,
        **kwargs: Any,
    ) -> "OctoHandle":
        """Load an OctoModel from a local checkpoint directory or HF hub ID."""
        if not _OCTO_AVAILABLE:
            raise ImportError(_INSTALL_MSG)
        model = _OctoModel.load_pretrained(checkpoint_path)
        return cls(model=model, spec=spec, **kwargs)

    @classmethod
    def from_pretrained_octo_small(
        cls,
        spec: LearnedPolicySpec,
        **kwargs: Any,
    ) -> "OctoHandle":
        """Load the public Octo-small checkpoint from HuggingFace."""
        return cls.from_checkpoint("hf://rail-berkeley/octo-small", spec, **kwargs)

    @classmethod
    def from_finetuned_npz(
        cls,
        checkpoint_dir: str,
        spec: LearnedPolicySpec,
        pretrained_path: str = "hf://rail-berkeley/octo-small",
        **kwargs: Any,
    ) -> "OctoHandle":
        """Load pretrained Octo then overlay fine-tuned params from a flat .npz file.

        The .npz must have been produced by ``scripts/finetune_octo.py``
        (keys use ``__`` as the path separator instead of ``/``).
        """
        if not _OCTO_AVAILABLE:
            raise ImportError(_INSTALL_MSG)

        import json
        from flax import traverse_util
        from pathlib import Path

        ckpt_dir = Path(checkpoint_dir)

        # Find the latest step checkpoint if a specific step dir isn't given
        if not (ckpt_dir / "params.npz").exists():
            step_dirs = sorted([d for d in ckpt_dir.iterdir()
                                if d.is_dir() and d.name.startswith("step_")])
            if not step_dirs:
                raise FileNotFoundError(
                    f"No step checkpoint found in '{checkpoint_dir}'")
            ckpt_dir = step_dirs[-1]

        meta_path = Path(checkpoint_dir) / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            pretrained_path = meta.get("pretrained_path", pretrained_path)

        print(f"[OctoHandle] Loading pretrained model from: {pretrained_path}")
        base_model = _OctoModel.load_pretrained(pretrained_path)

        print(f"[OctoHandle] Loading fine-tuned params from: {ckpt_dir}/params.npz")
        flat_np = dict(np.load(str(ckpt_dir / "params.npz"), allow_pickle=False))
        # Convert __ separator back to / then unflatten
        flat_params = {k.replace("__", "/"): v for k, v in flat_np.items()}
        nested = traverse_util.unflatten_dict(flat_params, sep="/")
        ft_params = jax.tree.map(jnp.array, nested)

        # Replace params on the frozen struct
        model = base_model.replace(params=ft_params)
        print("[OctoHandle] Fine-tuned params loaded.")
        return cls(model=model, spec=spec, **kwargs)

    # ------------------------------------------------------------------
    # LearnedPolicyHandle interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._img_window = None
        self._state_window = None
        self._task = None
        self._rng, _ = jax.random.split(self._rng)

    def predict_native_chunk(
        self,
        observation,
        instruction: Dict[str, Any],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> NativeActionChunk:
        """Produce one action chunk from the current observation.

        Args:
            observation: ``EnvState`` from the MuJoCo environment.
            instruction:  Dict with at least ``"instruction"`` key (plain text).
            action_low:   Per-joint lower bound, shape ``(n_ctrl,)``.
            action_high:  Per-joint upper bound, shape ``(n_ctrl,)``.

        Returns:
            ``NativeActionChunk`` with ``.actions`` of shape
            ``(chunk_size, n_ctrl)`` clipped to ``[action_low, action_high]``.
        """
        obs_dict = self._build_obs_dict(observation)
        if self._task is None:
            text = instruction.get("instruction", instruction.get("canonical", ""))
            self._task = self._model.create_tasks(texts=[text])

        self._rng, sample_rng = jax.random.split(self._rng)

        # Build unnorm statistics reference
        unnorm_stats = None
        if self._unnorm_key is not None:
            stats = getattr(self._model, "dataset_statistics", {})
            unnorm_stats = stats.get(self._unnorm_key, {}).get("action")

        # sample_actions returns shape (1, pred_horizon, action_dim)
        raw_actions = self._model.sample_actions(
            obs_dict,
            self._task,
            unnormalization_statistics=unnorm_stats,
            rng=sample_rng,
        )  # jax array

        actions_np = np.array(raw_actions[0])  # (pred_horizon, action_dim)
        chunk_size = self._spec.execution_horizon
        actions_np = actions_np[:chunk_size]

        # Clip to joint limits
        n_ctrl = action_low.shape[0]
        if actions_np.shape[-1] >= n_ctrl:
            actions_np = actions_np[:, :n_ctrl]
        else:
            # Pad with zeros if action_dim < n_ctrl (shouldn't happen, but be safe)
            pad = np.zeros((actions_np.shape[0], n_ctrl - actions_np.shape[-1]))
            actions_np = np.concatenate([actions_np, pad], axis=-1)

        actions_np = np.clip(actions_np, action_low, action_high)

        # Confidence from action variance (lower variance = higher confidence)
        confidence = float(np.exp(-np.mean(np.var(actions_np, axis=0))))
        return NativeActionChunk(
            actions=actions_np.astype(np.float32),
            metadata={"confidence": confidence, "source": "octo_sample"},
        )

    def health_check(self, spec: LearnedPolicySpec) -> Dict[str, Any]:
        return {
            "policy_type":   spec.policy_type if spec is not None else "octo",
            "schema_hash":   spec.schema_hash if spec is not None else "",
            "octo_available": _OCTO_AVAILABLE,
            "checkpoint":    getattr(self._spec, "checkpoint", "unknown"),
            "image_size":    list(self._image_size),
            "window_size":   _WINDOW_SIZE,
            "chunk_size":    spec.execution_horizon if spec is not None else 4,
        }

    def unload(self) -> None:
        self._model = None
        self._task = None
        self._img_window = None
        self._state_window = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs_dict(self, env_state) -> Dict[str, Any]:
        """Convert EnvState → Octo observation dict with rolling window."""
        img = self._render_primary(env_state)          # (H, W, 3) uint8
        state = self._extract_proprio(env_state)        # (state_dim,) float32

        H, W = _WINDOW_SIZE, 1

        if self._img_window is None:
            self._img_window   = np.stack([img]   * _WINDOW_SIZE, axis=0)
            self._state_window = np.stack([state] * _WINDOW_SIZE, axis=0)
        else:
            self._img_window   = np.roll(self._img_window,   -1, axis=0)
            self._state_window = np.roll(self._state_window, -1, axis=0)
            self._img_window[-1]   = img
            self._state_window[-1] = state

        # Octo expects batch dim: (batch, window, H, W, C) + timestep_pad_mask
        obs_dict = {
            "image_primary":     self._img_window[None].astype(np.uint8),
            "timestep_pad_mask": np.ones((1, _WINDOW_SIZE), dtype=bool),
        }
        return obs_dict

    def _render_primary(self, env_state) -> np.ndarray:
        """Extract or generate a primary camera image from the env state."""
        # Try to pull a rendered image from the env state's camera dict
        cam_imgs = getattr(env_state, "cam_frames", None) or {}
        for cam_name in ("overhead_cam", "front", "teaser"):
            img = cam_imgs.get(cam_name)
            if img is not None:
                return self._resize_img(np.array(img))

        # Fallback: return a zero image (inference degrades gracefully)
        H, W = self._image_size
        return np.zeros((H, W, 3), dtype=np.uint8)

    def _resize_img(self, img: np.ndarray) -> np.ndarray:
        H, W = self._image_size
        if img.shape[:2] == (H, W):
            return img
        try:
            import cv2
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            return img[:H, :W] if img.shape[0] >= H and img.shape[1] >= W else img

    def _extract_proprio(self, env_state) -> np.ndarray:
        """Extract a flat proprioception vector for the active agent."""
        agent_name = self._spec.embodiment_id
        agent_state = getattr(env_state, agent_name, None)

        parts = []
        if agent_state is not None:
            if hasattr(agent_state, "qpos"):
                parts.append(np.array(agent_state.qpos, dtype=np.float32).ravel())
            if hasattr(agent_state, "ee_xpos"):
                parts.append(np.array(agent_state.ee_xpos, dtype=np.float32).ravel())
            if hasattr(agent_state, "ee_xquat"):
                parts.append(np.array(agent_state.ee_xquat, dtype=np.float32).ravel())

        if parts:
            return np.concatenate(parts)
        return np.zeros(14, dtype=np.float32)
