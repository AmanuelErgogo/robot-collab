"""RoCoToRLDSConverter — converts RoCo NPZ demos to RLDS TFRecord format.

RLDS (Robot Learning Dataset Standard) is the dataset format consumed by
Octo's fine-tuning pipeline.  Each dataset is a directory of TFRecord files
with a ``dataset_info.json`` describing the episode schema.

Input layout (produced by ``collect_subtask_demos.py``):
    <data_root>/<task>/<SKILL>/
        episode_0000/
            observations.npz   # keys: <robot>_qpos, <robot>_ee_xpos, obj_*_xpos, ...
            actions.npz        # key: ctrl  shape (T, 64)
            metadata.json

Output layout (RLDS / Open-X compatible):
    <output_dir>/
        dataset_info.json
        train/
            data-00000-of-00001.tfrecord
        val/
            data-00000-of-00001.tfrecord

Requires: tensorflow  (``pip install tensorflow``)
Optional: tensorflow-datasets (``pip install tensorflow-datasets``)

If TensorFlow is not available, the converter writes a plain NumPy directory
layout instead, prefixed with a warning.  This fallback is not Octo-compatible
but lets the rest of the pipeline run in CI environments without a GPU stack.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
try:
    import PIL.Image  # noqa: F401
except ImportError:
    pass

import numpy as np

# Optional TF import
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore
    _TF_AVAILABLE = False

try:
    import tensorflow_datasets as tfds
    _TFDS_AVAILABLE = True
except ImportError:
    tfds = None  # type: ignore
    _TFDS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConvertConfig:
    data_root: str          # path to <task>/<SKILL>/ directory
    output_dir: str         # where to write the RLDS dataset
    task_id: str
    skill_id: str
    agent_name: str
    language_instruction: str = ""
    val_fraction: float = 0.1
    image_size: Tuple[int, int] = (256, 256)
    action_dim: int = 14    # how many ctrl dims to keep (leading slice)
    state_dim: int = 14     # how many proprio dims to keep


@dataclass
class ConvertResult:
    output_dir: str
    num_train_episodes: int
    num_val_episodes: int
    total_steps: int
    backend: str            # "tfds_rlds" | "numpy_fallback"


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class RoCoToRLDSConverter:
    """Convert a RoCo NPZ demo directory to an RLDS / Octo-compatible dataset."""

    def __init__(self, cfg: ConvertConfig) -> None:
        self.cfg = cfg

    def convert(self) -> ConvertResult:
        episodes = self._load_episodes()
        if not episodes:
            raise ValueError(
                f"No successful episodes found in '{self.cfg.data_root}'. "
                "Run `crie_bench collect` first."
            )

        np.random.shuffle(episodes)
        n_val = 0 if len(episodes) == 1 else max(1, int(len(episodes) * self.cfg.val_fraction))
        val_eps, train_eps = episodes[:n_val], episodes[n_val:]

        if _TF_AVAILABLE and _TFDS_AVAILABLE:
            return self._write_tfds_rlds(train_eps, val_eps)
        else:
            print(
                "[WARNING] TensorFlow / TensorFlow Datasets not found — writing numpy fallback dataset.\n"
                "  Install both for upstream Octo-compatible TFDS/RLDS format:\n"
                "  pip install tensorflow tensorflow-datasets"
            )
            return self._write_numpy_fallback(train_eps, val_eps)

    # ------------------------------------------------------------------
    # Episode loading
    # ------------------------------------------------------------------

    def _load_episodes(self) -> List[Dict[str, Any]]:
        root = Path(self.cfg.data_root)
        episodes = []
        for ep_dir in sorted(root.iterdir()):
            if not ep_dir.is_dir():
                continue
            obs_path = ep_dir / "observations.npz"
            act_path = ep_dir / "actions.npz"
            meta_path = ep_dir / "metadata.json"
            if not (obs_path.exists() and act_path.exists()):
                continue
            meta = {}
            if meta_path.exists():
                with open(meta_path) as fh:
                    meta = json.load(fh)
            if not meta.get("success", True):
                continue  # skip failed episodes

            obs = dict(np.load(obs_path))
            acts = dict(np.load(act_path))
            imgs_path = ep_dir / "images.npz"
            imgs = dict(np.load(imgs_path)) if imgs_path.exists() else {}
            episodes.append({"obs": obs, "acts": acts, "imgs": imgs, "meta": meta, "path": str(ep_dir)})
        return episodes

    # ------------------------------------------------------------------
    # Language instruction
    # ------------------------------------------------------------------

    def _instruction(self, meta: Dict[str, Any]) -> str:
        if self.cfg.language_instruction:
            return self.cfg.language_instruction
        skill = meta.get("skill_name", self.cfg.skill_id)
        call_args = meta.get("call_args", {})
        if call_args:
            args_str = ", ".join(f"{k}={v}" for k, v in call_args.items())
            return f"{skill}({args_str})"
        return skill

    # ------------------------------------------------------------------
    # State / action extraction
    # ------------------------------------------------------------------

    def _extract_state(self, obs_dict: Dict[str, np.ndarray], step_idx: int) -> np.ndarray:
        """Build a fixed-length proprio vector from one obs dict step."""
        parts = []
        # Prefer joint positions for the active embodiment
        for key in sorted(obs_dict.keys()):
            if "qpos" in key or "ee_xpos" in key or "ee_xquat" in key:
                arr = obs_dict[key]
                if arr.ndim == 2:
                    arr = arr[step_idx]
                parts.append(arr.ravel())

        if parts:
            vec = np.concatenate(parts).astype(np.float32)
        else:
            vec = np.zeros(self.cfg.state_dim, dtype=np.float32)

        # Fixed-length slice / pad
        d = self.cfg.state_dim
        if len(vec) >= d:
            return vec[:d]
        return np.pad(vec, (0, d - len(vec)))

    def _extract_action(self, acts_dict: Dict[str, np.ndarray], step_idx: int) -> np.ndarray:
        ctrl = acts_dict.get("ctrl", np.zeros((1, self.cfg.action_dim), dtype=np.float32))
        if ctrl.ndim == 2:
            ctrl = ctrl[step_idx]
        ctrl = ctrl.astype(np.float32)

        # Derive a scalar grasp signal from eq_active: 1.0 if any weld is active, else 0.0
        grasp = np.float32(0.0)
        eq_idxs = acts_dict.get("eq_active_idxs")
        eq_vals = acts_dict.get("eq_active_vals")
        if eq_idxs is not None and eq_vals is not None:
            row_idxs = eq_idxs[step_idx] if eq_idxs.ndim == 2 else eq_idxs
            row_vals = eq_vals[step_idx] if eq_vals.ndim == 2 else eq_vals
            if any(int(v) > 0 for i, v in zip(row_idxs, row_vals) if int(i) >= 0):
                grasp = np.float32(1.0)

        # Action = [ctrl_values, grasp] with length action_dim
        # Reserve last slot for grasp; fill remaining with relevant ctrl
        d = self.cfg.action_dim
        action = np.zeros(d, dtype=np.float32)
        usable = min(len(ctrl), d - 1)
        action[:usable] = ctrl[:usable]
        action[d - 1] = grasp
        return action

    def _extract_image(self, imgs_dict: Optional[Dict[str, np.ndarray]],
                       step_idx: int, cam: str = "teaser") -> Optional[np.ndarray]:
        """Return (H, W, 3) uint8 image for this step, or None."""
        if imgs_dict is None:
            return None
        arr = imgs_dict.get(f"image_{cam}")
        if arr is None:
            return None
        if arr.ndim == 4:   # (T, H, W, 3)
            return arr[step_idx].astype(np.uint8)
        return arr.astype(np.uint8)

    @property
    def dataset_name(self) -> str:
        return f"roco_{self.cfg.task_id}_{self.cfg.skill_id.lower()}"

    # ------------------------------------------------------------------
    # TFDS / RLDS writer (TensorFlow Datasets)
    # ------------------------------------------------------------------

    def _episode_to_rlds_example(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        obs, acts, imgs, meta = (
            episode["obs"], episode["acts"],
            episode.get("imgs", {}), episode["meta"]
        )
        T = acts["ctrl"].shape[0] if "ctrl" in acts else 1
        instruction = self._instruction(meta)

        steps = []
        for t in range(T):
            img = self._extract_image(imgs, t, cam="teaser")
            if img is None:
                img = np.zeros((1, 1, 3), dtype=np.uint8)
            steps.append({
                "observation": {
                    "image_primary": img,
                    "state": self._extract_state(obs, t),
                },
                "action": self._extract_action(acts, t),
                "language_instruction": instruction,
                "is_first": t == 0,
                "is_last": t == T - 1,
                "is_terminal": t == T - 1,
            })

        return {"steps": steps}

    def _write_tfds_rlds(
        self,
        train_eps: List[Dict[str, Any]],
        val_eps: List[Dict[str, Any]],
    ) -> ConvertResult:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dataset_name = self.dataset_name
        dataset_dir = out / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(str(dataset_dir))

        converter = self
        state_dim = self.cfg.state_dim
        action_dim = self.cfg.action_dim

        def _class_name(name: str) -> str:
            return "".join(part.capitalize() for part in name.split("_") if part)

        class _BaseRoCoDatasetBuilder(tfds.core.GeneratorBasedBuilder):
            VERSION = tfds.core.Version("1.0.0")
            RELEASE_NOTES = {"1.0.0": "Initial release."}

            def _info(self) -> "tfds.core.DatasetInfo":
                return tfds.core.DatasetInfo(
                    builder=self,
                    description=(
                        f"RoCo scripted demos for task={converter.cfg.task_id} "
                        f"skill={converter.cfg.skill_id} agent={converter.cfg.agent_name}"
                    ),
                    features=tfds.features.FeaturesDict({
                        "steps": tfds.features.Dataset({
                            "observation": tfds.features.FeaturesDict({
                                "image_primary": tfds.features.Image(
                                    shape=(None, None, 3),
                                    dtype=np.uint8,
                                    encoding_format="jpeg",
                                ),
                                "state": tfds.features.Tensor(
                                    shape=(state_dim,),
                                    dtype=np.float32,
                                ),
                            }),
                            "action": tfds.features.Tensor(
                                shape=(action_dim,),
                                dtype=np.float32,
                            ),
                            "language_instruction": tfds.features.Text(),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                        }),
                    }),
                    supervised_keys=None,
                    homepage="",
                    citation="",
                )

            def _split_generators(self, dl_manager):
                splits = {}
                if train_eps:
                    splits["train"] = self._generate_examples("train", train_eps)
                if val_eps:
                    splits["val"] = self._generate_examples("val", val_eps)
                return splits

            def _generate_examples(self, split: str, episodes: List[Dict[str, Any]]):
                for idx, episode in enumerate(episodes):
                    yield f"{split}_{idx:06d}", converter._episode_to_rlds_example(episode)

        builder_cls = type(_class_name(dataset_name), (_BaseRoCoDatasetBuilder,), {})
        builder = builder_cls(data_dir=str(out))
        if builder.name != dataset_name:
            raise ValueError(
                f"Internal TFDS builder name mismatch: {builder.name} != {dataset_name}"
            )
        builder.download_and_prepare()

        total_steps = sum(
            ep["acts"].get("ctrl", np.zeros((1,))).shape[0]
            for ep in train_eps + val_eps
        )
        return ConvertResult(
            output_dir=str(out),
            num_train_episodes=len(train_eps),
            num_val_episodes=len(val_eps),
            total_steps=total_steps,
            backend="tfds_rlds",
        )

    def _write_dataset_info(self, out: Path, all_eps: List[Dict[str, Any]]) -> None:
        info = {
            "name": f"roco_{self.cfg.task_id}_{self.cfg.skill_id.lower()}",
            "description": (
                f"RoCo scripted demos for task={self.cfg.task_id} "
                f"skill={self.cfg.skill_id} agent={self.cfg.agent_name}"
            ),
            "features": {
                "steps": {
                    "observation": {
                        "state": {"shape": [self.cfg.state_dim], "dtype": "float32"},
                    },
                    "action": {"shape": [self.cfg.action_dim], "dtype": "float32"},
                    "language_instruction": {"dtype": "string"},
                    "is_first": {"dtype": "bool"},
                    "is_last": {"dtype": "bool"},
                    "is_terminal": {"dtype": "bool"},
                }
            },
            "num_episodes": len(all_eps),
        }
        with open(out / "dataset_info.json", "w") as fh:
            json.dump(info, fh, indent=2)

    # ------------------------------------------------------------------
    # Numpy fallback (no TensorFlow)
    # ------------------------------------------------------------------

    def _write_numpy_fallback(
        self,
        train_eps: List[Dict[str, Any]],
        val_eps: List[Dict[str, Any]],
    ) -> ConvertResult:
        out = Path(self.cfg.output_dir)
        total_steps = 0

        for split, eps in [("train", train_eps), ("val", val_eps)]:
            split_dir = out / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for i, ep in enumerate(eps):
                acts = ep["acts"]
                obs = ep["obs"]
                T = acts["ctrl"].shape[0] if "ctrl" in acts else 1
                states = np.stack([self._extract_state(obs, t) for t in range(T)])
                actions = np.stack([self._extract_action(acts, t) for t in range(T)])
                instruction = self._instruction(ep["meta"])
                ep_path = split_dir / f"episode_{i:04d}"
                ep_path.mkdir(exist_ok=True)
                np.savez_compressed(
                    ep_path / "data.npz",
                    states=states,
                    actions=actions,
                    instruction=np.array([instruction]),
                )
                total_steps += T

        self._write_dataset_info(out, train_eps + val_eps)
        return ConvertResult(
            output_dir=str(out),
            num_train_episodes=len(train_eps),
            num_val_episodes=len(val_eps),
            total_steps=total_steps,
            backend="numpy_fallback",
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def convert_demos(
    data_root: str,
    output_dir: str,
    task_id: str,
    skill_id: str,
    agent_name: str,
    language_instruction: str = "",
    val_fraction: float = 0.1,
    action_dim: int = 14,
    state_dim: int = 14,
) -> ConvertResult:
    cfg = ConvertConfig(
        data_root=data_root,
        output_dir=output_dir,
        task_id=task_id,
        skill_id=skill_id,
        agent_name=agent_name,
        language_instruction=language_instruction,
        val_fraction=val_fraction,
        action_dim=action_dim,
        state_dim=state_dim,
    )
    return RoCoToRLDSConverter(cfg).convert()
