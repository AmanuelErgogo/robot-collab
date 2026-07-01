"""Self-contained Octo fine-tuning on RoCo RLDS demos.

Loads pretrained octo-small, reads RLDS TFRecords, and fine-tunes with the
diffusion action head's built-in loss() method.

Usage:
    python scripts/finetune_octo.py \
        --rlds_dir data/rlds/sandwich_pick_chad \
        --out_dir checkpoints/octo/sandwich_pick_chad_v1 \
        --steps 500 \
        --batch_size 2
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import traverse_util

from octo.model.octo_model import OctoModel

# ── constants ────────────────────────────────────────────────────────────────
IMAGE_HW       = 256           # octo-small was trained on 256×256
LANGUAGE_INSTR = "pick up the ingredient"
PRETRAINED     = "hf://rail-berkeley/octo-small"
WINDOW_SIZE    = 2
ACTION_HORIZON = 4
ACTION_DIM     = 7


# ────────────────────────────────────────────────────────────────────────────
# Dataset — TFRecord → fixed-window batches
# ────────────────────────────────────────────────────────────────────────────

def _parse_sequence_example(serialized: tf.Tensor, action_dim: int, state_dim: int):
    _, seq = tf.io.parse_single_sequence_example(
        serialized,
        context_features={},
        sequence_features={
            "steps/observation/image_primary":
                tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/action":
                tf.io.FixedLenSequenceFeature([action_dim], tf.float32),
        },
    )
    return seq


def _decode_images(seq: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    def decode_one(jpg):
        img = tf.image.decode_jpeg(jpg, channels=3)
        img = tf.image.resize(img, [IMAGE_HW, IMAGE_HW], method="bilinear")
        return tf.cast(img, tf.uint8)
    seq["steps/observation/image_primary"] = tf.map_fn(
        decode_one, seq["steps/observation/image_primary"],
        fn_output_signature=tf.uint8,
    )
    return seq


def _window_episode(seq: Dict[str, tf.Tensor], window_size: int) -> Dict[str, tf.Tensor]:
    """Pad/trim every sequence to exactly window_size steps."""
    out = {}
    for k, v in seq.items():
        T   = tf.shape(v)[0]
        pad = tf.maximum(window_size - T, 0)
        rank = len(v.shape)
        pads = [[pad, 0]] + [[0, 0]] * (rank - 1)
        padded = tf.pad(v, pads)
        out[k] = padded[-window_size:]
    return out


def load_dataset(rlds_dir: str, split: str, action_dim: int, state_dim: int,
                 batch_size: int, window_size: int = WINDOW_SIZE) -> tf.data.Dataset:
    record_path = str(Path(rlds_dir) / split / "data-00000-of-00001.tfrecord")
    parse_fn  = functools.partial(_parse_sequence_example,
                                  action_dim=action_dim, state_dim=state_dim)
    window_fn = functools.partial(_window_episode, window_size=window_size)
    return (
        tf.data.TFRecordDataset(record_path)
        .map(parse_fn,       num_parallel_calls=tf.data.AUTOTUNE)
        .map(_decode_images, num_parallel_calls=tf.data.AUTOTUNE)
        .map(window_fn,      num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat()
        .shuffle(200)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_octo_batch(tf_batch: Dict[str, tf.Tensor],
                     batch_size: int,
                     window_size: int,
                     action_horizon: int,
                     action_dim: int,
                     task_jax: Any) -> Tuple[Any, Any, Any, Any]:
    """Build (observation, task, actions, action_pad_mask) as numpy/jax arrays."""
    images  = tf_batch["steps/observation/image_primary"].numpy()  # (B, W, H, H, 3)
    actions = tf_batch["steps/action"].numpy()                     # (B, W, action_dim)

    # Octo action shape: (B, W, action_horizon, action_dim)
    # Tile the last-step action across the horizon
    last_act = actions[:, -1:, :]                                  # (B, 1, action_dim)
    actions_oh = np.tile(last_act[:, :, np.newaxis, :],
                         (1, window_size, action_horizon, 1))      # (B, W, H, D)

    pad_mask    = np.ones((batch_size, window_size), dtype=bool)
    act_pad_mask = np.ones((batch_size, window_size, action_horizon, action_dim), dtype=bool)

    observation = {
        "image_primary":     images.astype(np.uint8),
        "timestep_pad_mask": pad_mask,
    }
    return observation, task_jax, actions_oh, act_pad_mask


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def make_train_step(model: OctoModel, optimizer):
    """Return a JIT-compiled single-step training function."""

    @jax.jit
    def train_step(params, opt_state, obs_jax, task_jax,
                   actions_jax, action_pad_mask_jax, rng):
        def loss_fn(params):
            # 1) Run transformer
            transformer_outputs = model.module.bind(
                {"params": params}, rngs={"dropout": rng}
            ).octo_transformer(
                obs_jax, task_jax, obs_jax["timestep_pad_mask"], train=True
            )
            # 2) Compute diffusion loss via the head
            loss, metrics = model.module.bind(
                {"params": params}, rngs={"dropout": rng}
            ).heads["action"].loss(
                transformer_outputs,
                actions_jax,
                obs_jax["timestep_pad_mask"],
                action_pad_mask_jax,
                train=True,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


def finetune(
    rlds_dir: str,
    out_dir: str,
    steps: int = 500,
    batch_size: int = 2,
    lr: float = 3e-5,
    action_dim: int = ACTION_DIM,
    state_dim: int = 44,
    window_size: int = WINDOW_SIZE,
    save_every: int = 100,
    log_every: int = 10,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained octo-small ...")
    model = OctoModel.load_pretrained(PRETRAINED)
    print(f"  action_dim={ACTION_DIM}  action_horizon={ACTION_HORIZON}")

    print("Loading dataset ...")
    train_ds   = load_dataset(rlds_dir, "train", action_dim, state_dim, batch_size, window_size)
    train_iter = iter(train_ds)

    print("Pre-computing language task tokens ...")
    task = model.create_tasks(texts=[LANGUAGE_INSTR] * batch_size)
    task_jax = jax.tree.map(jnp.array, task)

    print("Setting up optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    params    = model.params
    opt_state = optimizer.init(params)

    train_step = make_train_step(model, optimizer)

    print(f"\nStarting fine-tuning: {steps} steps  batch={batch_size}  lr={lr}")
    losses = []
    rng    = jax.random.PRNGKey(0)

    for step in range(1, steps + 1):
        tf_batch = next(train_iter)
        obs, task_jax_step, acts, act_mask = build_octo_batch(
            tf_batch, batch_size, window_size, ACTION_HORIZON, action_dim, task_jax)

        obs_jax  = jax.tree.map(jnp.array, obs)
        acts_jax = jnp.array(acts, dtype=jnp.float32)
        mask_jax = jnp.array(act_mask, dtype=bool)

        rng, step_rng = jax.random.split(rng)
        try:
            params, opt_state, loss = train_step(
                params, opt_state, obs_jax, task_jax_step,
                acts_jax, mask_jax, step_rng,
            )
            loss_val = float(loss)
        except Exception as exc:
            print(f"  Step {step}: FAILED — {exc}")
            import traceback; traceback.print_exc()
            raise

        losses.append(loss_val)
        if step % log_every == 0:
            print(f"  step={step:5d}  loss={float(np.mean(losses[-log_every:])):.6f}")

        if step % save_every == 0 or step == steps:
            ckpt_dir = out_path / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            flat = traverse_util.flatten_dict(params, sep="/")
            np.savez_compressed(
                str(ckpt_dir / "params.npz"),
                **{k.replace("/", "__"): np.array(v) for k, v in flat.items()},
            )
            print(f"  Saved checkpoint: {ckpt_dir}")

    meta = {
        "pretrained_path":      PRETRAINED,
        "rlds_dir":             rlds_dir,
        "steps":                steps,
        "batch_size":           batch_size,
        "lr":                   lr,
        "action_dim":           action_dim,
        "state_dim":            state_dim,
        "window_size":          window_size,
        "action_horizon":       ACTION_HORIZON,
        "language_instruction": LANGUAGE_INSTR,
        "final_loss":           float(np.mean(losses[-log_every:])),
        "policy_type":          "octo",
    }
    (out_path / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Final loss: {meta['final_loss']:.6f}")
    print(f"Checkpoint dir: {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir",   default="data/rlds/sandwich_pick_chad")
    parser.add_argument("--out_dir",    default="checkpoints/octo/sandwich_pick_chad_v1")
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--action_dim", type=int,   default=ACTION_DIM)
    parser.add_argument("--state_dim",  type=int,   default=44)
    parser.add_argument("--log_every",  type=int,   default=10)
    parser.add_argument("--save_every", type=int,   default=100)
    args = parser.parse_args()

    finetune(
        rlds_dir   = args.rlds_dir,
        out_dir    = args.out_dir,
        steps      = args.steps,
        batch_size = args.batch_size,
        lr         = args.lr,
        action_dim = args.action_dim,
        state_dim  = args.state_dim,
        log_every  = args.log_every,
        save_every = args.save_every,
    )
