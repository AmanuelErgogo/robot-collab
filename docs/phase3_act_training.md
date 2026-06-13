# Phase 3 ACT Training

Phase 3 trains ACT on a Phase 2 LeRobotDataset export and validates that the
resulting checkpoint can be inspected and reloaded independently.  It does not
run closed-loop simulator rollout; that is Phase 4.

For a short runnable path, see
[`docs/phase3_short_tutorial.md`](phase3_short_tutorial.md).

The implementation keeps these boundaries separate:

- `integrations.lerobot_roco.training.config`: immutable debug, overfit, and
  baseline run configs.
- `integrations.lerobot_roco.training.feature_contract`: schema, LeRobot export,
  compatibility lock, and optional runtime-spec preflight.
- `integrations.lerobot_roco.training.compatibility`: lazy LeRobot environment
  inspection and `lerobot-train` command generation.
- `integrations.lerobot_roco.training.launch`: run directory, resolved config,
  preflight, compatibility report, manifest, and training launch orchestration.
- `integrations.lerobot_roco.training.checkpoint`: checkpoint metadata
  inspection and native action-unit postprocessing helpers.
- `integrations.lerobot_roco.training.evaluate_offline`: diagnostic offline
  action metrics.

## Required Dataset Export

Phase 3 requires the Phase 2 local records to be exported with the public
LeRobotDataset API before real training:

```python
from integrations.lerobot_roco.dataset.writer import export_local_dataset_to_lerobot

export_local_dataset_to_lerobot(
    dataset_root="artifacts/datasets/pack_put_object_debug",
    lerobot_root="artifacts/datasets/pack_put_object_debug_lerobot",
    repo_id="local/roco-pack-put-object-debug",
    robot_type="roco_pack_alice",
    use_videos=False,
)
```

## Launch Commands

Debug:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml
```

Tiny controlled overfit:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_overfit.yaml
```

Baseline:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_baseline.yaml
```

Dry-run preflight:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml \
  --dry-run \
  --overwrite
```

## Checkpoint Inspection

```bash
python scripts/inspect_roco_checkpoint.py \
  --checkpoint-dir artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050 \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --output-dir artifacts/training/phase3/act_pack_put_debug/checkpoint_inspection
```

## Offline Diagnostics

Offline action diagnostics consume an NPZ with `predictions` and `targets`:

```bash
python scripts/evaluate_roco_act_offline.py \
  --predictions-npz artifacts/training/phase3/act_pack_put_debug/inference_smoke/predictions_debug.npz \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --output-dir artifacts/training/phase3/act_pack_put_debug/offline_eval
```

Offline metrics are diagnostics only.  Do not report validation loss, MAE, or
overfit loss reduction as manipulation success.

## Current Workspace Status

The Phase 3 scaffolding and unit-testable boundaries are implemented. In this
workspace, the debug LeRobot environment is:

- conda env: `lerobot-roco`
- Python: `3.12.13`
- LeRobot: `0.5.1`
- LeRobot source pin in lock: `pypi:lerobot==0.5.1`
- Torch: `2.10.0`
- Gymnasium: `1.3.0`
- datasets: `4.8.5`

Commands run successfully here:

```bash
conda run --no-capture-output -n lerobot-roco \
  python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml \
  --dry-run \
  --overwrite

conda run --no-capture-output -n lerobot-roco \
  python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml \
  --overwrite

python scripts/inspect_roco_checkpoint.py \
  --checkpoint-dir artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050 \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --output-dir artifacts/training/phase3/act_pack_put_debug/checkpoint_inspection

python scripts/evaluate_roco_act_offline.py \
  --predictions-npz artifacts/training/phase3/act_pack_put_debug/inference_smoke/predictions_debug.npz \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --output-dir artifacts/training/phase3/act_pack_put_debug/offline_eval
```

Observed evidence:

- 50-step debug ACT training completed with return code 0.
- Final checkpoint: `artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050/pretrained_model`.
- Checkpoint inspection passed with required files present and no issues.
- Clean-process checkpoint metadata reload exited with code 0.
- One-sample ACT inference produced finite action output with shape `[1, 8]`.
- One-sample offline diagnostics ran and reported `mse=1.79482615`; this is a smoke diagnostic only and not manipulation success.

Not claimed here:

- 1-5 episode controlled overfit loss reduction.
- Closed-loop simulator rollout.
- Manipulation success from the learned policy.
