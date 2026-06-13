# Phase 3 Short Tutorial

This tutorial runs the Phase 3 ACT debug path from a Phase 2 LeRobot export to
a checked training checkpoint. It does not run closed-loop simulator rollout.

Run these commands from the repository root.

## 1. Use the LeRobot Environment

Phase 3 training runs in the Python 3.12+ LeRobot environment, not the Python
3.8 RoCo simulator environment:

```bash
conda activate lerobot-roco
```

Quick check:

```bash
python - <<'PY'
import torch
from lerobot.datasets import LeRobotDataset

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("LeRobotDataset OK")
PY
```

## 2. Confirm the Phase 2 Export

The debug config expects:

```text
artifacts/datasets/pack_put_object_debug
artifacts/datasets/pack_put_object_debug_lerobot
```

Validate the local dataset and require the exported LeRobotDataset to load:

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports \
  --require-lerobot
```

Expected result: `ok: True` with zero errors.

If the LeRobot export is missing, create it first:

```bash
python - <<'PY'
from integrations.lerobot_roco.dataset.writer import export_local_dataset_to_lerobot

export_local_dataset_to_lerobot(
    dataset_root="artifacts/datasets/pack_put_object_debug",
    lerobot_root="artifacts/datasets/pack_put_object_debug_lerobot",
    repo_id="local/roco-pack-put-object-debug",
    robot_type="roco_pack_alice",
    use_videos=False,
)
print("exported")
PY
```

## 3. Run Preflight

Preflight checks the dataset schema, LeRobot export, compatibility lock, and the
exact `lerobot-train` command without training:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml \
  --dry-run \
  --overwrite
```

Expected result: JSON output with `"status": "dry_run"`.

## 4. Train the Debug Policy

Run the 50-step CPU debug training job:

```bash
python scripts/train_roco_act.py \
  --config configs/training/act_pack_put_debug.yaml \
  --overwrite
```

Expected result: JSON output with `"status": "completed"` and
`"returncode": 0`.

Main outputs:

```text
artifacts/training/phase3/act_pack_put_debug/run_manifest.json
artifacts/training/phase3/act_pack_put_debug/preflight.json
artifacts/training/phase3/act_pack_put_debug/phase3_report.md
artifacts/training/phase3/act_pack_put_debug/logs/
artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050/pretrained_model/
```

## 5. Inspect the Checkpoint

```bash
python scripts/inspect_roco_checkpoint.py \
  --checkpoint-dir artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050 \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --output-dir artifacts/training/phase3/act_pack_put_debug/checkpoint_inspection
```

Expected result: `"ok": true`, required files present, and
`"clean_metadata_reload_exit_code": 0`.

## 6. Read the Report

```bash
sed -n '1,220p' artifacts/training/phase3/act_pack_put_debug/phase3_report.md
```

For this debug run, treat successful training and finite checkpoint reload as
pipeline evidence only. Do not report this as manipulation success; closed-loop
simulator rollout belongs to Phase 4.

