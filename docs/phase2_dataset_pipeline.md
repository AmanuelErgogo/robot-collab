# Phase 2 Dataset Pipeline

This phase records successful RRT-backed `PUT_OBJECT_IN_CONTAINER` executions
for `PackGroceryTask` into a validated staged expert dataset with replay and
LeRobot export tooling.

For setup, install, test, demo, and tutorial commands, see
[`docs/phase2_setup_install_test_demos.md`](phase2_setup_install_test_demos.md).

The implementation keeps three boundaries separate:

- RoCo/RRT execution: `rocobench.skills.RRTSkillExecutor`
- Dataset staging, validation, splits, replay: `integrations.lerobot_roco.dataset`
- LeRobot writing/loading: optional writer boundary using public
  `LeRobotDataset.create`, `add_frame`, `save_episode`, and `finalize`

The mandatory frame alignment is:

```text
sample[t].observation = observation immediately before action[t]
sample[t].action      = action actually applied after that observation
sample[t].timestamp   = simulator time for that observation
```

## Collection

Run in the Python 3.8 RoCo environment with MuJoCo/dm_control available:

```bash
python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_debug.yaml \
  --num-episodes 1 \
  --output-root artifacts/datasets/pack_put_object_debug \
  --master-seed 42
```

The collector stages each episode under `_staging/`, then atomically commits to
`episodes/` only when the executor succeeds, the requested object postcondition
passes, schema checks pass, and the frame count is within configured bounds.
Failed attempts are written under `quarantine/` with structured codes.

The committed local records can be exported in the Python 3.12+ LeRobot
environment with:

```python
from integrations.lerobot_roco.dataset.writer import export_local_dataset_to_lerobot

export_local_dataset_to_lerobot(
    dataset_root="artifacts/datasets/pack_put_object_debug",
    lerobot_root="artifacts/datasets/pack_put_object_debug_lerobot",
    repo_id="local/roco-pack-put-object-debug",
    robot_type="roco_pack_alice",
    use_videos=True,
)
```

This uses the public LeRobotDataset API and does not write LeRobot Parquet/MP4
internals directly.

## Validation

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports
```

Add `--require-lerobot` in the Python 3.12+ LeRobot environment once the dataset
has been exported into a LeRobotDataset v3 layout. If LeRobot is not installed,
the validator reports that as an explicit load failure instead of treating it as
success.

## Splits

```bash
python scripts/create_roco_dataset_splits.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --seed 42
```

Splits are made by variation ID, never by frame. The validator detects variation
leakage across `train`, `validation`, and `test`.

## Replay

Run in the RoCo simulator environment:

```bash
python scripts/replay_roco_dataset_episode.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --config configs/dataset/pack_put_object_debug.yaml \
  --compare
```

Replay resets from the saved variation, applies recorded public actions through
the Phase 0 action adapter, and reports success and state drift diagnostics.

## Visualization

```bash
python scripts/visualize_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --output artifacts/datasets/pack_put_object_debug/preview
```

This exports first and last frames for each recorded camera.

## Verification Status In This Workspace

Local checks run here:

```bash
conda run --no-capture-output -n roco python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_debug.yaml \
  --num-episodes 1 \
  --output-root artifacts/datasets/pack_put_object_debug \
  --master-seed 42 \
  --overwrite
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports
python scripts/create_roco_dataset_splits.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --seed 42
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The debug collection committed `episode_000000` with 42 frames, validation
reported no errors, and local tests passed. The local record validation warning
`LEROBOT_LOAD_SKIPPED` is expected before exporting to a LeRobotDataset layout.
Simulator replay and LeRobotDataset export/load were not run here, so do not
count those gates as passed until the commands above run in their required
environments.
