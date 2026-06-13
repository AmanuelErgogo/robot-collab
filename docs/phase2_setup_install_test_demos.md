# Phase 2 Setup, Install, Test, Demo, and Tutorial

This runbook sets up and exercises the Phase 2 expert dataset pipeline:

```text
deterministic variation
  -> typed PUT_OBJECT_IN_CONTAINER fixture
  -> RRT expert execution
  -> pre-action observation/action/native SimAction recording
  -> atomic commit or quarantine
  -> validation, splits, replay, visualization
  -> optional LeRobotDataset v3 export/load
```

Phase 2 uses two Python environments:

- RoCo collection runtime: Python 3.8 with MuJoCo/dm_control. This owns the
  simulator and RRT expert execution.
- LeRobot dataset runtime: Python 3.12+ with LeRobot. This exports and loads
  the committed local records through the public LeRobotDataset API.

Run commands from the repository root unless stated otherwise.

## 1. What Phase 2 Adds

Phase 2 records successful RRT-backed `PackGroceryTask` skill executions as
validated expert demonstrations. It does not train a policy and does not call an
LLM.

The core implementation lives in:

```text
integrations/lerobot_roco/dataset/
scripts/collect_roco_expert_dataset.py
scripts/validate_roco_dataset.py
scripts/create_roco_dataset_splits.py
scripts/replay_roco_dataset_episode.py
scripts/visualize_roco_dataset.py
configs/dataset/
```

The recorder attaches to the existing `RRTSkillExecutor` step boundary:

```text
policy.act(obs, physics)
  -> observer.before_step(obs, sim_action, metadata)
  -> env.step(sim_action)
  -> observer.after_step(result)
```

This preserves the mandatory alignment:

```text
sample[t].observation = observation immediately before action[t]
sample[t].action      = action actually applied after that observation
sample[t].timestamp   = simulator time for that observation
```

The public `action` vector is the LeRobot training target. Each episode also
stores `native_actions.npz`, which preserves the exact `SimAction` payload used
by RoCo replay, including equality/grasp attach and release toggles.

## 2. Environment Options

Use the lightweight path for schema, variation hashing, recorder alignment,
atomic writer, splits, validator, and LeRobot export boundary tests. It does not
need MuJoCo, dm_control, Gymnasium, or LeRobot.

Use the full RoCo path for real collection, replay, and rendered visualization.

Use the LeRobot path after collection to export/load the committed records as a
LeRobotDataset v3 dataset.

## 3. Lightweight Dev Setup

This path works in a normal Python environment from the repository root.

Install test dependencies if needed:

```bash
python -m pip install "pytest==7.4.4" "PyYAML==6.0.1"
```

Run the Phase 2 unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/dataset -q
```

Run the executor hook regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills/test_rrt_skill_executor.py -q
```

Run all available local tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Compile the Phase 2 Python modules and scripts:

```bash
python -m compileall integrations/lerobot_roco/dataset \
  scripts/collect_roco_expert_dataset.py \
  scripts/validate_roco_dataset.py \
  scripts/visualize_roco_dataset.py \
  scripts/replay_roco_dataset_episode.py \
  scripts/create_roco_dataset_splits.py \
  rocobench/skills/executor.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids failures from globally installed
pytest plugins that are unrelated to this repository.

## 4. RoCo Collection Runtime Install

Create or reuse the Python 3.8 simulator environment:

```bash
conda create -n roco python=3.8
conda activate roco
```

Install the simulator and Phase 2 dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install mujoco==2.3.0
python -m pip install dm_control==1.0.8
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On Linux headless machines:

```bash
export MUJOCO_GL=egl
```

Quick import checks:

```bash
python -c "from rocobench.envs.task_pack import PackGroceryTask; print('RoCo OK')"
python -c "from integrations.lerobot_roco.dataset import SkillDataSchema; print('Phase 2 dataset OK')"
```

## 5. LeRobot Dataset Runtime Install

Create the Python 3.12+ dataset environment:

```bash
conda create -n lerobot-roco python=3.12
conda activate lerobot-roco
```

Install the client package with LeRobot support:

```bash
python -m pip install --upgrade pip
python -m pip install -e "integrations/lerobot_roco/client[lerobot,test]"
python -m pip install "PyYAML==6.0.1"
```

Quick import checks:

```bash
python -c "from lerobot.datasets import LeRobotDataset; print('LeRobotDataset OK')"
python -c "from integrations.lerobot_roco.dataset.writer import export_local_dataset_to_lerobot; print('export OK')"
```

Do not install LeRobot into the RoCo Python 3.8 environment.

## 6. Demo: One-Episode Debug Collection

Run in the RoCo collection environment:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_debug.yaml \
  --num-episodes 1 \
  --output-root artifacts/datasets/pack_put_object_debug \
  --master-seed 42
```

Expected local layout:

```text
artifacts/datasets/pack_put_object_debug/
  dataset_manifest.json
  schema.json
  episodes/
  quarantine/
  _staging/
```

A successful expert episode appears under `episodes/episode_000000/`. A failed
planning, execution, or postcondition attempt appears under `quarantine/` with a
structured reason.

Inspect the manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/datasets/pack_put_object_debug")
print(json.dumps(json.loads((root / "dataset_manifest.json").read_text()), indent=2))
PY
```

## 7. Demo: Validate Local Records

Run in either environment after collection:

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports
```

Reports are written to:

```text
artifacts/datasets/pack_put_object_debug/reports/
  validation_report.json
  validation_report.md
```

The validator checks schema/features, finite state/action values, action bounds,
monotonic timestamps, frame indices, metadata completeness, duplicate
variations, split leakage, and camera sanity warnings.

## 8. Demo: Create Variation-Group Splits

Run after at least one committed episode:

```bash
python scripts/create_roco_dataset_splits.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --seed 42
```

This writes:

```text
artifacts/datasets/pack_put_object_debug/splits.json
```

Splits are made by `variation_id`, never by frame. Validate again after
creating splits:

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports
```

## 9. Demo: Visualize First and Last Frames

Run after collection:

```bash
python scripts/visualize_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --output artifacts/datasets/pack_put_object_debug/preview
```

Expected preview files:

```text
artifacts/datasets/pack_put_object_debug/preview/
  front_first.png
  front_last.png
  active_agent_first.png
  active_agent_last.png
```

## 10. Demo: Replay a Recorded Episode

Run in the RoCo collection environment:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/replay_roco_dataset_episode.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --config configs/dataset/pack_put_object_debug.yaml \
  --compare
```

Replay resets from the saved variation and applies the recorded native
`SimAction` sidecar when present. The command reports success, frame count,
termination reason, action source, and optional state drift diagnostics. If the
native sidecar is missing, replay falls back to public action vectors through
the Phase 0 action adapter.

Do not count Phase 2 as passing the replay gate unless this command actually
runs in a simulator-backed environment.

## 11. Demo: Export to LeRobotDataset v3

Run in the Python 3.12+ LeRobot environment after local collection:

```bash
conda activate lerobot-roco
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

Then load through the public LeRobotDataset API:

```bash
python - <<'PY'
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/roco-pack-put-object-debug",
    root="artifacts/datasets/pack_put_object_debug_lerobot",
)
print("length:", len(dataset))
sample = dataset[0]
for key, value in sorted(sample.items()):
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    print(key, shape, dtype)
PY
```

Finally validate with LeRobot loading required:

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports \
  --require-lerobot
```

Do not claim LeRobotDataset compatibility unless the load command above
actually succeeds in the LeRobot environment.

## 12. Tutorial: Collect a Small Debug Dataset End to End

This tutorial runs a small local collection and prepares it for Phase 3.

1. Run local unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/dataset tests/skills/test_rrt_skill_executor.py -q
```

2. Collect three debug attempts in RoCo:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_debug.yaml \
  --num-episodes 3 \
  --output-root artifacts/datasets/pack_put_object_debug \
  --master-seed 42
```

3. Validate committed records:

```bash
python scripts/validate_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --report-dir artifacts/datasets/pack_put_object_debug/reports
```

4. Create splits:

```bash
python scripts/create_roco_dataset_splits.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --seed 42
```

5. Visualize one committed episode:

```bash
python scripts/visualize_roco_dataset.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --output artifacts/datasets/pack_put_object_debug/preview
```

6. Replay one committed episode in RoCo:

```bash
python scripts/replay_roco_dataset_episode.py \
  --dataset-root artifacts/datasets/pack_put_object_debug \
  --episode-id episode_000000 \
  --config configs/dataset/pack_put_object_debug.yaml \
  --compare
```

7. Export and load in LeRobot:

```bash
conda activate lerobot-roco
python - <<'PY'
from integrations.lerobot_roco.dataset.writer import export_local_dataset_to_lerobot
from lerobot.datasets import LeRobotDataset

export_local_dataset_to_lerobot(
    dataset_root="artifacts/datasets/pack_put_object_debug",
    lerobot_root="artifacts/datasets/pack_put_object_debug_lerobot",
    repo_id="local/roco-pack-put-object-debug",
    robot_type="roco_pack_alice",
    use_videos=False,
)
dataset = LeRobotDataset(
    repo_id="local/roco-pack-put-object-debug",
    root="artifacts/datasets/pack_put_object_debug_lerobot",
)
print("length:", len(dataset))
print("keys:", sorted(dataset[0].keys()))
PY
```

## 13. Full Collection Config

Use `configs/dataset/pack_put_object_alice.yaml` for the broader Phase 2
collection:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_alice.yaml \
  --num-episodes 100 \
  --output-root artifacts/datasets/pack_put_object_v1 \
  --master-seed 42
```

Run validation, splits, visualization, replay, export, and LeRobot loading on
`artifacts/datasets/pack_put_object_v1` before using the data for Phase 3.

## 14. Expected Artifacts

Local staged dataset:

```text
artifacts/datasets/<name>/
  dataset_manifest.json
  schema.json
  splits.json
  episodes/<episode_id>/
    frames.npz
    native_actions.npz
    metadata.json
    variation.json
  quarantine/<episode_id>/
    quarantine.json
  reports/
    validation_report.json
    validation_report.md
  preview/
    *_first.png
    *_last.png
```

LeRobot export:

```text
artifacts/datasets/<name>_lerobot/
  meta/
  data/
  videos/
```

The exact LeRobot layout is owned by the installed LeRobotDataset version.

## 15. Troubleshooting

If pytest fails before collecting tests with an unrelated plugin import error,
rerun with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

If MuJoCo rendering fails on a headless machine, set:

```bash
export MUJOCO_GL=egl
```

If collection quarantines every attempt, inspect:

```text
artifacts/datasets/<name>/quarantine/<episode_id>/quarantine.json
```

Common quarantine codes:

- `EXPERT_PLANNING_FAILED`
- `EXPERT_EXECUTION_FAILED`
- `POSTCONDITION_FAILED`
- `INTERRUPTED`

The debug config contains an explicit expert motion-target calibration for the
`bin_front_left` fixture:

```yaml
expert_place_target_overrides:
  bin_front_left: bin_front_right
```

This keeps the requested target and postcondition at `bin_front_left` while
using the legacy RRT placement target that reliably lands the object there. The
collector records the requested and motion targets in episode metadata.

If validation reports `LEROBOT_LOAD_SKIPPED`, you validated the local staged
records rather than a LeRobot export, or you did not pass `--require-lerobot`.

If LeRobot export fails, confirm you are in the Python 3.12+ environment and
that `lerobot` imports successfully:

```bash
python -c "from lerobot.datasets import LeRobotDataset; print('ok')"
```

## 16. Gate Checklist

Phase 2 is ready for Phase 3 only when all applicable checks have actually run:

- local unit tests pass;
- one debug expert episode is committed, or failures are quarantined with clear
  reasons;
- validator has no critical errors on the local records;
- variation-group splits exist and have no leakage;
- replay succeeds in the RoCo simulator environment;
- local records export through the public LeRobotDataset API;
- exported dataset loads with `LeRobotDataset`;
- feature keys, shapes, dtypes, action bounds, cadence, episode count, and frame
  count are recorded in the validation report.
