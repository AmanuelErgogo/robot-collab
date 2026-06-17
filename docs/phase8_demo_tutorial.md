# Phase 8 Full Demo Tutorial

This tutorial runs the `RoCoBench-Pack-Skills-v1.0.0` release as an end-to-end benchmark demo.

The demo performs:

1. deterministic release validation;
2. live Phase 0 bridge reset/step/render validation;
3. bridge-backed hold baseline evaluation;
4. real simulator-backed RRT baseline evaluation;
5. paired comparison on the same fixed variation;
6. summary artifact generation.

## Runtime Requirements

The full demo uses two runtimes:

- RoCo simulator runtime: Python 3.8 with MuJoCo and `dm_control`.
- LeRobot/Gym client runtime: Python 3.12+ with Gymnasium, pyzmq, msgpack, and the bridge client.

If your environments are named like the repository runbooks:

```bash
conda env list
```

should show:

```text
roco
lerobot-roco
```

The demo auto-detects those names and uses:

```text
conda run -n roco python
conda run -n lerobot-roco python
```

## One-Command Demo

From the repository root:

```bash
python scripts/demo_phase8_benchmark_release.py --overwrite
```

Expected outputs:

```text
artifacts/benchmark/phase8_demo/
  release_validation.json
  release_validation_live_bridge.json
  hold_bridge/
  rrt/
  rrt_vs_hold_compare.json
  demo_summary.json
  demo_summary.md
  logs/
```

## Custom Runtime Commands

If your environments have different names, pass explicit commands:

```bash
python scripts/demo_phase8_benchmark_release.py \
  --roco-python "conda run -n roco python" \
  --client-python "conda run -n lerobot-roco python" \
  --overwrite
```

You can also point at direct interpreter paths:

```bash
python scripts/demo_phase8_benchmark_release.py \
  --roco-python /path/to/roco/bin/python \
  --client-python /path/to/lerobot-roco/bin/python \
  --overwrite
```

## What To Inspect

Start with:

```bash
cat artifacts/benchmark/phase8_demo/demo_summary.md
```

Then inspect the raw benchmark outputs:

```bash
cat artifacts/benchmark/phase8_demo/rrt/metrics.json
cat artifacts/benchmark/phase8_demo/hold_bridge/metrics.json
cat artifacts/benchmark/phase8_demo/rrt_vs_hold_compare.json
```

The RRT run should succeed on the validated fixed Alice variation. The hold baseline should normally reach `MAX_STEPS`. The comparison uses the identical task and variation key:

```text
pack.put.alice::pack-put-alice-000
```

## Manual Demo Commands

Run Tier 1 release validation:

```bash
python scripts/validate_rocobench_release.py \
  --output artifacts/benchmark/manual_demo/release_validation.json
```

Start the bridge server in the RoCo runtime:

```bash
conda run -n roco python scripts/start_roco_bridge.py \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --image-height 64 \
  --image-width 64 \
  --max-episode-steps 4 \
  --headless
```

In another shell, validate the live bridge:

```bash
conda run -n lerobot-roco python scripts/validate_rocobench_release.py \
  --run-bridge-smoke \
  --require-bridge \
  --output artifacts/benchmark/manual_demo/release_validation_live_bridge.json
```

Run the hold baseline:

```bash
conda run -n lerobot-roco python scripts/evaluate_rocobench.py \
  --track skill_policy \
  --method hold \
  --task-id pack.put.alice \
  --episodes-per-task 1 \
  --output artifacts/benchmark/manual_demo/hold_bridge \
  --overwrite
```

Run the RRT baseline:

```bash
conda run -n roco python scripts/evaluate_rocobench.py \
  --track skill_policy \
  --method rrt \
  --task-id pack.put.alice \
  --episodes-per-task 1 \
  --output artifacts/benchmark/manual_demo/rrt \
  --overwrite
```

Compare the paired runs:

```bash
python scripts/compare_rocobench_runs.py \
  artifacts/benchmark/manual_demo/rrt \
  artifacts/benchmark/manual_demo/hold_bridge \
  --output artifacts/benchmark/manual_demo/rrt_vs_hold_compare.json
```

## Reading Results Correctly

Do not mix tracks. This demo uses only the `skill_policy` track.

Do not count fallback as learned success. Every episode row has separate fields for:

- `success`
- `overall_success`
- `learned_success`
- `fallback_success`

Do not compare unpaired runs. The comparison script only uses shared `task_id::variation_id` keys.

## Common Failures

If the bridge check fails, confirm the server command is still running and that the client command uses the LeRobot/Gym runtime.

If RRT times out, keep the validated fixed variation `pack-put-alice-000` first in the manifest and check the RoCo simulator runtime. The benchmark records motion failures as failures; it does not convert executor status into task success.

If `pytest` fails before collecting tests because of third-party pytest plugins, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/benchmark -q
```

