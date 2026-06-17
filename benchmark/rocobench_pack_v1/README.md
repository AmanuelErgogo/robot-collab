# RoCoBench-Pack-Skills-v1

`RoCoBench-Pack-Skills-v1.0.0` is a fixed benchmark suite for PackGrocery skill-policy and planner+skill evaluation.

## Tracks

- `skill_policy`: direct policy execution for single-agent `PUT_OBJECT_IN_CONTAINER` tasks. Core methods are `hold`, `random`, `act`, and `rrt`.
- `planner_skill`: typed `SkillPlan` execution for sequential and Phase 7-safe concurrent multi-agent cases. Scores are never mixed with skill-policy scores.

## Entry Point

The EnvHub-shaped entry point is:

```python
from benchmark.rocobench_pack_v1 import make_env

env = make_env(n_envs=1, use_async_envs=False, cfg={"active_agent": "Alice"})
```

The RoCo simulator still runs in the separate Python 3.8 runtime through the Phase 0 bridge. Start that server before bridge-backed evaluations.

## Evaluation

```bash
python scripts/evaluate_rocobench.py \
  --suite rocobench_pack_skills_v1 \
  --track skill_policy \
  --method hold \
  --episodes-per-task 2 \
  --output artifacts/benchmark/hold_debug \
  --overwrite
```

ACT requires `--policy-path` pointing at a LeRobot pretrained model directory. RRT runs through the real PackGrocery simulator and RRT executor when simulator dependencies are installed.

## Release Validation

```bash
python scripts/validate_rocobench_release.py
python scripts/validate_rocobench_release.py --run-bridge-smoke --require-bridge
```

The first command runs deterministic Tier 1 checks. The second additionally requires a live Phase 0 bridge server for reset/step/render.

## Full Demo

```bash
python scripts/demo_phase8_benchmark_release.py --overwrite
```

This starts the bridge, validates it, runs hold and RRT baselines on the same fixed variation, compares them, and writes `artifacts/benchmark/phase8_demo/demo_summary.md`.

## Output Files

Each run writes:

- `run_manifest.json`
- `episodes.csv`
- `metrics.json`
- `report.md`
- per-episode `benchmark_result.json`

Every episode records benchmark, schema, protocol, predicate, variation manifest hash, variation hash, method, track, learned success, fallback success, and overall success.
