# Phase 8 Benchmark Release

Phase 8 adds the `RoCoBench-Pack-Skills-v1.0.0` benchmark release.

## Files

- `benchmark/rocobench_pack_v1/`: benchmark package.
- `benchmark/manifests/`: fixed task, variation, and result schema manifests.
- `benchmark/cards/`: environment, dataset, and model-card template.
- `scripts/evaluate_rocobench.py`: benchmark evaluation CLI.
- `scripts/validate_rocobench_release.py`: release validator.
- `scripts/compare_rocobench_runs.py`: paired run comparison.
- `scripts/generate_rocobench_report.py`: regenerate metrics and markdown report.

## Tracks

The benchmark has separate tracks:

- `skill_policy`: direct policy execution for single-agent skills.
- `planner_skill`: typed planner-plus-skill execution for sequential and safe concurrent tasks.

Scores across these tracks are not comparable and are not combined.

## Manifests And Hashes

Tasks are fixed in `benchmark/manifests/pack_v1_tasks.json`. Variations are fixed in `benchmark/manifests/pack_v1_variations.json`.

Every run records:

- benchmark version;
- schema version;
- protocol version;
- predicate version;
- full variation manifest hash;
- per-episode variation hash.

## Baselines

Supported CLI methods:

- `hold`: bridge-backed hold/no-op baseline.
- `random`: bridge-backed bounded random baseline.
- `act`: bridge-backed LeRobot ACT checkpoint evaluation.
- `rrt`: simulator-backed RRT skill baseline for supported sequential tasks.

`learned_rrt_fallback` is represented in the schema and must be reported with separate learned and fallback fields. The CLI requires an injected Phase 5 executor path for a real learned+fallback run and does not fabricate one.

## Commands

Validate deterministic release files:

```bash
python scripts/validate_rocobench_release.py
```

Require a live bridge reset/step/render:

```bash
python scripts/validate_rocobench_release.py --run-bridge-smoke --require-bridge
```

Run a bridge-backed hold baseline:

```bash
python scripts/evaluate_rocobench.py \
  --suite rocobench_pack_skills_v1 \
  --track skill_policy \
  --method hold \
  --episodes-per-task 2 \
  --output artifacts/benchmark/hold_debug \
  --overwrite
```

Run ACT with a checkpoint:

```bash
python scripts/evaluate_rocobench.py \
  --suite rocobench_pack_skills_v1 \
  --track skill_policy \
  --method act \
  --policy-path artifacts/training/phase3/act_pack_put_debug/lerobot_output/checkpoints/000050/pretrained_model \
  --episodes-per-task 2 \
  --output artifacts/benchmark/act_debug \
  --overwrite
```

Compare paired runs:

```bash
python scripts/compare_rocobench_runs.py artifacts/benchmark/hold_debug artifacts/benchmark/act_debug
```

## Full Demo Tutorial

For a one-command walkthrough that starts the bridge, runs hold and RRT baselines on the same fixed variation, compares them, and writes a demo summary, see [docs/build-phases/phase8_demo_tutorial.md](phase8_demo_tutorial.md).

## CI Tiers

- Tier 1: package import, manifest/schema validation, processor contract, fake baseline schema, docs/cards.
- Tier 2: live Phase 0 bridge reset/step/render and RRT smoke when simulator dependencies are available.
- Tier 3: full learned rollout and concurrency subsets, run manually or nightly.

## Limitations

The release is intentionally conservative. The included debug ACT checkpoint validates the pipeline but should not be treated as a strong manipulation model. The simulator-backed checks require the RoCo Python 3.8 runtime and MuJoCo dependencies.
