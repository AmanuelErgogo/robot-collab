# Phase 8 — Benchmark Hardening and LeRobot-Native Evaluation
## Versioned Suite, Fixed Variations, Baselines, Release, and CI

# 1. Objective

Release a reproducible `RoCoBench-Pack-Skills-v1` benchmark with fixed tasks/variations, standard metrics, LeRobot-compatible loading/evaluation, baselines, raw results, release validation, CI, and documentation.

# 2. Minimum suite

```text
put_object_in_container / Alice
put_object_in_container / Bob
sequential two-agent packing
safe concurrent two-agent packing only if Phase 7 passed
```

Separate tracks:
- **Skill Policy Track:** benchmark supplies skill/instruction.
- **Planner + Skill Track:** planner must produce typed SkillPlan.

Never mix their scores.

# 3. Files

```text
benchmark/rocobench_pack_v1/
  env.py config.py tasks.py variations.py processors.py metrics.py
  evaluator.py baselines.py report.py version.py README.md
  LICENSE pyproject.toml
benchmark/manifests/
  pack_v1_tasks.json pack_v1_variations.json pack_v1_schema.json
benchmark/cards/
  environment_card.md dataset_card.md model_card_template.md
scripts/
  evaluate_rocobench.py validate_rocobench_release.py
  compare_rocobench_runs.py generate_rocobench_report.py
.github/workflows/rocobench-integration.yml
docs/phase8_benchmark_release.md
```

# 4. Versioning

Use semantic benchmark version, e.g. `RoCoBench-Pack-Skills-v1.0.0`.

Breaking changes:
- observation/action schema;
- physics/robot/control rate;
- task predicate;
- variation manifest;
- episode horizon;
- camera setup;
- instruction semantics.

Every result includes benchmark, schema, protocol, predicate versions and variation hash.

# 5. Tasks and variations

```python
@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    description: str
    active_agents: Tuple[str, ...]
    max_steps: int
    success_predicate: str
    failure_predicates: Tuple[str, ...]
    action_representation: str
    required_cameras: Tuple[str, ...]
    variation_group: str
```

Freeze exact seeds, identities, target slots, poses, distractors, assignments, and concurrency cases. Manifest is immutable and checksummed.

# 6. LeRobot integration

Preferred external package/EnvHub-shaped entry:

```python
make_env(n_envs=1, use_async_envs=False, cfg=None)
```

A separate Python 3.8 server may still be required; document startup honestly.

Use native LeRobot benchmark/plugin modifications only if the pinned version lacks an external mechanism and the maintenance burden is justified.

Environment processor only maps RoCo raw formats to standard keys. Normalization/device/tokenization stay in policy processors.

# 7. Evaluation CLI

```bash
python scripts/evaluate_rocobench.py \
  --suite rocobench_pack_skills_v1 \
  --track skill_policy \
  --policy-path ... \
  --episodes-per-task 20 \
  --output artifacts/benchmark/run_001
```

Baselines:
1. RRT expert;
2. hold/no-op;
3. bounded random;
4. ACT;
5. learned + RRT fallback, reported separately.

Optional after core release:
- SmolVLA;
- planner + learned;
- sequential vs concurrent.

All methods use identical variation manifest/horizons/predicates.

# 8. Metrics

Skill track:
- success + Wilson CI;
- steps/sim time;
- latency;
- failures;
- action violations;
- drops/collisions;
- final error.

Planner track adds:
- plan validity;
- replans;
- fallback;
- learned-attempt success vs overall;
- token/planner latency;
- recovery success.

Multi-agent adds:
- combined/per-agent success;
- makespan;
- parallel speedup;
- resource violations;
- central stops.

Use paired comparisons on identical variations. Save raw `episodes.csv`.

# 9. Statistics

Implement appropriate confidence intervals, paired bootstrap or McNemar for paired binary outcomes, effect sizes, and explicit N. Do not overstate small-sample findings.

# 10. Release validator

Check:
- clean install/import;
- versions/hashes;
- manifests;
- reset/step/render;
- deterministic variations;
- success predicate tests;
- baseline smoke;
- result schemas;
- docs/license;
- no credentials/absolute paths/API-key dependency/private checkpoint refs.

# 11. CI tiers

Tier 1: compile/schema/fakes/package/manifest.  
Tier 2: headless MuJoCo bridge and RRT smoke.  
Tier 3 manual/nightly: learned rollout/full subset/concurrency.

# 12. Cards

Environment card: tasks, robots, spaces, rate, predicates, variations, limits, dependency architecture, citation.  
Dataset card: expert source, episodes, splits, schema, failure filtering, biases, license.  
Model card: data/checkpoint/config/results/fallback/limits/reproducibility.

# 13. SmolVLA extension

Optional after ACT release. Require same schema, multiple cameras, language instructions, sufficient repeated variation data, separate training/results, license review, and pinned official APIs. It is not a core release blocker.

# 14. Acceptance criteria

- [ ] Versioned task and variation manifests.
- [ ] Separate skill/planner tracks.
- [ ] Identical variations for methods.
- [ ] RRT/hold/random/ACT baselines.
- [ ] Raw and aggregate results.
- [ ] LeRobot-compatible entry point.
- [ ] Isolated tested processor.
- [ ] Release validator and CI tiers.
- [ ] Environment/dataset/model cards.
- [ ] No live LLM needed for skill track.
- [ ] Explicit limitations.

# 15. Copy-ready Codex prompt

```text
Implement Phase 8 according to 08_PHASE8_BENCHMARK_RELEASE.md.

Create versioned RoCoBench-Pack-Skills-v1 with immutable task/variation manifests, separate skill-policy and planner tracks, an external LeRobot/EnvHub-shaped package or clean supported plugin, isolated environment processor, evaluation CLI, RRT/hold/random/ACT/learned+fallback baselines, raw and aggregate schemas, statistical reporting, release validator, CI tiers, cards, tests, and docs.

Do not require a live LLM for core skill-policy scoring.
Do not change predicates per method.
Do not compare different variations.
Do not make SmolVLA a blocker.
Pin exact LeRobot revision.
If EnvHub cannot launch the Python 3.8 simulator safely, document the required server rather than hiding it.
```

# 16. Review and verification

**Review**
```text
Find mutable variations, unequal seeds, predicate changes, fallback counted as learned, mixed tracks, missing versions, private paths/credentials, live LLM core dependency, weak statistics, local-layout dependency, hidden normalization in environment processor, docs/command mismatch, and CI that never touches simulator. Fix and rerun release validation.
```

**Verify**
```text
Run release validator, clean client install/import, bridge reset-step-render, variation hash, RRT/hold/random/ACT smokes, result schema, report generation, processor tests, and CI-equivalent tiers 1–2. Print all hashes, variations, raw/aggregate paths, metrics, and blocked Tier 3 checks.
```
