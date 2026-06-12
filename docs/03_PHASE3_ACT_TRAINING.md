# Phase 3 — ACT Training and Controlled Overfit
## Validated Dataset → Reproducible Checkpoint

# 1. Objective

Train ACT on the Phase 2 single-skill dataset and prove:
- public dataset loading;
- feature compatibility;
- debug optimization;
- controlled tiny-set overfit;
- checkpoint creation;
- clean-process checkpoint reload;
- finite, correctly shaped one-sample inference.

Closed-loop simulator evaluation is Phase 4.

# 2. Scope and non-goals

Initial policy: ACT, Alice, `PUT_OBJECT_IN_CONTAINER`, Phase 2 cameras/state/action. Do not integrate the planner, run online RRT fallback, add concurrency, claim generalization from loss, tune on test, or add a custom architecture unless the pinned ACT cannot consume the validated schema.

# 3. Files

```text
integrations/lerobot_roco/training/
  config.py compatibility.py feature_contract.py launch.py
  evaluate_offline.py checkpoint.py reproducibility.py report.py
scripts/
  train_roco_act.py inspect_roco_checkpoint.py
  evaluate_roco_act_offline.py run_act_overfit_test.py
configs/training/
  act_pack_put_debug.yaml
  act_pack_put_overfit.yaml
  act_pack_put_baseline.yaml
tests/training/
docs/phase3_act_training.md
```

Prefer a thin wrapper around the pinned `lerobot-train` command/public trainer, not a copied trainer.

# 4. Feature preflight

Implement:

```python
@dataclass(frozen=True)
class PolicyFeatureContract:
    input_features: Mapping[str, FeatureSpec]
    output_features: Mapping[str, FeatureSpec]
    action_representation: str
    fps: float
    dataset_schema_hash: str
```

Before GPU allocation compare:
- dataset image keys/order/shapes/channels;
- state shape/field order;
- action shape/field order;
- dtypes and units;
- action representation;
- camera count;
- FPS;
- statistics/normalization availability;
- Phase 0 environment schema.

Fail loudly on mismatch.

# 5. Configurations

**Debug**
```text
50 steps, batch 2, workers 0, frequent save/eval, no WandB required
```

**Tiny overfit**
- fixed 1–5 episode subset;
- no augmentation;
- fixed seed;
- frequent checkpoints;
- predicted-vs-target plots;
- substantial loss reduction;
- clean reload.

**Baseline**
Start from pinned ACT defaults. Typical initial settings may be batch 8 and 100k steps, but use actual pinned field names and record every deviation. Never treat illustrative values as API truth.

# 6. Processing

Use LeRobot policy/dataset processors and dataset stats. Record:
- normalization mode/stats revision;
- image resize/crop/augmentation;
- action normalization/denormalization;
- chunk size;
- observation history;
- prediction and intended execution horizon;
- temporal ensembling.

Overfit run has augmentation disabled. Baseline augmentation must be conservative and documented.

# 7. Reproducibility

Each run saves:
```text
resolved config, exact command, dataset path/revision
split and schema hashes
LeRobot/PyTorch/CUDA/cuDNN versions
GPU model, seeds, repository commit/dirty flag
stdout/stderr, metrics, checkpoints, start/end
```

Create `run_manifest.json`. Never overwrite completed runs by default.

# 8. Offline evaluation

Generate:
- validation loss;
- per-action-dimension MAE;
- chunk error by horizon;
- gripper accuracy/threshold metric when meaningful;
- native-bound violations after postprocessing;
- action smoothness;
- selected predicted-vs-target plots.

Offline error is diagnostic, not task success.

# 9. Checkpoint validation

Checkpoint must:
1. contain feature/config metadata;
2. reload in a clean process;
3. load processor/normalizer state;
4. accept one dataset sample;
5. produce expected finite action/chunk shape;
6. postprocess to native units;
7. reset internal queues/state.

# 10. Tests

Unit:
- config;
- feature mismatch;
- manifest;
- command generation;
- metric math;
- checkpoint metadata;
- compatibility adapter.

Integration:
- debug train;
- tiny overfit;
- clean reload;
- one-sample inference;
- optional baseline launch smoke.

GPU tests are marked; debug CPU support is preferred.

# 11. Acceptance criteria

- [ ] Debug training completes.
- [ ] Tiny-set loss clearly decreases.
- [ ] Checkpoint reloads in a new process.
- [ ] One-sample output is finite and correct shape.
- [ ] Native-unit postprocessing is tested.
- [ ] Test split is untouched for tuning.
- [ ] Full resolved config/provenance is saved.
- [ ] Offline report exists.

# 12. Copy-ready Codex prompt

```text
Implement Phase 3 according to 03_PHASE3_ACT_TRAINING.md.

Inspect and pin the actual LeRobot training API. Build a thin reproducible launcher, not a duplicate trainer. Implement immutable debug/overfit/baseline configs, PolicyFeatureContract preflight, full run manifests, clean checkpoint inspection/reload, one-sample inference, offline metrics/plots, tests, and docs.

Start with ACT only. Do not connect the simulator/planner or add SmolVLA.

Required evidence:
1. 50-step debug training;
2. 1–5 episode controlled overfit with clear loss reduction;
3. independent checkpoint reload;
4. finite correct-shaped sample prediction;
5. processor/normalizer artifacts;
6. native action-unit postprocessing;
7. offline report.

Do not report offline loss as manipulation success.
```

# 13. Review and verification

**Review**
```text
Find feature-order mismatch, test leakage, missing stats/processors, training-process-only checkpoints, normalized actions mistaken for native units, stale action queues, moving LeRobot dependency, irreproducible configs, augmentation in overfit, and claims of success from loss. Fix and rerun.
```

**Verify**
```text
Run feature preflight, unit tests, debug train, tiny overfit, clean-process reload, one-sample prediction, and offline evaluation. Report dataset revision, split counts, schema hash, loss start/end, action shape/range, checkpoint, versions, and tests. Gate fails if reload or native postprocessing fails.
```
