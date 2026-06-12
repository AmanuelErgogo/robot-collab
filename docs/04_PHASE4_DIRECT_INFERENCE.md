# Phase 4 — Direct Closed-Loop ACT Inference
## Checkpoint → RoCoGymEnv → Task Predicate

# 1. Objective

Run the Phase 3 ACT checkpoint directly against the Phase 0 Gym/RPC environment, with no high-level planner and no production SkillExecutor.

```text
raw RoCo observation
 -> environment/policy processors
 -> ACT select_action
 -> explicit action queue/horizon
 -> native action postprocessing
 -> env.step
 -> task predicate and failure monitors
```

# 2. Files

```text
integrations/lerobot_roco/evaluation/
  config.py policy_loader.py processor_adapter.py action_queue.py
  rollout.py metrics.py artifacts.py variation_suite.py report.py
scripts/
  rollout_roco_policy.py evaluate_roco_policy.py
  compare_roco_act_vs_rrt.py inspect_roco_rollout.py
configs/evaluation/
  act_pack_put_debug.yaml
  act_pack_put_validation.yaml
  act_pack_put_test.yaml
tests/evaluation/
docs/phase4_direct_inference.md
```

# 3. Non-goals

No planner, production executor, concurrency, default async inference/RTC, hidden RRT fallback, success-predicate changes, or test-set tuning.

# 4. Result model

```python
@dataclass
class PolicyRolloutResult:
    success: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    num_env_steps: int
    sim_time: float
    inference_latency_ms: Sequence[float]
    action_bound_violations: int
    no_progress_events: int
    final_info: Mapping[str, Any]
    artifact_dir: str
```

# 5. Processor contract

At runtime validate:
- raw and processed keys;
- image HWC→tensor layout;
- state shape;
- instruction/task field behavior;
- device/dtype;
- model output shape;
- denormalization to native units;
- bounds and finite values.

Environment processors only map RoCo formats. Normalization, batching, device, and tokenization remain policy processors.

# 6. ACT queue semantics

Record checkpoint chunk size. Implement configurable `execution_horizon <= chunk_size`:
- clear queue on reset/termination/error;
- record chunk ID and offset for every executed action;
- predict again when horizon consumed;
- do not accidentally execute the whole chunk;
- test horizons 1, 5, and checkpoint default where valid.

Temporal ensembling is used only if the pinned implementation supports it and it is explicit.

# 7. Rollout loop

Reset policy and environment for every episode. Use the exact pinned policy API. Success only when skill task predicate confirms requested object is released, stable, and in requested target.

Termination reasons:
```text
SUCCESS, MAX_STEPS, NONFINITE_ACTION, ACTION_OUT_OF_BOUNDS
POLICY_ERROR, BRIDGE_ERROR, COLLISION_LIMIT, NO_PROGRESS
OBJECT_LOST, UNSAFE_STATE, MANUAL_INTERRUPT
```

No-progress monitor must be conservative and use a window of state/action/task progress; a brief pause is not failure.

# 8. Suites

- Debug: known train-like variations.
- Validation: held-out validation variations; tune horizon/monitors here.
- Test: frozen variations; no changes after viewing.

Metrics:
- success and Wilson CI;
- steps/sim time;
- inference mean/p50/p95;
- bound violations;
- termination distribution;
- drop/slip rate;
- final object-target error.

Compare against RRT expert and hold/no-op; bounded random optional.

# 9. Artifacts

Per episode:
```text
episode_config.json rollout_manifest.json actions.npz states.npz
policy_chunk_trace.jsonl events.jsonl video.mp4 final_state.json result.json
```

Aggregate:
`metrics.json`, `episodes.csv`, `termination_reasons.csv`, `latency.csv`, `report.md`.

# 10. Tests

Unit:
- queue/horizon/reset;
- processor mapping;
- denormalization/bounds;
- termination/no-progress;
- metric confidence intervals;
- artifact alignment.

Integration:
- checkpoint load;
- one rollout;
- known and held-out variations;
- max-step case;
- forced invalid action;
- two episodes proving policy reset;
- validation suite smoke.

# 11. Acceptance criteria

- [ ] Direct rollout completes end to end.
- [ ] Native action units are proven.
- [ ] Queue semantics are tested/traced.
- [ ] Task predicate determines success.
- [ ] Timeouts/failures are structured.
- [ ] Videos/traces are aligned.
- [ ] Validation/test discipline is maintained.
- [ ] RRT comparison exists.
- [ ] Planner remains disconnected.

Low learned success does not invalidate engineering correctness, but blocks production replacement.

# 12. Copy-ready Codex prompt

```text
Implement Phase 4 according to 04_PHASE4_DIRECT_INFERENCE.md.

Reuse the pinned checkpoint and Phase 0 RoCoGymEnv. Implement version-isolated policy loading, exact processor pipeline, explicit ACT action queue/execution horizon, synchronous rollout, conservative monitors, structured termination, artifacts, validation/frozen-test suites, metrics, RRT comparison, tests, and docs.

Do not use the planner, production SkillExecutor, async inference, RTC, or fallback.

Prove model outputs are denormalized to native joint/gripper units. Run:
1. known train-like variation;
2. held-out validation variation;
3. forced max-step;
4. two consecutive episodes proving reset;
5. RRT comparison.

Report real success/failure distributions.
```

# 13. Review and verification

**Review**
```text
Find normalized actions sent as native controls, wrong image layout, stale queue, incorrect horizon, success inferred from done, split leakage, no-progress false positives, missing reset, incorrect latency, hidden fallback, and misaligned artifacts. Fix and rerun.
```

**Verify**
```text
Report checkpoint/schema hashes, raw/processed shapes, chunk/horizon, native bounds, seeds/variations, results, latency p50/p95, artifact paths, and RRT comparison. Manually inspect one chunk trace and prove queue clearing. Gate fails if the direct loop or failure semantics are unreliable.
```
