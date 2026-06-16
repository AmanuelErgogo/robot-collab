# Phase 6 Planner Integration

Phase 6 adds a Python 3.8-compatible planner integration layer under
`rocobench.planning`. It connects typed `SkillPlan` parsing and validation to
deterministic backend routing, learned execution, grounded failure feedback,
bounded retries/replans, explicit rollback, and RRT fallback.

## Architecture

```text
current observation
  -> Phase6PromptRenderer
  -> canned or external planner response
  -> SkillResponseParser
  -> SkillPlanValidator
  -> SkillExecutorRouter
  -> SimulatorStateController snapshot
  -> learned or RRT SkillExecutor
  -> current observation summary
  -> RecoveryEngine
  -> retry, replan, rollback, fallback, or abort
```

The implementation keeps the existing boundaries:

- Phase 1 parser, registry, validator, compiler, and RRT executor are reused.
- Phase 5 `LearnedSkillExecutor` remains a backend implementation of the
  `SkillExecutor.execute(plan, obs, artifact_dir=None)` contract.
- `rocobench.planning` imports neither Gymnasium nor LeRobot.
- Automated tests use canned planner responses and fake executors.

## Files

- `executor_router.py`: deterministic route table for `learned`, `rrt`, and
  `unsupported` backends.
- `replanning_policy.py`: failure-code to recovery-action mapping and rollback
  rules.
- `retry_budget.py`: plan, retry, learned-failure, and fallback budgets that do
  not reset between planner rounds.
- `state_summary.py`: measured current-state facts and stable state digests.
- `feedback_renderer.py`: structured grounded feedback with measured facts
  separated from inferred explanation.
- `recovery.py`: failure fingerprinting and recovery decisions.
- `planner_events.py`: event log for routing, execution, rollback, feedback,
  fallback, success, and budget exhaustion.
- `run_controller.py`: sequential Phase 6 controller and canned planner helper.

## Configuration

`configs/planning/pack_learned_sequential.yaml` routes Alice's
`PUT_OBJECT_IN_CONTAINER` skill to the learned backend with explicit RRT
fallback. Bob remains RRT-only for Phase 6. The LLM sees capability classes, not
policy IDs or checkpoint paths.

## Verification

Run the real simulator-backed RRT demo:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario success \
  --output-dir artifacts/planning/phase6_real_pack/success \
  --overwrite
```

Run the no-LLM demo tutorial:

```bash
conda run --no-capture-output -n roco \
  python scripts/demo_phase6_planner.py --scenario all
```

See [`phase6_demo_tutorial.md`](phase6_demo_tutorial.md) for artifact
inspection and optional simulator-backed checks.

Lightweight Phase 6 checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/planning
```

Broader skill regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/planning tests/skills tests/test_legacy_parser_regression.py
```

These tests do not require MuJoCo, LeRobot, or a live LLM. They cover learned
success, invalid-plan replan, missed-grasp retry, slippage replan from current
state, inference-failure rollback and RRT fallback, repeated-fingerprint loop
termination, budget exhaustion, and RRT-only routing.

## Known Provenance Note

The checked-in compatibility lock records LeRobot `0.3.4` and Python `3.10.18`,
while the Phase 3 document notes a later local environment. Phase 6 does not
change the lock; resolve that version provenance before claiming a full
cross-phase LeRobot gate.
