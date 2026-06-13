# Phase 5 Learned Skill Executor

Phase 5 adds `LearnedSkillExecutor`, a backend implementation of the existing
Phase 1 `SkillExecutor.execute(plan, obs, artifact_dir=None)` interface.

The learned executor consumes the same `SkillPlan` grammar as RRT:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

## Boundaries

- `rocobench/skills/learned` does not import LeRobot or Gymnasium.
- Policy inference is supplied through a typed `LearnedPolicyHandle`.
- The default environment adapter is minimal and intended for tests or typed
  bridge clients. Real deployments should provide an adapter backed by the
  Phase 0 bridge/Phase 4 policy stack.
- RRT remains explicit expert/baseline/fallback. Default fallback mode is
  `return_request`, so the caller decides whether to run RRT.
- Executor reports failures; caller owns rollback/snapshot restore.

## Components

- `models.py`: `LearnedPolicySpec`, lifecycle states, monitor events.
- `registry.py`: exact skill/agent/embodiment/task policy resolution.
- `policy_handle.py`: LeRobot-free policy handle and bounded lazy cache.
- `executor.py`: lifecycle state machine, cancellation, queue reset, execution.
- `success.py`: stable task predicate checker.
- `monitors.py`: action, no-progress, and info-event monitors.
- `fallback.py`: disabled/return-request/automatic fallback decision.
- `artifacts.py`: JSON/JSONL/NPZ artifact writer.

## Artifacts

When `artifact_dir` is supplied, the executor writes:

- `skill_call.json`
- `executor_config.json`
- `policy_spec.json`
- `checkpoint_metadata.json`
- `instruction.json`
- `state_machine.jsonl`
- `monitor_events.jsonl`
- `actions.npz`
- `states.npz`
- `policy_chunks.jsonl`
- `result.json`
- `fallback.json`

Learned and fallback outcomes are separated in `result.json` metadata:
`learned_success`, `fallback_recommended`, `fallback_used`, and
`fallback_success`.

## Verification

Run the learned unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/skills/learned
```

Run the skill regression tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/skills tests/test_legacy_parser_regression.py
```

Live learned rollout still requires wiring a real `LearnedPolicyHandle` and
environment adapter to the Phase 4 client/bridge runtime. Do not claim live
LeRobot/MuJoCo success unless that command is run.

