# Phase 7 Multi-Agent Execution

Phase 7 adds a conservative multi-agent execution layer under
`rocobench.multi_agent`. It admits multi-agent skill calls through typed
resource claims, defaults to deterministic sequential execution, and supports a
feature-gated synchronized execution path where one central stepper merges all
agent action fragments into exactly one `env.step()` per joint tick.

This phase does not claim learned multi-agent handoff, live LLM integration, or
completed learned concurrent manipulation. Handoff remains disabled. Learned
concurrency remains gated behind explicit configuration and safety checks.

## Architecture

```text
SkillPlan
  -> ResourceClaimDeriver
  -> CompatibilityMatrix
  -> MultiAgentScheduler
  -> sequential groups or gated concurrent group

sequential group
  -> SequentialMultiAgentExecutor
  -> one active skill call, passive agents WAIT
  -> existing single-agent SkillExecutor contract

concurrent group
  -> SynchronizedExecutor
  -> per-agent action providers
  -> CentralSafetyMonitor
  -> CentralJointStepper.merge(...)
  -> exactly one env.step(...) per joint tick
```

## Files

- `rocobench/multi_agent/models.py`: schedule, resource, action-fragment,
  safety-event, joint-step, and per-agent outcome models.
- `rocobench/multi_agent/resources.py`: exclusive resource claims for robots,
  objects, and targets.
- `rocobench/multi_agent/workspace.py`: conservative PackGrocery workspace zone
  hints.
- `rocobench/multi_agent/compatibility.py`: versioned compatibility matrix for
  safe concurrent admission.
- `rocobench/multi_agent/scheduler.py`: deterministic sequential default and
  gated concurrent admission.
- `rocobench/multi_agent/joint_stepper.py`: central action merge plus one
  simulator step.
- `rocobench/multi_agent/safety_monitor.py`: STOP_ALL checks for nonfinite
  actions, latency, merge failures, contacts, reservation violations, and
  ownership conflicts.
- `rocobench/multi_agent/synchronized_executor.py`: sequential and synchronized
  executor helpers.
- `tests/multi_agent/test_phase7_multi_agent.py`: Level A and gated Level B
  tests.
- `scripts/demo_phase7_real_hold.py`: real PackGrocery simulator demo using
  synchronized hold-position providers.

## Configurations

Production-safe default:

```text
configs/multi_agent/pack_sequential.yaml
```

This keeps concurrency disabled, rejects conflicts, disables handoff, and
requires central stepping if concurrency is ever enabled.

Controlled concurrent smoke:

```text
configs/multi_agent/pack_concurrent_safe.yaml
```

This enables the scheduler's concurrency path for disjoint PackGrocery claims,
but marks learned multi-agent concurrency as feature-gated and not production
default.

## Automated Tests

Fast local Phase 7 checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/multi_agent
```

RoCo Python 3.8 environment:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/multi_agent
```

These tests cover:

- deterministic sequential ordering by agent route;
- same-object and same-target rejection;
- overlapping-workspace rejection;
- disjoint concurrent admission when feature-enabled;
- all-WAIT rejection;
- sequential executor passive-agent WAIT insertion;
- central merge with one `env.step()`;
- duplicate control-index rejection;
- STOP_ALL for nonfinite action fragments;
- STOP_ALL for policy latency;
- STOP_ALL for forbidden contact info.

Commands run successfully in this workspace:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/multi_agent
11 passed

conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/multi_agent
11 passed
```

## Real Simulator Demo

Run the PackGrocery synchronized hold demo:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTHONBREAKPOINT=0 python scripts/demo_phase7_real_hold.py \
  --ticks 3 \
  --output-dir artifacts/multi_agent/phase7_real_hold \
  --overwrite
```

What this demo really does:

- creates an actual `PackGroceryTask`;
- schedules two disjoint `PUT_OBJECT_IN_CONTAINER` calls;
- admits the group through `phase7.pack.put.disjoint`;
- uses real robot joint indices from the simulator;
- publishes hold-position fragments for Alice and Bob;
- merges both fragments with `CentralJointStepper`;
- advances the real simulator once per joint tick.

What this demo does not claim:

- no learned concurrent manipulation policy was run;
- no object-level packing postcondition was attempted;
- no handoff behavior was enabled;
- no live LLM integration was exercised.

Artifacts:

```text
artifacts/multi_agent/phase7_real_hold/
  schedule.json
  result.json
  summary.json
```

Command run successfully in this workspace:

```text
success: true
central_status: SUCCEEDED
schedule_mode: concurrent
rule_id: phase7.pack.put.disjoint
central_steps: 3
summed_agent_steps: 6
parallel_speedup: 2.0
stop_all_events: 0
learned_multi_agent_policy: false
```

The saved `summary.json` also records `independent_env_step_allowed: false` and
`central_joint_stepper_required: true`.

## Safety Boundaries

The scheduler admits concurrency only when all active calls have distinct
exclusive resources and disjoint workspace hints. Unknown skills, unknown zones,
handoff zones, repeated objects, repeated targets, and overlapping corridors
reject or fall back to sequential behavior.

The synchronized executor stops all agents on:

- nonfinite control or qpos targets;
- excessive policy latency;
- action merge conflicts;
- forbidden contact/collision reports;
- reservation violations;
- object ownership conflicts.

On STOP_ALL, the combined result recommends sequential fallback rather than
continuing concurrent execution.
