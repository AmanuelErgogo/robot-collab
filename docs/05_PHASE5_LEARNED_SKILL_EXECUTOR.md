# Phase 5 — Production Learned Skill Executor
## Typed SkillPlan → Learned Policy Lifecycle → Structured Result

# 1. Objective

Implement `LearnedSkillExecutor` against the actual Phase 1 `SkillExecutor` contract, reusing the validated Phase 4 rollout stack.

```text
SkillPlan
 -> policy registry resolution
 -> checkpoint/feature validation
 -> policy reset
 -> closed-loop execution
 -> progress/failure monitors
 -> stable postcondition verification
 -> structured SkillExecutionResult
```

RRT remains expert, baseline, and explicit fallback.

# 2. Scope

Initial backend:
- ACT;
- Alice;
- `PUT_OBJECT_IN_CONTAINER`;
- one active learned call; other agents must WAIT.

No planner connection, concurrency, training, hidden retries, hidden fallback, or boolean-only results.

# 3. Files

```text
rocobench/skills/learned/
  models.py registry.py policy_handle.py executor.py monitors.py
  success.py fallback.py artifacts.py errors.py config.py
configs/skills/learned_pack_put.yaml
tests/skills/learned/
docs/phase5_learned_executor.md
```

If runtime separation remains, `LearnedSkillExecutor` uses a typed inference/environment client; it must not import LeRobot into Python 3.8 modules.

# 4. Registry

```python
@dataclass(frozen=True)
class LearnedPolicySpec:
    policy_id: str
    skill_name: str
    agent_name: str
    embodiment_id: str
    task_id: str
    checkpoint: str
    checkpoint_revision: str
    policy_type: str
    schema_hash: str
    action_representation: str
    cameras: Tuple[str, ...]
    max_steps: int
    execution_horizon: int
    success_monitor: str
    failure_monitors: Tuple[str, ...]
    enabled: bool = True
```

Requirements:
- exact agent+skill+embodiment resolution;
- reject ambiguity;
- validate checkpoint metadata/schema;
- lazy bounded cache;
- explicit unload/health check;
- policy revision in every result;
- deterministic config resolution.

# 5. State machine

```text
CREATED -> VALIDATING -> LOADING_POLICY -> RESETTING_POLICY
 -> RUNNING -> VERIFYING -> SUCCEEDED/FAILED/TIMED_OUT/INTERRUPTED
 -> FALLBACK_REQUESTED -> CLOSED
```

Log every transition. Add a cancellation token checked between environment steps. Reset policy and action queue for every skill.

# 6. Context

Deterministically render:
```text
canonical: PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
instruction: Put the apple into the front-left bin slot.
template version: ...
```

ACT may ignore language; record it without claiming language conditioning.

# 7. Success

Use a stable task/skill predicate:
- requested object in requested target;
- object released;
- stable for configurable consecutive checks;
- no prohibited target conflict.

Never infer success from model confidence, episode end, or gripper opening alone.

# 8. Monitors

Modular events:
```text
TIMEOUT, NONFINITE_ACTION, ACTION_OUT_OF_BOUNDS, NO_PROGRESS
MISSED_GRASP, SLIPPAGE, OBJECT_LOST, TARGET_OCCUPIED
COLLISION_RISK, BRIDGE_FAILURE, POLICY_INFERENCE_FAILURE
MANUAL_INTERRUPT
```

Only classify when measurable. Unknown failure stays generic with evidence.

```python
MonitorEvent(code, severity, message, evidence, recommended_recovery)
```

Progress stages:
`not_started`, `approaching`, `grasped`, `transporting`, `over_target`, `released`, `stable`.

# 9. Fallback

Config:
```yaml
fallback:
  mode: disabled | return_request | automatic
  backend: rrt
  allowed_reasons: [POLICY_INFERENCE_FAILURE, NO_PROGRESS]
  max_fallbacks: 1
```

Default: `return_request`. The caller chooses fallback. If automatic fallback exists, report:
- learned attempt success;
- fallback used;
- fallback success;
- overall success.

Never count fallback success as learned success.

# 10. Result

Include:
```text
success, status, reason, failure_code
policy ID/revision, learned_backend
steps, progress stage, evidence
fallback recommended/used/success
artifact path
```

# 11. Rollback

Executor reports; caller owns snapshot/restore. Add tests for partial execution failure followed by caller rollback. Executor must not silently restore.

# 12. Artifacts

```text
skill_call.json executor_config.json policy_spec.json
checkpoint_metadata.json instruction.json state_machine.jsonl
monitor_events.jsonl actions.npz states.npz policy_chunks.jsonl
video.mp4 result.json fallback.json
```

Learned and fallback segments must be distinguishable.

# 13. Tests

Unit:
- registry exact/ambiguous/missing;
- checkpoint mismatch;
- cache;
- state machine;
- cancellation;
- monitor positive/negative cases;
- stable success;
- false-success tests;
- fallback decision;
- serialization.

Integration:
- learned attempt;
- timeout;
- forced inference error;
- no-progress;
- cancellation;
- fallback request;
- explicit RRT fallback;
- caller rollback;
- two consecutive skills proving reset.

# 14. Acceptance criteria

- [ ] Actual generic `SkillExecutor` interface implemented.
- [ ] Exact policy registry resolution.
- [ ] Checkpoint/schema validation.
- [ ] State machine and cancellation.
- [ ] Stable task-predicate success.
- [ ] Structured failure evidence.
- [ ] No hidden fallback.
- [ ] Learned/fallback metrics separated.
- [ ] Policy reset between skills.
- [ ] Caller rollback tested.
- [ ] RRT executor regression passes.

# 15. Copy-ready Codex prompt

```text
Implement Phase 5 according to 05_PHASE5_LEARNED_SKILL_EXECUTOR.md.

Inspect the actual Phase 1 interfaces and reuse Phase 4 policy loading, processors, queue, rollout, and monitors. Do not create a second inference stack.

Implement LearnedPolicySpec/registry, lazy policy handle, LearnedSkillExecutor, explicit lifecycle state machine, cancellation, deterministic instruction context, stable success predicate, modular progress/failure monitors, structured results, explicit fallback recommendation/optional RRT fallback, artifacts, tests, and docs.

Accept only one active learned call with other agents WAIT.
Do not connect the LLM planner.
Do not silently retry, restore, or fall back.
Do not count fallback as learned success.
Keep rollback in the caller.

Verify two consecutive skills clear policy state and queues.
```

# 16. Review and verification

**Review**
```text
Find hidden fallback, false success, stale queues, ambiguous registry, schema mismatch, executor-side hidden restore, unbounded retry, monitor overclassification, missing cancellation, learned/fallback metric conflation, missing checkpoint revision, and multiple active calls accepted. Fix and rerun all failure paths.
```

**Verify**
```text
Run valid attempt, timeout, nonfinite action, no-progress, inference failure, cancellation, fallback recommendation, explicit RRT fallback, caller rollback, and two sequential skills. Report state transitions, monitor evidence, result JSON, policy revision, and learned-vs-fallback outcome.
```
