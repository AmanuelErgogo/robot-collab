# Phase 7 — Multi-Agent Learned Skill Execution
## Scheduling, Reservations, Joint Stepping, and Safe Concurrency

# 1. Objective

Support multiple learned agents and safe optional concurrency.

```text
multi-agent SkillPlan
 -> capability/resource/workspace analysis
 -> schedule: sequential|concurrent|reject
 -> per-agent policy queues
 -> centralized merged action
 -> one simulator step
 -> centralized safety/progress monitoring
 -> per-agent + combined results
```

# 2. Incremental levels

**Level A mandatory:** Alice and Bob may each have learned policies, but execute sequentially under deterministic scheduling.

**Level B:** admit only proven non-conflicting concurrent skills with centralized synchronized stepping and STOP_ALL.

**Level C:** coordinated handoff is outside initial scope.

# 3. Files

```text
rocobench/multi_agent/
  models.py capability.py resources.py workspace.py compatibility.py
  scheduler.py synchronized_executor.py joint_stepper.py
  safety_monitor.py cancellation.py combined_result.py metrics.py
configs/multi_agent/pack_sequential.yaml
configs/multi_agent/pack_concurrent_safe.yaml
tests/multi_agent/
docs/phase7_multi_agent.md
```

# 4. Resource claims

Typed resources:
```text
object:<name>, target:<name>, workspace:<zone>
robot:<agent>, shared_tool:<name>
```

```python
SkillResourceClaim(exclusive, shared, workspace_trajectory_hint)
```

Derive from typed skill/task geometry, never arbitrary text.

Conflicts:
- same object;
- same target;
- overlapping workspace/corridor;
- expected crossing paths;
- shared object manipulation without dedicated skill.

# 5. Workspace

Define conservative zones such as left, right, center, bin_left, bin_right, handoff. Map object, target, robot reach, and expected skill corridor. Default to sequential when uncertain.

# 6. Compatibility matrix

Versioned rules, no permissive wildcard:

```yaml
- skills: [PUT_OBJECT_IN_CONTAINER, PUT_OBJECT_IN_CONTAINER]
  concurrent: conditional
  conditions:
    distinct_objects: true
    distinct_targets: true
    disjoint_workspace_zones: true
```

Record rule ID/reason in schedule artifacts.

# 7. Scheduler

```python
ScheduleDecision(
  mode="sequential"|"concurrent"|"reject",
  ordered_groups=...,
  resource_claims=...,
  rule_id=...,
  reason=...
)
```

Sequential ordering is deterministic using dependencies, occupancy, and configured agent order. Free-form LLM ordering is never trusted without validation.

# 8. Central joint stepper

Concurrent per-agent executors must not independently call `env.step()`.

```text
joint observation
 -> per-agent views
 -> each policy produces action
 -> merge by validated control layout
 -> joint bounds/safety
 -> exactly one environment step
 -> distribute new state
```

Add a versioned joint-action extension to Phase 0 without breaking single-agent clients. Queues are independent but aligned to the same environment step index.

# 9. Timing

Start synchronous. If one policy is slow, both wait for correctness. Async inference/RTC may be a later optimization only after correctness and timing tests.

# 10. Central safety

Check:
- robot-robot distance/contact;
- forbidden contacts;
- joint/action limits;
- reservation violations;
- object ownership conflict;
- unexpected shared-object contact;
- reserved-zone intrusion;
- no-progress/deadlock;
- policy failure.

Severity:
`INFO`, `WARN`, `STOP_ONE`, `STOP_ALL`. Severe default is STOP_ALL with safe holds and queue cancellation.

# 11. Combined result

Keep:
- schedule mode/rule;
- per-agent outcome;
- central outcome;
- fallback-to-sequential recommendation;
- learned/fallback attribution.

# 12. Test ladder

Level A:
- two learned calls sequential;
- deterministic order;
- first success/failure propagation;
- per-agent reset.

Level B fakes:
- disjoint admitted;
- same object/target rejected;
- overlapping zone rejected;
- latency stall;
- nonfinite action;
- STOP_ALL/cancellation;
- action merge index correctness.

Level B simulator:
- hold+hold;
- small disjoint scripted motion;
- safe RRT/scripted trajectories;
- learned concurrency only after harness passes;
- forced collision/contact negatives.

Concurrent mode remains feature-gated until all required tests pass.

# 13. Metrics

```text
schedule modes/admission/fallback
combined and per-agent success
makespan and summed agent time
parallel speedup
collision/near-miss
reservation violations
central stops/cancellations
deadlock/no-progress
```

Compare sequential and concurrent on identical variations.

# 14. Acceptance criteria

- [ ] Per-agent policy routing.
- [ ] Stable sequential scheduler.
- [ ] Structured combined result.
- [ ] Central joint step advances once per tick.
- [ ] Resource/workspace conflicts enforced.
- [ ] STOP_ALL works.
- [ ] Safe concurrent smoke passes.
- [ ] Sequential fallback works.
- [ ] Concurrency feature-gated.
- [ ] Identical-suite sequential/concurrent metrics.
- [ ] No handoff claim.

# 15. Copy-ready Codex prompt

```text
Implement Phase 7 according to 07_PHASE7_MULTI_AGENT_EXECUTION.md in two gates.

Level A: per-agent policies, deterministic sequential scheduling, combined results, failure propagation.

Level B: typed resource claims, conservative workspace zones, versioned compatibility matrix, centralized joint action merge/stepper, central safety monitor, cancellation/STOP_ALL, feature-gated concurrency, sequential fallback, metrics, tests, docs.

Never let two executors call env.step independently.
Never admit concurrency just because object names differ.
Default to sequential on uncertainty.
Do not implement/claim handoff.

Before learned concurrency, prove hold/scripted joint stepping and forced failure behavior.
```

# 16. Review and verification

**Review**
```text
Find double simulator stepping, races, omitted hold controls, permissive resource rules, ignored zones, failed cancellation, stale chunks, hidden sequential fallback, action index/order errors, deadlocks, conflated metrics, and concurrency enabled by default. Fix and rerun the full ladder.
```

**Verify**
```text
Run sequential success/failure, conflict rules, merge equivalence, hold-hold, disjoint scripted motion, same object/target rejection, overlap rejection, nonfinite STOP_ALL, cancellation/queue clear, concurrent→sequential fallback, and makespan comparison. Report schedule rules/resources/zones and all outcomes.
```
