# Phase 6 — RoCo Planner Integration and Failure-Aware Replanning
## Skill Selection → Learned Execution → Grounded Feedback

# 1. Objective

Connect RoCo’s typed high-level planner to learned execution while preserving deterministic routing, current-state grounding, bounded retry/replan, explicit rollback, and RRT fallback.

```text
observe -> prompt skills/capabilities -> SkillPlan -> validate
 -> route backend -> snapshot -> execute -> reobserve
 -> structured grounded feedback -> recovery/replan/fallback
```

Initial mode is sequential: one learned active skill; other calls WAIT or RRT according to configuration.

# 2. Non-goals

No concurrency, no LLM-chosen low-level controller/checkpoint, no raw stack traces in prompts, no infinite retries, no stale-state feedback, no hidden fallback, and no removal of RRT.

# 3. Files

```text
rocobench/planning/
  executor_router.py replanning_policy.py feedback_renderer.py
  state_summary.py retry_budget.py recovery.py planner_events.py
  run_controller.py
configs/planning/pack_learned_sequential.yaml
tests/planning/
docs/phase6_planner_integration.md
```

# 4. Executor router

```python
class SkillExecutorRouter:
    def resolve(self, call, context) -> ExecutorDecision: ...
```

Decision:
```text
backend learned|rrt|unsupported
policy ID (internal)
reason
fallback backend
```

Config example:
```yaml
routes:
  - agent: Alice
    skill: PUT_OBJECT_IN_CONTAINER
    primary: learned
    fallback: rrt
  - agent: Bob
    skill: PUT_OBJECT_IN_CONTAINER
    primary: rrt
```

The LLM sees capabilities, not checkpoint paths.

# 5. Replanning policy

```python
@dataclass
class ReplanningPolicy:
    max_plan_rounds: int
    max_retries_per_skill: int
    max_learned_failures: int
    allow_same_plan_retry: bool
    fallback_rules: Mapping[str, str]
    rollback_rules: Mapping[str, bool]
```

Map failure codes deterministically to:
`RETRY_LEARNED`, `REPLAN_FROM_CURRENT_STATE`, `ROLLBACK_AND_REPLAN`, `USE_RRT_FALLBACK`, `ABORT_TASK`.

Examples:
- inference failure → rollback + RRT fallback;
- target occupied → reject before execution + replan;
- slippage → reobserve and replan, rollback configurable;
- missed grasp → bounded retry only if state remains valid.

Do not delegate rollback policy to free-form LLM text.

# 6. Grounded feedback

Render from current observation after failure:

```text
Agent: Alice
Skill: PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
Status: FAILED
Failure: SLIPPAGE
Progress: transporting
Measured current state:
- apple is on the table near bin_front_left
- Alice holds nothing
- bin_front_left is empty
Retry budget: 1
Choose one valid skill plan from the available capabilities.
```

Distinguish measured facts from inferred explanation. No fabricated positions or stale pre-execution state.

# 7. Failure-loop detection

Fingerprint:
```text
skill + object + target + failure code + quantized current state
```

Repeated fingerprint beyond configured threshold forces fallback or abort. Budgets never reset accidentally between planner rounds.

# 8. Event log

Record:
```text
OBSERVED, PROMPTED, PLAN_PARSED, PLAN_REJECTED
EXECUTOR_SELECTED, STATE_SNAPSHOTTED, EXECUTION_STARTED
EXECUTION_FAILED, STATE_RESTORED, FEEDBACK_RENDERED
REPLAN_REQUESTED, FALLBACK_STARTED, SKILL_SUCCEEDED
TASK_SUCCEEDED, BUDGET_EXHAUSTED
```

Include run/plan IDs, backend, state digest, reason, and budget.

# 9. Prompt changes

Include:
- current state summary;
- per-agent skills/capabilities;
- previous structured outcome;
- budget/fallback constraints;
- exact typed grammar;
- prohibition on low-level path/action output.

A learned capability is available, not guaranteed.

# 10. Tests

Unit:
- routing;
- budgets;
- failure→recovery mapping;
- grounded feedback;
- measured/inferred separation;
- fingerprint loop;
- rollback rules;
- event log;
- prompt rendering.

Canned end-to-end tests without external LLM:
1. success;
2. invalid plan;
3. learned failure then retry;
4. slippage then replan from current state;
5. inference failure then RRT fallback;
6. repeated failure loop;
7. budget exhaustion;
8. rollback digest equality.

External LLM tests are optional/manual/marked.

# 11. Metrics

```text
task success
learned attempts/success
fallback rate/success
replans and invalid plans
recovery success
failure distribution
rollbacks
steps/time
token usage and planner latency when used
```

Keep planner, learned executor, fallback, and overall metrics separate.

# 12. Acceptance criteria

- [ ] Typed plan routes deterministically.
- [ ] Current observation grounds feedback.
- [ ] Retry/replan/fallback budgets enforced.
- [ ] Rollback explicit and verified.
- [ ] Repeated failure loops terminate.
- [ ] Automated tests need no LLM.
- [ ] RRT-only mode remains.
- [ ] Learned/fallback/overall success separated.

# 13. Copy-ready Codex prompt

```text
Implement Phase 6 according to 06_PHASE6_PLANNER_INTEGRATION.md.

Inspect the current RoCo prompter/runner and Phase 1/5 interfaces. Make the smallest clean integration.

Implement deterministic executor routing, ReplanningPolicy/budgets, snapshot/rollback controller, state-grounded feedback, deterministic recovery strategies, failure fingerprint loop detection, planner event log, prompt capability/outcome additions, metrics, tests, and docs.

Use canned planner responses for all automated tests. Initial mode is sequential.

Never allow unbounded retries, let the LLM fabricate state, hide fallback, conflate learned and overall success, or put raw exceptions into prompts.

Verify learned failure→feedback→replan, inference failure→RRT fallback, repeated loop termination, and budget exhaustion.
```

# 14. Review and verification

**Review**
```text
Find stale feedback, unbounded replanning, incorrect rollback, backend controlled by free-form LLM, hidden fallback, repeated loops, accidental budget reset, raw trace leakage, synthetic legacy actions in skill history, and live-LLM test dependencies. Fix and rerun canned scenarios.
```

**Verify**
```text
Run success, invalid plan, missed-grasp retry, slippage replan, inference-failure fallback, repeated fingerprint, budget exhaustion, and rollback digest equality. Report event order, feedback, routing, budgets, metrics, and final state.
```
