# CRIE-BT API And Extension Guide

## Add A New Skill

1. Add a typed skill declaration to the relevant registry.
2. Ensure the planner can emit a `SkillCall` with typed arguments.
3. Add validation rules for required arguments and task-specific constraints.
4. Add resource claims if the skill can run in multi-agent schedules.
5. Add executor support in scripted, RRT, or learned backends.

Minimal skill call:

```python
SkillCall(
    agent="Alice",
    skill_name="PUT_OBJECT_IN_CONTAINER",
    arguments={"object": "apple", "container": "bin_front_left"},
    instruction="Put apple into bin_front_left.",
)
```

## PackGrocery CRIE-BT Contract

The PackGrocery adapter in `rocobench.crie_bt.roco_adapters` uses the existing
`rocobench.envs.task_pack.PackGroceryTask` and existing `rocobench.skills`
contracts. It does not define a new task.

Skills:

```text
PUT_OBJECT_IN_CONTAINER(object, container)
WAIT()
```

Subtasks are one `PUT_OBJECT_IN_CONTAINER` call per unpacked PackGrocery item.
Objects and slots come from the live task instance:

```text
env.item_names
env.bin_slot_xposes
```

CRIE-BT converts each subtask into a one-active-agent RoCo `SkillPlan`:

```text
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The existing compiler/executor then owns low-level simulator control:

```text
RRTSkillCompiler -> legacy PICK/PLACE response
RRTSkillExecutor -> PlannedPathPolicy -> env.step(SimAction)
POSTCONDITION_CHECK -> env.get_packed_slot_for_object(obs, object)
```

Observations are existing RoCo `EnvState` objects. CRIE-BT progress reads:

```text
obs.objects[item].xpos
obs.objects[item].contacts
obs.<robot_name>.ee_xpos
obs.<robot_name>.contacts
env.get_agent_held_object(obs, agent)
env.get_slot_occupancy(obs)
env.get_packed_slot_for_object(obs, object)
```

Subtask success is true when:

```text
env.get_packed_slot_for_object(obs, object) == container
```

Failure/pass detection maps existing PackGrocery validation, compilation, RRT,
timeout, and postcondition failures to CRIE-BT `FailureCode` values.

The adapter also emits fake policy metadata:

```text
confidence
entropy
action_norm
chunk_disagreement
```

Use `UncertaintyEstimator(mode="policy_metadata")` to consume this metadata.
The values are deterministic placeholders for later learned-model entropy or
ensemble estimators.

## Legacy RoCoBench Task Contract

The remaining RoCoBench tasks are integrated through
`rocobench.crie_bt.legacy_tasks`. This adapter reuses each task's existing
`EXECUTE`, `NAME`, and `ACTION` grammar, the existing `LLMResponseParser`, and
the existing RRT execution path. It does not introduce new task semantics.

Supported task IDs:

```text
pack
sort
sweep
sandwich
rope
cabinet
```

The legacy CRIE-BT skill is:

```text
LEGACY_ACTION_PLAN(response)
```

`response` is the raw RoCoBench action response for that task, for example:

```text
EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION WAIT
```

For non-Pack tasks, subtask success means the raw response was accepted by the
task parser and the RRT executor completed the compiled path plan. Whole-task
success remains the task's own simulator postcondition:

```text
env.get_reward_done(obs)[1]
```

Task-specific failure/pass checks come from:

```text
LLMResponseParser.parse(...)
env.get_task_feedback(...)
rocobench.skills.executor.RRTSkillExecutor
env.get_reward_done(...)
```

The legacy adapter also emits fake policy metadata with the same placeholder
fields as PackGrocery:

```text
confidence
entropy
action_norm
chunk_disagreement
```

This keeps CRIE-BT progress, failure, and uncertainty logs consistent while
leaving room to replace the fake metadata with learned-policy entropy or
ensemble estimators later.

## Add A Failure Detector

Extend `FailureDetector.detect(...)` or compose a new detector with the same
signature:

```python
detect(skill_call, progress, uncertainty, observation, executor_feedback=None)
```

Return a `FailureState` with:

```text
is_failure
failure_code
severity
message
evidence
```

Keep evidence JSON-serializable and avoid throwing on missing environment
fields.

## Add An Uncertainty Source

Add a mode to `UncertaintyEstimator.estimate(...)`. Do not claim calibrated
uncertainty unless calibration has actually been implemented. Use:

```text
confidence
uncertainty
risk_level
source
evidence
```

## Add A Communication Event

Add event handling to `CommunicationManager.generate_utterance(...)` and
`generate_overlay(...)`. Communication events should be logged even when no real
TTS, STT, or visual overlay system is connected.

## Add A Real Executor Backend

Implement the `BaseSkillExecutor` interface:

```python
reset(env, context)
start_skill(skill_call, observation)
step(observation)
stop()
```

The executor should return `ExecutionFeedback` on every step. RRT and learned
models should be injected through adapters rather than imported into unrelated
RoCo modules.

## Backward Compatibility

CRIE-BT is optional. Existing `run_dialog.py`, `action_only`, `action_and_path`,
Phase 0 bridge, Phase 8 benchmark, and legacy LLM behavior should continue to
work without importing CRIE-BT.
