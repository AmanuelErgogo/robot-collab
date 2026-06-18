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
