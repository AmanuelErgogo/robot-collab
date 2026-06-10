# Behavior Tree Integration

## What changed

The execution layer now accepts a `BehaviorTreePlan` instead of treating a parsed
LLM response as a raw list of sequential `LLMPathPlan` objects.

The current integration keeps the existing motion-planning stack intact and
repositions it as a Behavior Tree leaf skill:

- `prompting/parser.py` parses one `EXECUTE` block into `MotionPrimitive`s and
  compiles them into a `BehaviorTreePlan`
- `prompting/feedback.py` can validate either a single primitive or a whole BT
- `rocobench/policy.py` now has:
  - `MotionPrimitivePolicy`: the existing RRT-backed primitive executor
  - `BehaviorTreePolicy`: a BT-driven executor that activates leaf motion
    primitives online against the current environment state
- `run_dialog.py` executes one BT per round and persists:
  - `bt_plan.pkl`
  - per-primitive `llm_plan_*.pkl`
  - `rrt_plan.pkl`
  - `actions.pkl`

## Why this is useful

This is a first structural step away from a purely open-loop
subtask-command interface.

Compared with the previous representation, the BT layer gives us:

- an explicit execution structure
- a place to attach preconditions and recovery branches
- online leaf activation from current state instead of flattening everything
  up front
- a cleaner interface for future reactive execution and human interruption

## Current scope

This change is intentionally conservative.

The current `BehaviorTreePlan` is mostly a compiled sequence of existing motion
primitives. In other words, the architecture is now BT-shaped, but the leaves
still use the same underlying RRT and grasp/release machinery as before.

That means:

- simulator execution works through the BT path
- failed subplans are surfaced cleanly through BT initialization and feedback
- recovery is still limited because the tree currently behaves like a validated
  sequence rather than a rich reactive controller

## Validation performed

The following checks were run during development:

- Python compilation of the refactored BT-related modules
- focused unit tests for BT sequencing and validation behavior
- end-to-end simulator smoke tests on `SweepTask`
  - `WAIT / WAIT`
  - `MOVE red_cube / MOVE red_cube`
- `LLMRunner.one_run` smoke test with an injected `BehaviorTreePlan`
- scripted multi-round runner test on `SweepTask`

These checks confirmed that the BT execution path is wired end-to-end through
the runner. They also showed that later-round failures are now caught at the
BT/feedback boundary instead of being executed blindly.

## TODO for next time

- Add recovery nodes instead of failing the round immediately on validation
  errors.
- Introduce BT condition nodes for common execution predicates such as
  `object_visible`, `slot_free`, `human_clear`, and `grasp_stable`.
- Add fallback branches for manipulation-heavy tasks:
  - retry grasp
  - pick alternate item
  - choose alternate bin slot
  - reobserve before replanning
- Make the planner or stub planner feedback-aware so later BT rounds can change
  actions after a failed attempt.
- Separate expensive motion-planning failures from task-level failures in the
  logs and blackboard.
- Add task-specific BT templates instead of only compiling a flat sequence of
  primitives.
- Extend the runner to persist BT tick traces for easier debugging.
- Add an end-to-end regression test that runs multiple BT rounds automatically
  in simulation.
