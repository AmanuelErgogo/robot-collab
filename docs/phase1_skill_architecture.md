# Phase 1 Skill Architecture

For setup, installation, tests, demos, and debugging steps, see
[`phase1_setup_install_test_demos.md`](phase1_setup_install_test_demos.md).

## Before

```text
LLM response
  -> prompting.parser.LLMResponseParser
  -> List[LLMPathPlan]
  -> prompting.feedback.FeedbackManager
  -> rocobench.policy.PlannedPathPolicy
  -> rocobench.rrt_multi_arm.MultiArmRRT
  -> env.step(...)
```

## After

```text
LLM skill response
  -> prompting.skill_parser.SkillResponseParser
  -> rocobench.skills.models.SkillPlan
  -> rocobench.skills.validation.PackGrocerySkillPlanValidator
  -> rocobench.skills.compiler.RRTSkillCompiler
  -> existing action-only LLMResponseParser
  -> existing FeedbackManager
  -> rocobench.skills.executor.RRTSkillExecutor
  -> existing PlannedPathPolicy / MultiArmRRT
  -> env.step(...)
```

`SkillPlan` remains the runner-facing representation. The synthetic legacy
response is backend metadata used only to prepare the RRT execution.

## Grammar

The final executable section must be:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The parser ignores discussion before `EXECUTE`, requires one `NAME ... ACTION`
line for each configured agent, rejects positional arguments, rejects duplicate
arguments, and resolves skill names case-insensitively through the registry.
Quoted string values are parsed without `eval` or `exec`.

## Grocery Skills

`PUT_OBJECT_IN_CONTAINER(object=<object>, container=<bin_slot>)`

Picks the object if needed, moves it to the requested bin slot, places it, and
releases it.

`WAIT()`

Keeps the robot stationary while another agent acts.

## Validation

`PackGrocerySkillPlanValidator` checks registry shape, configured agents,
object names, bin-slot names, object presence, packed objects, occupied target
slots, held-object recovery, inter-agent object conflicts, inter-agent slot
conflicts, and all-wait no-progress plans.

`PackGroceryTask` exposes reusable helpers:

```text
get_packed_slot_for_object(obs, object_name)
get_slot_occupancy(obs)
get_agent_held_object(obs, agent_name)
```

## Compilation

`RRTSkillCompiler` converts valid skill plans into a complete action-only legacy
response, then calls the existing `LLMResponseParser`.

```text
PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

compiles to:

```text
PICK apple PLACE bin_front_left
```

If the acting agent already holds `apple`, it compiles to:

```text
PLACE apple bin_front_left
```

`WAIT()` compiles to `WAIT`.

## Execution

`SkillExecutor` is backend-neutral. `RRTSkillExecutor` requires a prepared RRT
plan, iterates through compiled `LLMPathPlan` objects, creates
`PlannedPathPolicy`, runs RRT planning, steps the environment, and returns a
structured `SkillExecutionResult`.

Rollback remains owned by `LLMRunner`: it saves simulator state before execution
and restores it when the skill executor reports failure.

## Artifacts

Skill mode writes:

```text
skill_plan.json
skill_preparation.json
compiled_path_plans.pkl
rrt_plan_<i>.pkl
actions_<i>.pkl
skill_execution_result.json
execute.mp4
```

High-level prompt history uses `SkillPlan.get_action_desp()` and does not expose
synthetic `PICK`/`PLACE` text except in backend diagnostics.

## Commands

Run lightweight tests:

```bash
pytest tests/skills tests/test_legacy_parser_regression.py
```

Run integration tests when simulator dependencies are installed:

```bash
pytest -m integration
```

Run the no-LLM smoke test:

```bash
python scripts/smoke_test_pack_skill.py
```

Attempt real RRT execution:

```bash
python scripts/smoke_test_pack_skill.py --execute
```

Run a skill-mode dialogue:

```bash
python run_dialog.py \
  --task pack \
  --output_mode skill \
  --comm_mode dialog \
  --run_name pack_skill_phase1
```

This still uses RRT internally.

## Adding A Skill

Add a `SkillSpec` to the task-specific registry, extend the semantic validator
for task-state checks, add compiler rules for the first backend, and add parser,
validator, compiler, feedback, and executor tests. Keep task semantics out of
the geometric feedback and RRT code.

## Future Learned Executor

A future `LearnedSkillExecutor` should implement `SkillExecutor.execute(plan,
obs, artifact_dir=None)` and consume the same `SkillPlan`. It should not change
the LLM grammar, registry, parser, or semantic validator.

## Limitations

Phase 1 supports only `PackGroceryTask` with Alice and Bob. It does not add
LeRobot, dataset recording, learned-policy inference, new controllers, new IK,
or replacement motion planning. Other tasks fail early in `--output_mode skill`.
