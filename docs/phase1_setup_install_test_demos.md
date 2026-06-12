# Phase 1 Setup, Install, Test, Demo, and Debugging

This runbook sets up and exercises Phase 1 skill-level planning for
`PackGroceryTask`.

Phase 1 runs in the normal RoCo Python 3.8 simulator environment. It does not
use LeRobot, Gymnasium, or the Python 3.12 client package.

Run commands from the repository root unless stated otherwise.

## 1. What Phase 1 Adds

Skill mode accepts one structured skill call per agent:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

The runtime flow is:

```text
SkillResponseParser
  -> SkillPlan
  -> PackGrocerySkillPlanValidator
  -> RRTSkillCompiler
  -> existing action-only LLMResponseParser
  -> existing FeedbackManager
  -> RRTSkillExecutor
  -> existing PlannedPathPolicy / MultiArmRRT
```

The high-level LLM history stays in skill terms. The synthetic `PICK`/`PLACE`
plan is saved as backend diagnostics.

## 2. Environment Options

Use the lightweight path when you only need parser, validator, compiler, and
fake-backed executor checks.

Use the full RoCo path when you need MuJoCo, dm_control, geometric feedback,
RRT planning, rendered videos, or `run_dialog.py`.

## 3. Lightweight Dev Setup

This path needs no MuJoCo, no dm_control, no OpenAI key, and no LeRobot.

Use any Python with the repository on `PYTHONPATH`; running from the repo root
is enough:

```bash
python -c "from rocobench.skills import build_pack_grocery_skill_registry; print('skills OK')"
python -c "from prompting.skill_parser import SkillResponseParser; print('parser OK')"
```

If pytest is not installed in the active environment:

```bash
python -m pip install "pytest==7.4.4"
```

Run the lightweight tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills -q
```

Run the no-LLM smoke test:

```bash
python scripts/smoke_test_pack_skill.py
```

Expected output includes:

```text
Canonical skill plan:
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()

Synthetic legacy plan:
EXECUTE
NAME Alice ACTION PICK apple PLACE bin_front_left
NAME Bob ACTION WAIT
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is optional in a clean environment. It is
useful when globally installed pytest plugins are incompatible with the active
Python version.

## 4. Full RoCo Runtime Install

Create the simulator environment:

```bash
conda create -n roco python=3.8
conda activate roco
```

Install MuJoCo, dm_control, and RoCo dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install mujoco==2.3.0
python -m pip install dm_control==1.0.8
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On Linux headless machines, set a MuJoCo rendering backend before simulator
commands:

```bash
export MUJOCO_GL=egl
```

Quick simulator import check:

```bash
python -c "from rocobench.envs.task_pack import PackGroceryTask; print('RoCo PackGrocery OK')"
```

OpenAI or Claude keys are not needed for parser tests or smoke tests. They are
only needed when running an actual LLM dialogue through `run_dialog.py`.

The legacy OpenAI path expects `openai_key.json` at the repository root:

```json
"YOUR_OPENAI_KEY"
```

## 5. Test Commands

Run all available tests in the active environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Show skipped dependency-backed checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -rs
```

Run only Phase 1 skill tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills -q
```

Run simulator-backed Phase 1 integration when `dm_control` is installed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/integration/test_pack_skill_smoke.py -q
```

Compile changed Python modules:

```bash
python -m compileall rocobench/skills prompting run_dialog.py scripts/smoke_test_pack_skill.py tests
```

## 6. Demo: No-LLM Skill Smoke Test

This is the fastest Phase 1 demo and works without simulator dependencies:

```bash
python scripts/smoke_test_pack_skill.py
```

It verifies:

- strict skill parsing;
- canonical `SkillPlan.get_action_desp()`;
- semantic validation through fake PackGrocery state;
- compilation to the legacy action-only response;
- fake geometric feedback pass.

## 7. Demo: Real RRT-Backed Smoke Test

Use the RoCo Python 3.8 environment with MuJoCo and dm_control installed:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/smoke_test_pack_skill.py --execute
```

This attempts:

- real `PackGroceryTask` creation;
- canned skill response parsing;
- semantic validation;
- compilation through the existing `LLMResponseParser`;
- existing geometric feedback;
- RRT-backed execution through `RRTSkillExecutor`.

If simulator dependencies are unavailable, the script exits nonzero with a
clear dependency message.

## 8. Demo: Manual Skill-Mode Runner

This exercises `run_dialog.py` without an external LLM. It still requires the
full RoCo simulator environment.

```bash
conda activate roco
export MUJOCO_GL=egl
python run_dialog.py \
  --task pack \
  --output_mode skill \
  --comm_mode plan \
  --debug_mode \
  --run_name pack_skill_manual_debug \
  --tsteps 1 \
  --num_replans 1 \
  --skip_display
```

When prompted, enter skill calls only, one per agent:

```text
PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
WAIT()
```

Do not include `NAME`, `ACTION`, or `EXECUTE` in each typed answer. The debug
prompter wraps those fields.

## 9. Demo: LLM Skill-Mode Dialogue

Use this only after the full simulator environment and API key are configured:

```bash
conda activate roco
export MUJOCO_GL=egl
python run_dialog.py \
  --task pack \
  --output_mode skill \
  --comm_mode dialog \
  --run_name pack_skill_phase1 \
  --tsteps 3 \
  --num_replans 3 \
  --skip_display
```

Skill mode is supported only for `--task pack`. Other tasks fail early with:

```text
Skill mode is currently implemented only for --task pack.
```

## 10. Expected Artifacts

Skill-mode runner steps write artifacts under:

```text
data/<run_name>/run_<id>/step_<n>/
```

Expected Phase 1 files include:

```text
skill_plan.json
skill_preparation.json
compiled_path_plans.pkl
rrt_plan_<i>.pkl
actions_<i>.pkl
skill_execution_result.json
execute.mp4
```

`skill_plan.json` contains the canonical skill-level plan.

`skill_preparation.json` contains backend diagnostics, including the synthetic
legacy action-only response.

`skill_execution_result.json` contains:

```json
{
  "success": true,
  "status": "success",
  "reason": "",
  "num_sim_steps": 143,
  "reward": 0,
  "done": false
}
```

Exact values depend on the scene and planner result.

## 11. Debugging Parser Failures

Parser failures happen before semantic validation.

Common causes:

- missing `EXECUTE`;
- missing an agent line;
- duplicate agent line;
- unknown agent name;
- unknown skill name;
- positional arguments such as `PUT_OBJECT_IN_CONTAINER(apple, bin_front_left)`;
- missing required argument;
- duplicate argument key;
- malformed parentheses;
- `WAIT(object=apple)`.

Valid final output:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

Invalid final output:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(apple, bin_front_left)
NAME Bob ACTION WAIT()
```

Run parser-focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills/test_skill_parser.py -q
```

## 12. Debugging Semantic Validation

Semantic failures use stable error codes suitable for LLM replanning feedback.

Examples:

```text
[UNKNOWN_OBJECT] orange is not a valid grocery object.
[UNKNOWN_TARGET] bin_missing is not a valid bin slot.
[TARGET_OCCUPIED] bin_front_left is already occupied by banana.
[OBJECT_HELD_BY_OTHER_AGENT] apple is already held by Bob.
[AGENT_HOLDING_DIFFERENT_OBJECT] Alice is holding banana, not apple.
[RESOURCE_CONFLICT] Alice and Bob both target apple.
[NO_PROGRESS] At least one agent must execute PUT_OBJECT_IN_CONTAINER.
```

Inspect current task-state helpers in a RoCo shell:

```bash
python - <<'PY'
from rocobench.envs.task_pack import PackGroceryTask

env = PackGroceryTask(render_cameras=["teaser"], randomize_init=False)
obs = env.get_obs()

print("Alice holds:", env.get_agent_held_object(obs, "Alice"))
print("Bob holds:", env.get_agent_held_object(obs, "Bob"))
print("slot occupancy:", env.get_slot_occupancy(obs))
for obj in env.item_names:
    print(obj, "packed slot:", env.get_packed_slot_for_object(obs, obj))
PY
```

Run validator-focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills/test_pack_skill_validation.py -q
```

## 13. Debugging Compilation

Compilation converts skill calls to a synthetic legacy response. For example:

```text
EXECUTE
NAME Alice ACTION PICK apple PLACE bin_front_left
NAME Bob ACTION WAIT
```

If Alice already holds `apple`, the Alice line becomes:

```text
NAME Alice ACTION PLACE apple bin_front_left
```

Compiler failures usually mean the synthetic response failed the existing
action-only parser. Check `skill_preparation.json` when it exists, or rerun the
smoke script to print the synthetic response.

Run compiler-focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills/test_rrt_skill_compiler.py -q
```

## 14. Debugging Geometric Feedback And RRT Execution

After semantic validation and compilation, Phase 1 reuses the existing
`FeedbackManager`, `PlannedPathPolicy`, and `MultiArmRRT`.

Common backend failures:

- target pose is out of robot reach;
- inverse kinematics fails;
- endpoint collision is detected;
- RRT cannot find a collision-free path;
- execution exceeds the skill executor step limit.

Useful files:

```text
skill_preparation.json
compiled_path_plans.pkl
rrt_plan_<i>.pkl
actions_<i>.pkl
skill_execution_result.json
execute.mp4
```

If `skill_execution_result.json` has:

```json
{"status": "motion_planning_failed"}
```

read the `reason` field and inspect the corresponding `rrt_plan_<i>.pkl`
absence or failure context.

If it has:

```json
{"status": "timeout"}
```

increase the executor limit only after checking that the policy is not stuck in
a repeated failed action sequence.

## 15. Common Problems

### `ModuleNotFoundError: No module named 'dm_control'`

You are not in the RoCo Python 3.8 environment, or simulator dependencies were
not installed. Use the lightweight smoke test if you do not need real RRT:

```bash
python scripts/smoke_test_pack_skill.py
```

For simulator checks:

```bash
conda activate roco
python -m pip install dm_control==1.0.8 mujoco==2.3.0
```

### `ModuleNotFoundError: No module named 'gymnasium'`

Gymnasium is not required for Phase 1. You are probably running Phase 0 client
tests. Use Phase 1 commands under `tests/skills` or install the Phase 0 client
environment separately.

### `No module named 'lerobot'`

LeRobot is not required for Phase 1. Do not install LeRobot into the RoCo
runtime just for skill mode.

### `Please put your OpenAI API key...`

The no-LLM smoke test and unit tests do not need an API key. Actual LLM dialogue
through `run_dialog.py` does.

Create `openai_key.json` at the repository root:

```json
"YOUR_OPENAI_KEY"
```

### Rendering Fails On A Headless Machine

Try:

```bash
export MUJOCO_GL=egl
```

Then restart the simulator command. If EGL is unavailable, use an X display or
another MuJoCo-supported backend.

### Global Pytest Plugin Error Before Test Collection

Disable auto-loaded third-party plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

### Skill Mode Ignores `--output_mode skill`

Confirm the command uses `--task pack`. Phase 1 intentionally fails early for
other tasks.

### The LLM Outputs `PICK`, `PLACE`, `MOVE`, Or `PATH`

That is a prompt/format failure in skill mode. Feed back the parser or
validation message and require this final form:

```text
EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=<object>, container=<bin_slot>)
NAME Bob ACTION WAIT()
```

## 16. Quick Acceptance Checklist

Use this checklist after changes to Phase 1 code:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/skills -q
python scripts/smoke_test_pack_skill.py
python -m compileall rocobench/skills prompting run_dialog.py scripts/smoke_test_pack_skill.py tests
```

When simulator dependencies are available:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/integration/test_pack_skill_smoke.py -q
python scripts/smoke_test_pack_skill.py --execute
```
