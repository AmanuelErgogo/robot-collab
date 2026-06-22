# CRIE-BT Evaluation

CRIE-BT includes a stdlib-only scripted evaluation path for ablation studies.
It does not require LLM API keys, learned checkpoints, MuJoCo, or LeRobot.

## Run A Pack Evaluation

```bash
python scripts/run_crie_bt_eval.py \
  --task pack \
  --mode all \
  --episodes 2 \
  --executor scripted \
  --planner scripted \
  --output results/crie_bt/pack_all.jsonl
```

Supported modes:

```text
open_loop
direct_feedback
bt_mediated
all
```

Supported scripted tasks:

```text
sort
cabinet
rope
sweep
sandwich
pack
```

Unsupported real backends are skipped gracefully with a structured skip reason.

## Run CRIE-BT On RoCoBench Simulators

Use the existing RoCoBench simulator tasks through the CRIE-BT runner:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task pack \
  --mode all \
  --episodes 1 \
  --uncertainty policy_metadata \
  --output results/crie_bt/pack_grocery_sim.jsonl
```

Supported simulator task IDs:

```text
pack
sort
sweep
sandwich
rope
cabinet
all
```

For `--task pack`, `--adapter auto` uses the typed PackGrocery CRIE-BT
contract:

```text
PUT_OBJECT_IN_CONTAINER(object, container)
WAIT()
```

Each CRIE-BT subtask is converted into an existing RoCo `SkillPlan` with one
active agent and passive-agent `WAIT()` calls. Low-level simulator execution is
owned by:

```text
PackGrocerySkillPlanValidator
RRTSkillCompiler
rocobench.skills.executor.RRTSkillExecutor
PackGroceryTask.step(SimAction)
```

Optional object-target overrides use existing PackGrocery object and slot names:

```text
--object-targets apple:bin_front_left,banana:bin_front_right
```

For all tasks, including PackGrocery, `--adapter legacy` executes one existing
RoCoBench raw action response through `LLMResponseParser` and the existing RRT
executor:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task sort \
  --adapter legacy \
  --legacy-response-file artifacts/crie_bt/sort_action.txt \
  --output results/crie_bt/sort_legacy_sim.jsonl
```

The legacy response file should use the task's normal action grammar:

```text
EXECUTE
NAME Alice ACTION ...
NAME Bob ACTION ...
```

If no legacy response is provided, the runner emits a no-op `WAIT` action for
each task agent. That is useful as a smoke test for parser/executor wiring; it
does not imply the simulator task is complete.

### Run Proper RoCoBench Planners With Open Loop

The CRIE-BT simulator runner can call the existing RoCoBench planner prompting
modes for open-loop execution:

```text
plan
chat
dialog
```

These modes require the same LLM credentials as `run_dialog.py`; by default the
legacy prompters expect `openai_key.json` in the repository root.

Centralized plan mode:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task sandwich \
  --planner-mode plan \
  --mode open_loop \
  --episodes 1 \
  --artifact-dir artifacts/crie_bt/sandwich_plan_open_loop/executor_artifacts \
  --prompt-artifact-dir artifacts/crie_bt/sandwich_plan_open_loop/prompts \
  --output artifacts/crie_bt/sandwich_plan_open_loop/sandwich_sim.jsonl
```

Centralized chat-style mode:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task sandwich \
  --planner-mode chat \
  --mode open_loop \
  --episodes 1 \
  --artifact-dir artifacts/crie_bt/sandwich_chat_open_loop/executor_artifacts \
  --prompt-artifact-dir artifacts/crie_bt/sandwich_chat_open_loop/prompts \
  --output artifacts/crie_bt/sandwich_chat_open_loop/sandwich_sim.jsonl
```

Per-agent dialogue mode:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
  --task sandwich \
  --planner-mode dialog \
  --mode open_loop \
  --episodes 1 \
  --artifact-dir artifacts/crie_bt/sandwich_dialog_open_loop/executor_artifacts \
  --prompt-artifact-dir artifacts/crie_bt/sandwich_dialog_open_loop/prompts \
  --output artifacts/crie_bt/sandwich_dialog_open_loop/sandwich_sim.jsonl
```

Current implementation status:

```text
implemented: plan   + open_loop
implemented: chat   + open_loop
implemented: dialog + open_loop
planned:     plan   + direct_feedback
planned:     plan   + bt_mediated
planned:     chat   + direct_feedback
planned:     chat   + bt_mediated
planned:     dialog + direct_feedback
planned:     dialog + bt_mediated
```

The planned modes need planner-state updates after CRIE-BT feedback so that
prompt history, environment feedback, BT retry decisions, and replanning
requests stay coherent.

Fake uncertainty profiles:

```text
nominal
medium
low_confidence
```

The JSONL rows use the same episode schema as the scripted eval and also add
`task_id`, `adapter`, `sim_success`, `initial_scene`, `final_scene`, and
`task_spec`.

## Failure Scenarios

Use executor-level failure injection:

```bash
python scripts/run_crie_bt_eval.py \
  --task pack \
  --mode all \
  --episodes 2 \
  --executor scripted \
  --planner scripted \
  --failure-scenario missed_grasp \
  --max-retries 1 \
  --output results/crie_bt/pack_missed_grasp.jsonl
```

Available scenarios:

```text
none
missed_grasp
slippage
no_progress
target_occupied
human_interrupt
```

## Analyze Logs

```bash
python scripts/analyze_crie_bt_eval.py \
  results/crie_bt/pack_all.jsonl \
  --output-dir results/crie_bt/analysis
```

Outputs:

```text
summary.csv
summary.md
```

Aggregated fields include success rate, average steps, planner calls, replans,
local retries, failure counts, recovery rate, explanation count, and an
unnecessary-replanning proxy.

## Episode Log Schema

Each JSONL row contains:

```text
mode
task
success
steps
completed_subtasks
failed_subtasks
planner_calls
replans
local_retries
failure_counts
events
subtask_results
explanations
```

This common schema makes the three controller variants comparable.
