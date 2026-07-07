# Running the Robot-Robot Simulator Experiment

This runbook is for the current paper-facing Robot-Robot simulator scope:
Sandwich, Pack Grocery, Cabinet, and Sort. The main comparison has two methods:
CRIE-BT-Dialog and VLM/SARM-Monitor-Planner-Dialog.

## Canonical Result Paths

Write new results under one root:

```text
results/robot_robot_sim_v1/{task_id}/{method}/episodes.jsonl
results/robot_robot_sim_v1/{task_id}/{method}/analysis/
results/robot_robot_sim_v1/{task_id}/{method}/prompts/
```

Use these method directory names:

```text
crie_bt_dialog
vlm_sarm_dialog
crie_bt_cent
vlm_sarm_cent
```

Older directories such as `results/sandwich_bt_mediated/` and
`results/c3_to_c6_runs/` are legacy artifacts. Do not append new paper runs to
them.

## Environment

Simulation runs use the `roco` conda environment:

```bash
conda activate roco
export MUJOCO_GL=egl
```

For Gemini through Vertex AI, set credentials without printing secrets:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

Quick checks:

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -c "from rocobench.envs import MakeSandwichTask; print('ok')"
```

## Methods

Primary methods:

| Method | Runner flags |
| --- | --- |
| CRIE-BT-Dialog | `--mode bt_mediated --planner-mode dialog --adapter legacy` |
| VLM/SARM-Monitor-Planner-Dialog | `--mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy` |

Centralized ablations:

| Ablation | Runner flags |
| --- | --- |
| CRIE-BT-Cent | `--mode bt_mediated --planner-mode chat --adapter legacy` |
| VLM/SARM-Monitor-Planner-Cent | `--mode vlm_sarm_monitor_planner --planner-mode chat --adapter legacy` |

The VLM/SARM simulator baseline uses the same monitor interface as the planned
real VLM/SARM backend. In simulator runs, `SimulatorSignalVLMSARMMonitor` uses
`env.get_reward_done()` as the done signal and executor feedback as the failed
signal.

## Smoke Tests

Run a stdlib-only controller smoke test:

```bash
python scripts/run_crie_bt_eval.py \
  --task sandwich \
  --mode vlm_sarm_monitor_planner \
  --episodes 1 \
  --executor scripted \
  --planner scripted \
  --output results/smoke/vlm_sarm_scripted.jsonl
```

Run one MuJoCo/RRT no-LLM smoke test:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --adapter legacy \
    --mode open_loop \
    --episodes 1 \
    --output results/smoke/sandwich_mujoco_wait.jsonl
```

The MuJoCo smoke test uses generated `WAIT` actions when no legacy response is
supplied. It checks parser, simulator, and executor wiring; it is not a
task-success run.

## Four-Task Primary Runs

Run from the repository root. These commands create output directories
automatically and write prompt artifacts to the canonical `prompts/` directory.

```bash
RESULT_ROOT=results/robot_robot_sim_v1
TASKS="sandwich pack cabinet sort"
EPISODES=15
SEED=0
MODEL=gemini-2.5-flash

for task in $TASKS; do
  conda run --no-capture-output -n roco env MUJOCO_GL=egl \
    GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS" \
    GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
    GOOGLE_CLOUD_LOCATION="$GOOGLE_CLOUD_LOCATION" \
    GOOGLE_GENAI_PYTHON_BIN="$GOOGLE_GENAI_PYTHON_BIN" \
    python scripts/run_crie_bt_sim.py \
      --task "$task" \
      --adapter legacy \
      --mode bt_mediated \
      --planner-mode dialog \
      --llm-source "$MODEL" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --max-steps 30 \
      --num-replans 2 \
      --max-retries 1 \
      --output "$RESULT_ROOT/$task/crie_bt_dialog/episodes.jsonl" \
      --prompt-artifact-dir "$RESULT_ROOT/$task/crie_bt_dialog/prompts"

  conda run --no-capture-output -n roco env MUJOCO_GL=egl \
    GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS" \
    GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
    GOOGLE_CLOUD_LOCATION="$GOOGLE_CLOUD_LOCATION" \
    GOOGLE_GENAI_PYTHON_BIN="$GOOGLE_GENAI_PYTHON_BIN" \
    python scripts/run_crie_bt_sim.py \
      --task "$task" \
      --adapter legacy \
      --mode vlm_sarm_monitor_planner \
      --planner-mode dialog \
      --llm-source "$MODEL" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --max-steps 30 \
      --num-replans 2 \
      --output "$RESULT_ROOT/$task/vlm_sarm_dialog/episodes.jsonl" \
      --prompt-artifact-dir "$RESULT_ROOT/$task/vlm_sarm_dialog/prompts"
done
```

## Centralized Ablations

Use the same loop with these substitutions:

```text
CRIE-BT-Cent:
  --mode bt_mediated --planner-mode chat
  output: $RESULT_ROOT/$task/crie_bt_cent/episodes.jsonl
  prompts: $RESULT_ROOT/$task/crie_bt_cent/prompts

VLM/SARM-Monitor-Planner-Cent:
  --mode vlm_sarm_monitor_planner --planner-mode chat
  output: $RESULT_ROOT/$task/vlm_sarm_cent/episodes.jsonl
  prompts: $RESULT_ROOT/$task/vlm_sarm_cent/prompts
```

## Output Schema

Each JSONL line is one episode. Simulator rows include:

```text
episode
task_id
task_name
adapter
planner_mode
paper_method
mode
success
sim_success
steps
planner_calls
replans
local_retries
failure_counts
events
subtask_results
explanations
wall_time_s
llm_call_latencies_s
llm_prompt_tokens
llm_completion_tokens
llm_total_tokens
initial_scene
final_scene
task_spec
```

The VLM/SARM baseline additionally logs monitor events:

```text
event_type = VLM_SARM_MONITOR
payload.monitor_decision.status = DONE | FAILED | IN_PROGRESS
payload.monitor_decision.source = simulator_done_failed_signal
```

Metric mapping:

| Paper metric | Source |
| --- | --- |
| Task success rate | `sim_success` |
| Controller success rate | `success` |
| Completion time | `wall_time_s` |
| Token consumption | `llm_prompt_tokens`, `llm_completion_tokens`, `llm_total_tokens` |
| LLM latency | `llm_call_latencies_s` |
| Recovery behavior | `replans`, `local_retries`, `failure_counts`, `events` |
| Reactivity | `reactivity_s` when externally annotated |
| Hallucination rate | `hallucination_count / hallucination_annotation_count` when annotated |

## Analysis

Analyze one run:

```bash
python scripts/analyze_crie_bt_eval.py \
  results/robot_robot_sim_v1/sandwich/crie_bt_dialog/episodes.jsonl \
  --output-dir results/robot_robot_sim_v1/sandwich/crie_bt_dialog/analysis \
  --group-by task_method
```

Analyze all four tasks and both primary methods:

```bash
RESULT_ROOT=results/robot_robot_sim_v1
cat \
  "$RESULT_ROOT"/sandwich/crie_bt_dialog/episodes.jsonl \
  "$RESULT_ROOT"/sandwich/vlm_sarm_dialog/episodes.jsonl \
  "$RESULT_ROOT"/pack/crie_bt_dialog/episodes.jsonl \
  "$RESULT_ROOT"/pack/vlm_sarm_dialog/episodes.jsonl \
  "$RESULT_ROOT"/cabinet/crie_bt_dialog/episodes.jsonl \
  "$RESULT_ROOT"/cabinet/vlm_sarm_dialog/episodes.jsonl \
  "$RESULT_ROOT"/sort/crie_bt_dialog/episodes.jsonl \
  "$RESULT_ROOT"/sort/vlm_sarm_dialog/episodes.jsonl \
  > "$RESULT_ROOT"/all_primary.jsonl

python scripts/analyze_crie_bt_eval.py \
  "$RESULT_ROOT"/all_primary.jsonl \
  --output-dir "$RESULT_ROOT"/analysis \
  --group-by task_method
```

The analyzer writes `summary.csv` and `summary.md`.

## Troubleshooting

`ModuleNotFoundError: No module named 'mujoco'` means the command did not run
inside the `roco` environment.

`FileNotFoundError: Gemini Vertex AI access requires Application Default
Credentials` means `GOOGLE_APPLICATION_CREDENTIALS` is unset or points to a
missing file.

`PLANNER_ERROR` in every episode usually means the LLM response did not contain
a parseable `EXECUTE / NAME / ACTION` block. Check the prompt artifacts under
the run's `prompts/` directory.

`RuntimeError: No ready agents` means the LLM used agent names that do not match
the task's active robot names. Inspect `initial_scene` and `task_spec` in the
episode row.

Long RRT planning calls can be shortened for debugging with
`--max-sim-steps 2000`, at the cost of more timeout/no-progress failures.
