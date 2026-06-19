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
