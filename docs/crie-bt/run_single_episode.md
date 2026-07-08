# Run one task with a chosen method (video + planner output)

A focused workflow: run a **single task** with a **single method (condition)**,
record the episode **video**, and watch the **planner's output**. Uses the
canonical pipeline runner. For the full four-task paper sweep see
[running_the_experiment.md](running_the_experiment.md).

## 1. Environment (once per shell)

```bash
conda activate roco
python -c "import openai" 2>/dev/null || pip install "openai==0.28.1"
export MUJOCO_GL=egl
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

## 2. Pick a task and a method

- **Task** (`--tasks`): `sandwich`, `pack`, `cabinet`, or `sort`.
- **Method** (`--conditions`, pick one):

  | Condition | Meaning |
  |---|---|
  | `CRIE-BT-RR-Dialog` | ours, dialog coordination |
  | `CRIE-BT-RR-Cent` | ours, centralized |
  | `VLM-RR-Dialog` | baseline, dialog |
  | `VLM-RR-Cent` | baseline, centralized |

## 3. Run it (one episode, with video)

```bash
OUT=results/single/sandwich_criebt_dialog/results.jsonl

python scripts/run_condition_matrix.py \
  --stage step1 --backend roco \
  --conditions CRIE-BT-RR-Dialog \
  --tasks sandwich \
  --seeds 0 --episodes 1 \
  --llm-source gemini-2.5-flash --max-steps 30 \
  --record-video \
  --output "$OUT"
```

Everything lands under the output directory (`<tag>` =
`criebt_rr_dialog_sandwich_seed0_ep000`):

```text
results/single/sandwich_criebt_dialog/results.jsonl                  # the metrics row
results/single/sandwich_criebt_dialog/events/<tag>_events.jsonl      # event stream
results/single/sandwich_criebt_dialog/events/prompts/<tag>/          # planner prompts + LLM output
results/single/sandwich_criebt_dialog/events/video/<tag>/execute.mp4 # episode video
```

## 4. Watch the planner output while it runs

The event stream and the per-call planner prompts are flushed to disk **as the
episode runs**, so in a second terminal:

```bash
D=results/single/sandwich_criebt_dialog/events

# live event stream (planner calls, skill starts, monitor updates, replans)
tail -f "$D"/*_events.jsonl

# each planner call drops a folder here as it happens
watch -n 2 'ls -R '"$D"'/prompts'
```

## 5. Inspect the planner output (during or after)

```bash
# full planner reasoning + EXECUTE blocks, in call order
python scripts/show_plans.py results/single/sandwich_criebt_dialog/events/prompts

# just the EXECUTE blocks (the actions chosen each round)
python scripts/show_plans.py results/single/sandwich_criebt_dialog/events/prompts --execute-only

# also show the prompt the planner saw
python scripts/show_plans.py results/single/sandwich_criebt_dialog/events/prompts --full
```

## 6. Watch the video

```text
results/single/sandwich_criebt_dialog/events/video/<tag>/execute.mp4
```

Renders the task's `teaser` camera; frames accumulate across the episode, so
`execute.mp4` is the whole run. (`--record-video` requires `opencv`, present in
`roco`.)

## Notes

- One episode = `--episodes 1 --seeds 0`. Add more seeds/episodes to average.
- The metrics in `results.jsonl` and how each is measured:
  [crie_bt_conditions_tasks_metrics.md](crie_bt_conditions_tasks_metrics.md).
- To compare a method against the legacy runner on the same task/seed, use
  [equivalence_check.md](equivalence_check.md).
