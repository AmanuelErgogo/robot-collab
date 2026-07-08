# Running the Robot-Robot Simulator Experiment

Paper-facing Robot-Robot simulator scope: **Sandwich, Pack Grocery, Cabinet,
Sort**. The comparison is baseline (`VLM-*`) vs. ours (`CRIE-BT-*`), in
centralized (`-Cent`) and dialog (`-Dialog`) coordination.

> **The pipeline is the canonical runner (decided 2026-07-08).** Use
> `scripts/run_condition_matrix.py --backend roco` (the `rocobench/crie_bt/pipeline/`
> layer) for all new runs. The legacy `scripts/run_crie_bt_sim.py` is **deprecated
> but kept** for reproducing prior runs — see the appendix at the bottom. Both run
> the *same* engine via `rocobench/crie_bt/roco_runtime.py`;
> [crie_bt_pipeline_merge.md](crie_bt_pipeline_merge.md) has the flag mapping and
> the equivalence result.

For a single task+method run with video and planner inspection, see
[run_single_episode.md](run_single_episode.md).

## Environment (one-time)

Runs use the `roco` conda env (MuJoCo + `google.genai`).

```bash
conda activate roco

# The prompting package imports openai at load even for Gemini — install once if missing.
python -c "import openai" 2>/dev/null || pip install "openai==0.28.1"

export MUJOCO_GL=egl
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

Readiness check (no tokens):

```bash
python -c "import openai, mujoco; from google import genai; print('deps ok')"
python -c "from rocobench.envs import MakeSandwichTask; print('rocobench.envs ok')"
test -f "$GOOGLE_APPLICATION_CREDENTIALS" && echo "creds present"
```

## Conditions

The eight paper conditions come from `configs/conditions/condition_registry.yaml`;
Robot-Robot uses these four (see
[crie_bt_conditions_tasks_metrics.md](crie_bt_conditions_tasks_metrics.md)):

| Condition | Family | Coordination | Prompter (legacy flag) |
|---|---|---|---|
| `CRIE-BT-RR-Dialog` | ours | dialog | `bt_mediated + dialog` |
| `VLM-RR-Dialog` | baseline | dialog | `vlm_sarm_monitor_planner + dialog` |
| `CRIE-BT-RR-Cent` | ours | centralized | `bt_mediated + chat` |
| `VLM-RR-Cent` | baseline | centralized | `vlm_sarm_monitor_planner + chat` |

## Primary four-task run (pipeline)

```bash
RESULT=results/robot_robot_sim_v1/results.jsonl

python scripts/run_condition_matrix.py \
  --stage step1 --backend roco \
  --conditions CRIE-BT-RR-Dialog VLM-RR-Dialog CRIE-BT-RR-Cent VLM-RR-Cent \
  --tasks sandwich pack cabinet sort \
  --seeds 0 1 2 --episodes 5 \
  --llm-source gemini-2.5-flash --max-steps 30 \
  --record-video \
  --output "$RESULT"
```

Outputs, all under the `--output` directory:

```text
results/robot_robot_sim_v1/results.jsonl                     # one row per episode
results/robot_robot_sim_v1/events/<tag>_events.jsonl         # per-episode event stream
results/robot_robot_sim_v1/events/prompts/<tag>/             # planner (LLM) prompt+output per call
results/robot_robot_sim_v1/events/video/<tag>/execute.mp4    # with --record-video
```

`<tag>` is `<code_name>_<task>_seed<N>_ep<NNN>` (e.g. `criebt_rr_dialog_sandwich_seed0_ep000`).
Start with one condition/task/seed to sanity-check, then scale up.

## Output schema (pipeline)

Each JSONL row conforms to
[../crie_next_stage_plan/logging_schema.json](../crie_next_stage_plan/logging_schema.json)
and always carries the full condition metadata:

```text
episode_id  stage  condition_name  controller_family  team_type  coordination_mode
skill_backend  monitor_backend  monitor_privileged  environment  planner_input_type
task_id  seed  episode_index
success  task_done  num_steps  planner_calls  replans  local_retries
failed_subtasks  dialogue_turns  human_interventions  monitor_updates  wall_time_s
```

Metric definitions and how each is measured (and why `replans` differs from the
legacy runner) are in
[crie_bt_conditions_tasks_metrics.md](crie_bt_conditions_tasks_metrics.md).

## Analysis

```bash
python scripts/analyze_condition_matrix.py \
  results/robot_robot_sim_v1/results.jsonl \
  --output-dir results/robot_robot_sim_v1/analysis --print
```

Writes CSV + LaTeX for: `summary`, `replanning`, `monitor`, `dialogue`,
`per_task`.

## Video and planner output

- **Video:** add `--record-video` (renders the task's `teaser` camera →
  `events/video/<tag>/execute.mp4`, the whole episode).
- **Planner output:** the LLM prompts + responses + `EXECUTE` blocks are saved
  under `events/prompts/<tag>/`. View them with `show_plans.py`:

```bash
python scripts/show_plans.py results/robot_robot_sim_v1/events/prompts/<tag>
python scripts/show_plans.py <...> --execute-only     # just the actions
```

See [run_single_episode.md](run_single_episode.md) for a focused one-run workflow.

## Smoke tests (no LLM / no tokens)

Synthetic (no MuJoCo, no LLM):

```bash
python scripts/run_condition_matrix.py --stage step1 --dry-run --episodes 1 \
  --output results/smoke/dry.jsonl
python scripts/run_condition_matrix.py --stage step1 \
  --conditions CRIE-BT-RR-Cent --tasks sandwich --episodes 1 \
  --output results/smoke/synthetic.jsonl
```

Real MuJoCo, no LLM (WAIT action — checks env/RRT/executor wiring):

```bash
MUJOCO_GL=egl python scripts/validate_roco_step1.py --task pack
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'mujoco'` — not in the `roco` env.
- `ModuleNotFoundError: No module named 'openai'` — run the one-time
  `pip install "openai==0.28.1"` above.
- `Gemini … Application Default Credentials` — `GOOGLE_APPLICATION_CREDENTIALS`
  unset or missing.
- `gladLoadGL error` — set `MUJOCO_GL=egl` for offscreen rendering.
- planner errors every episode — the LLM returned no parseable `EXECUTE` block;
  inspect `events/prompts/<tag>/` with `show_plans.py`.
- Long RRT calls: shorten with a smaller `--max-steps` while debugging.

---

## Appendix: deprecated legacy runner

`scripts/run_crie_bt_sim.py` is retained for reproducing pre-2026-07-08 runs and
will be removed in a future cleanup. It uses `--mode` / `--planner-mode` instead of
condition names, writes a different row schema (`paper_method`, `sim_success`,
`steps`, …), and is analyzed by `scripts/analyze_crie_bt_eval.py`. Mapping:

```text
CRIE-BT-RR-Dialog  ->  --mode bt_mediated              --planner-mode dialog --adapter legacy
VLM-RR-Dialog      ->  --mode vlm_sarm_monitor_planner --planner-mode dialog --adapter legacy
CRIE-BT-RR-Cent    ->  --mode bt_mediated              --planner-mode chat   --adapter legacy
VLM-RR-Cent        ->  --mode vlm_sarm_monitor_planner --planner-mode chat   --adapter legacy
```

Legacy example (one task, records video via `--artifact-dir`):

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS" \
  GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" GOOGLE_CLOUD_LOCATION="$GOOGLE_CLOUD_LOCATION" \
  GOOGLE_GENAI_PYTHON_BIN="$GOOGLE_GENAI_PYTHON_BIN" \
  python scripts/run_crie_bt_sim.py \
    --task sandwich --adapter legacy --mode bt_mediated --planner-mode dialog \
    --llm-source gemini-2.5-flash --episodes 15 --seed 0 --max-steps 30 \
    --num-replans 2 --max-retries 1 \
    --output results/robot_robot_sim_v1_legacy/sandwich/crie_bt_dialog/episodes.jsonl \
    --prompt-artifact-dir results/robot_robot_sim_v1_legacy/sandwich/crie_bt_dialog/prompts \
    --artifact-dir results/robot_robot_sim_v1_legacy/sandwich/crie_bt_dialog
```

Legacy analysis: `python scripts/analyze_crie_bt_eval.py <episodes.jsonl>
--output-dir <analysis> --group-by task_method` (writes `summary.csv` /
`summary.md`).
