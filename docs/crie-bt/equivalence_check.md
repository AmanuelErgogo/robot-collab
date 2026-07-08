# Equivalence check: legacy runner vs new pipeline

The [merge](crie_bt_pipeline_merge.md) makes the new registry-driven pipeline run
the **same** real engine (LLM planner + RRT executor + coded monitor) as the
legacy `scripts/run_crie_bt_sim.py`. `scripts/equivalence_check.py` runs both
paths on the same task and seed and diffs the comparable metrics so you can
confirm the merge did not change behaviour before trusting the pipeline for paper
runs.

## Prerequisites (one-time)

Run from the repo root in the `roco` conda env (has MuJoCo + `google.genai`).

```bash
conda activate roco

# The prompting package imports `openai` at module load even when using Gemini,
# so it must be importable. Install once if missing:
python -c "import openai" 2>/dev/null || pip install "openai==0.28.1"

# Offscreen rendering + Gemini/Vertex credentials
# (see docs/getting-started/llm_setup_and_credentials.md):
export MUJOCO_GL=egl
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

Quick check that the environment is ready (no tokens spent):

```bash
python -c "import openai, mujoco; from google import genai; print('deps ok')"
test -f "$GOOGLE_APPLICATION_CREDENTIALS" && echo "creds present"
```

## Run it

```bash
# 1) Meaningful check: real LLM, compare success + recovery on one task/seed.
python scripts/equivalence_check.py \
    --task sandwich --seed 0 \
    --conditions CRIE-BT-RR-Cent \
    --llm-source gemini-2.5-flash \
    --max-steps 12 \
    --out-dir results/equivalence/sandwich

# 2) Headless harness smoke (no credentials/tokens): drives both paths with WAIT.
python scripts/equivalence_check.py --task pack --no-llm --out-dir results/equivalence/smoke

# 3) Compare two already-produced result rows (no run).
python scripts/equivalence_check.py --compare \
    results/equivalence/sandwich/legacy_CRIE_BT_RR_Cent_sandwich_seed0.jsonl \
    results/equivalence/sandwich/pipeline_CRIE_BT_RR_Cent_sandwich_seed0.jsonl
```

Each run writes `legacy_<cond>_<task>_seed<N>.jsonl`, `pipeline_<...>.jsonl`, and
prompt artifacts under `--out-dir`, then prints the comparison table.

`--conditions` accepts any of the four names in the mapping below (space-separated
for several). Because a real run costs LLM tokens and RRT time (a few minutes per
episode), start with **one** condition/seed; scale up to several seeds once it
works — real-LLM equivalence is judged on **success rate over seeds**, not one
episode (see caveats).

If you are not in an interactive shell, the same run as a one-liner:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json" \
  GOOGLE_CLOUD_PROJECT="bloom-475216" GOOGLE_CLOUD_LOCATION="global" \
  GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python" \
  python scripts/equivalence_check.py --task sandwich --seed 0 \
    --conditions CRIE-BT-RR-Cent --llm-source gemini-2.5-flash --max-steps 12 \
    --out-dir results/equivalence/sandwich
```

## Record a video of the episode

Add `--record-video` to any run. Each path writes `execute.mp4` under `--out-dir`:

```bash
python scripts/equivalence_check.py \
    --task sandwich --seed 0 --conditions CRIE-BT-RR-Cent \
    --llm-source gemini-2.5-flash --max-steps 12 \
    --record-video --out-dir results/equivalence/sandwich
# -> results/equivalence/sandwich/legacy_video_<tag>/execute.mp4
#    results/equivalence/sandwich/pipeline_video_<tag>/execute.mp4
```

How it works: the RRT executor renders the task's `teaser` camera during MuJoCo
stepping and, when given an artifact dir, calls `env.export_render_to_video()`.
Frames accumulate across the episode, so the final `execute.mp4` is the **whole
episode**. (Requires `opencv` in the env, which `roco` has; a `WAIT`-only
`--no-llm` run still produces a short clip.)

To record video from the standard runners directly, pass their artifact dir:
`run_crie_bt_sim.py --artifact-dir DIR` (legacy) writes `DIR/execute.mp4`; the
pipeline's `build_roco_condition(..., artifact_dir=DIR)` / `build_roco_step1(...,
artifact_dir=DIR)` do the same.

## See the planner's output

Every planner (LLM) call is saved as JSON while the run happens. Locations:

- **pipeline** — `<out-dir>/pipeline_prompts_<cond>/planner_call_NNN/replan{k}_*.json`
- **legacy** — `<out-dir>/legacy_<tag>_prompts/planner_call_NNN/replan{k}_*.json`
  (or wherever `run_crie_bt_sim.py --prompt-artifact-dir` points; default is
  `<output>_prompts/`).

Each file is a JSON array of `{sender, message}` turns: `SystemPrompt`,
`UserPrompt`, then the model output (`Planner` for chat/plan, or per-agent turns
for dialog), plus a usage record. `replan{k}_feedback_*.json` holds the parse /
execution feedback for that round.

View them without hand-parsing JSON:

```bash
# all planner output for a run (reasoning + EXECUTE blocks), in call order
python scripts/show_plans.py results/equivalence/sandwich/pipeline_prompts_CRIE_BT_RR_Cent

# just the EXECUTE blocks the planner emitted
python scripts/show_plans.py results/equivalence/sandwich --execute-only

# also print the system/user prompt the planner saw
python scripts/show_plans.py <prompt-or-run-dir> --full
```

`show_plans.py` accepts either the prompt dir or the parent run dir (it searches
underneath). The pipeline's per-round rationale is also in the episode
`events.jsonl` (`planner_call_start` / `planner_call_end`), but the full LLM text
lives in these prompt artifacts.

## Condition → legacy flags

| Pipeline condition | legacy `--mode` | legacy `--planner-mode` |
|---|---|---|
| `CRIE-BT-RR-Cent` | `bt_mediated` | `chat` |
| `CRIE-BT-RR-Dialog` | `bt_mediated` | `dialog` |
| `VLM-RR-Cent` | `vlm_sarm_monitor_planner` | `chat` |
| `VLM-RR-Dialog` | `vlm_sarm_monitor_planner` | `dialog` |

## Reading the report

Per condition it prints legacy vs pipeline for `success`, `steps`, `replans`,
`local_retries`, `planner_calls` and a `✅ match` / `⚠️ diverge` verdict (task
success uses the legacy `sim_success` and the pipeline's task-done `success`).

## Results so far (real Gemini, 2026-07-08)

| Task | Condition | Verdict | Notes |
|---|---|---|---|
| pack | CRIE-BT-RR-Cent | ✅ exact match | Deterministic: Gemini produced no parseable plan; both paths made 1 planner call, recorded a planner error, ended `success=False`. All metrics identical. |
| sandwich | CRIE-BT-RR-Cent | ⚠️ diverge | Both attempted the *identical* first action (`Chad PUT bread_slice1 / Dave PICK tomato`; Gemini deterministic at temp 0), then the LLM produced no follow-up plan. But failure accounting differed: legacy `failed_subtasks=1` (TIMEOUT), pipeline `failed_subtasks=0`; and `replans` 1 vs 0. |

## Known caveats

1. **`replans` is defined differently and is not directly comparable.** Legacy
   counts *every* re-plan after the first; the pipeline counts only monitor-
   triggered recovery replans (normal next-stage proposals are `planner_calls`,
   not replans). Compare `planner_calls`, not `replans`.

2. **Failure/postcondition accounting — reconciled (2026-07-08).** The sandwich
   run first showed legacy `failed_subtasks=1` (TIMEOUT/postcondition) vs pipeline
   `0` for the identical action, because the pipeline's `CodedSimProgressMonitor`
   inferred stage-done from the executor's status while the legacy stack applied a
   separate `FailureDetector`/postcondition check. The plan
   (`01_locked_decisions.md §2`) says the coded monitor *is* the privileged judge
   and should use task predicates / postcondition / timeout signals — i.e. the
   legacy behaviour is the plan-aligned one. That logic is now **merged into the
   pipeline**: `RoCoRRTSkillExecutorAdapter` surfaces `postcondition_satisfied`
   and a real RRT `timeout`, and `CodedSimProgressMonitor` only reports STAGE_DONE
   when the postcondition holds (a completed motion with an unmet goal is not
   done). Verified by unit tests. Residual per-episode differences are expected
   from (a) RRT stochasticity and (b) a legacy `elapsed_steps ≥ max_steps` quirk
   (sim-steps vs episode budget) we deliberately did not copy.

3. **Real-LLM equivalence is statistical, not per-episode.** Two independent
   real-LLM runs may attempt different actions; compare **success rate over many
   episodes/seeds**, not one episode. (Per-episode exact match only holds in
   deterministic cases, e.g. the pack planner-failure above.)

4. The `--no-llm` WAIT smoke validates the harness only; the two control loops
   legitimately differ on an action that never completes (legacy re-plans to
   `max_steps`; pipeline stops when the planner returns no steps).
