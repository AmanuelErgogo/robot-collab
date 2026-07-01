# Running the CRIE-BT Experiment

## Condition matrix

| | Centralised (`chat`) | Dialog (`dialog`) |
|---|---|---|
| **No-Feedback** (`open_loop`) | C1 ✅ done | C2 ✅ done |
| **With-Feedback** (`direct_feedback`) | C3 | C4 |
| **Feedback+BT** (`bt_mediated`) | C5 | C6 |

C1 and C2 results are in `results/sandwich_open_loop/`. C3–C6 need to be run.

---

## 1. Activate the environment

All simulation runs use the `roco` conda environment (Python 3.8, MuJoCo 2.3.0,
dm_control 1.0.8). This is the only env with both MuJoCo and `rocobench`
installed.

```bash
conda activate roco
```

On a headless Linux machine (no display), also set:

```bash
export MUJOCO_GL=egl
```

Verify the env works:

```bash
python -c "import mujoco; print(mujoco.__version__)"          # → 2.3.0
python -c "from rocobench.envs import MakeSandwichTask; print('ok')"
```

---

## 2. Set up LLM credentials (Gemini via Vertex AI)

The run script uses `llm_api.py` to call Gemini 2.5 Flash through Vertex AI.
Credentials are resolved in this order (first match wins):

```text
GOOGLE_APPLICATION_CREDENTIALS  ← env var pointing to service account JSON
GEMINI_API_KEY_PATH              ← env var pointing to service account JSON
LLM_API_KEY_PATH                 ← env var pointing to service account JSON
gemini_key.json                  ← file in repo root
gemini_api_key.json              ← file in repo root
google_api_key.json              ← file in repo root
```

Recommended setup — set these before running (add to your shell profile to
persist):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json"
export GOOGLE_CLOUD_PROJECT="bloom-475216"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python"
```

`GOOGLE_GENAI_PYTHON_BIN` is required because the `roco` env is Python 3.8
but the `google-genai` SDK requires Python 3.9+. The run script uses this
venv's Python as a subprocess to make Gemini calls.

Check credentials are visible without printing them:

```bash
python - <<'PY'
import os
for name in ["GOOGLE_APPLICATION_CREDENTIALS","GOOGLE_CLOUD_PROJECT",
             "GOOGLE_CLOUD_LOCATION","GOOGLE_GENAI_PYTHON_BIN"]:
    print("{}: {}".format(name, "set" if os.environ.get(name) else "UNSET"))
PY
```

Do not commit credential files. The `.gitignore` already excludes
`gemini_key.json`, `.env`, and `.secrets/`.

---

## 3. Run a smoke test first

Run one episode with scripted planners (no LLM, no MuJoCo) to confirm the
CRIE-BT wiring is correct:

```bash
python scripts/run_crie_bt_eval.py \
  --task sandwich \
  --mode all \
  --episodes 1 \
  --executor scripted \
  --planner scripted \
  --output results/smoke/sandwich_scripted.jsonl
```

Then run one live MuJoCo episode with a scripted (non-LLM) planner:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --mode open_loop \
    --episodes 1 \
    --output results/smoke/sandwich_mujoco.jsonl
```

Both should exit cleanly and write a JSONL file. If the second command fails,
check `MUJOCO_GL=egl` is set and the `roco` env is active.

---

## 4. Run C3–C6

All commands use `conda run` so that the `roco` env and the `MUJOCO_GL`
variable are set correctly regardless of which shell you are in.

Run from the repository root. Each command creates its output directory
automatically and saves raw LLM prompts to `<output>_prompts/`.

### C3 — With-Feedback × Centralised

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json" \
  GOOGLE_CLOUD_PROJECT="bloom-475216" \
  GOOGLE_CLOUD_LOCATION="global" \
  GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python" \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --mode direct_feedback \
    --planner-mode chat \
    --llm-source gemini-2.5-flash \
    --episodes 15 \
    --seed 0 \
    --max-steps 30 \
    --num-replans 2 \
    --output results/sandwich_direct_feedback/episodes.jsonl
```

### C4 — With-Feedback × Dialog

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json" \
  GOOGLE_CLOUD_PROJECT="bloom-475216" \
  GOOGLE_CLOUD_LOCATION="global" \
  GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python" \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --mode direct_feedback \
    --planner-mode dialog \
    --llm-source gemini-2.5-flash \
    --episodes 15 \
    --seed 0 \
    --max-steps 30 \
    --num-replans 2 \
    --output results/sandwich_direct_feedback_dialog/episodes.jsonl
```

### C5 — Feedback+BT × Centralised (proposed method)

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json" \
  GOOGLE_CLOUD_PROJECT="bloom-475216" \
  GOOGLE_CLOUD_LOCATION="global" \
  GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python" \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --mode bt_mediated \
    --planner-mode chat \
    --llm-source gemini-2.5-flash \
    --episodes 15 \
    --seed 0 \
    --max-steps 30 \
    --num-replans 2 \
    --max-retries 1 \
    --output results/sandwich_bt_mediated/episodes.jsonl
```

### C6 — Feedback+BT × Dialog (proposed method)

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  GOOGLE_APPLICATION_CREDENTIALS="$HOME/.secrets/bloom-gemini-sa.json" \
  GOOGLE_CLOUD_PROJECT="bloom-475216" \
  GOOGLE_CLOUD_LOCATION="global" \
  GOOGLE_GENAI_PYTHON_BIN="$PWD/.venv-google-genai/bin/python" \
  python scripts/run_crie_bt_sim.py \
    --task sandwich \
    --mode bt_mediated \
    --planner-mode dialog \
    --llm-source gemini-2.5-flash \
    --episodes 15 \
    --seed 0 \
    --max-steps 30 \
    --num-replans 2 \
    --max-retries 1 \
    --output results/sandwich_bt_mediated_dialog/episodes.jsonl
```

---

## 5. Key flags

| Flag | Paper value | What it controls |
| --- | --- | --- |
| `--task` | `sandwich` | Task: `sandwich`, `pack`, `sort`, `sweep`, `rope`, `cabinet`, `all` |
| `--mode` | see above | Controller: `open_loop`, `direct_feedback`, `bt_mediated`, `all` |
| `--planner-mode` | `chat` or `dialog` | `chat` = centralised+history (C1/C3/C5), `dialog` = per-agent turns (C2/C4/C6) |
| `--llm-source` | `gemini-2.5-flash` | Model name passed to `llm_api.py` |
| `--episodes` | `15` | Independent episodes (each gets seed `--seed + episode_index`) |
| `--seed` | `0` | Base RNG seed |
| `--max-steps` | `30` | Max controller ticks per episode (sandwich has ~10 actions; headroom for replans) |
| `--num-replans` | `2` | Max LLM retries per planning round before `PLANNER_ERROR` |
| `--max-retries` | `1` | BT local retries before escalating to replan (`bt_mediated` only) |
| `--temperature` | `0.0` (default) | LLM sampling temperature |
| `--max-tokens` | `1024` (default) | Max tokens in LLM response |
| `--output` | `results/.../episodes.jsonl` | Output JSONL path (directory auto-created) |
| `--artifact-dir` | unset | If set, simulator frames/artifacts go here |
| `--prompt-artifact-dir` | unset | Override for LLM prompt/response JSON; defaults to `<output>_prompts/` |

---

## 6. Output format

Each JSONL line is one episode:

```json
{
  "episode": 0,
  "task_id": "sandwich",
  "mode": "bt_mediated",
  "success": true,
  "sim_success": true,
  "steps": 12,
  "planner_calls": 3,
  "replans": 1,
  "local_retries": 0,
  "completed_subtasks": 10,
  "failed_subtasks": 1,
  "failure_counts": {"MISSED_GRASP": 1},
  "initial_scene": "...",
  "final_scene": "...",
  "subtask_results": [...],
  "explanations": [...]
}
```

Key field → paper metric mapping:

| Field | Paper metric |
|---|---|
| `sim_success` | **TCR** — `env.get_reward_done()` returned `done=True` |
| `success` | CRIE-BT controller success (plan-level; use `sim_success` for TCR) |
| `planner_calls` | LLM calls per episode |
| `replans` | Replan count (always 0 for C1/C2) |
| `local_retries` | BT local retries (C5/C6 only) |
| `failure_counts` | Breakdown by `FailureCode` |

---

## 7. Analyse results

```bash
python scripts/analyze_crie_bt_eval.py \
  results/sandwich_bt_mediated/episodes.jsonl \
  --output-dir results/sandwich_bt_mediated/analysis
```

Produces `analysis/summary.csv` and `analysis/summary.md`.

To compare all conditions in one table:

```bash
cat results/sandwich_direct_feedback/episodes.jsonl \
    results/sandwich_direct_feedback_dialog/episodes.jsonl \
    results/sandwich_bt_mediated/episodes.jsonl \
    results/sandwich_bt_mediated_dialog/episodes.jsonl \
    > results/c3_to_c6.jsonl

python scripts/analyze_crie_bt_eval.py \
  results/c3_to_c6.jsonl \
  --output-dir results/c3_to_c6_analysis
```

C1/C2 are analysed separately with `scripts/eval_sandwich_open_loop.py` (which
computes Wilson CIs and per-prompter-mode breakdowns). The result is already in
`results/sandwich_open_loop/summary.json`.

---

## 8. Increasing episode count for tighter CIs

n=15 gives a Wilson 95% CI width of ±13–20 pp. For paper-quality intervals use
n=30 or n=50 — just change `--episodes`:

```bash
  --episodes 30 \
  --seed 0 \
```

Seeds 0–29 are used automatically (one per episode).

---

## 9. Troubleshooting

**`FileNotFoundError: Gemini Vertex AI access requires Application Default Credentials`**
The four environment variables in §2 are not set, or the service account JSON
path is wrong. Run the credential check in §2 and confirm the file exists at
`$GOOGLE_APPLICATION_CREDENTIALS`.

**`ModuleNotFoundError: No module named 'mujoco'`**
You are not in the `roco` conda env. Run `conda activate roco` or use
`conda run --no-capture-output -n roco`.

**`PLANNER_ERROR` in every episode**
The LLM is returning unparseable output. Check:

1. `GOOGLE_GENAI_PYTHON_BIN` points to `.venv-google-genai/bin/python` which
   has `google-genai` installed.
2. `--max-tokens 1024` is not too small (sandwich prompts need ~800 tokens out).
3. The model name is exactly `gemini-2.5-flash` (not `gemini/2.5-flash`).

**`RuntimeError: No ready agents`**
The LLM returned a valid `EXECUTE` block but no agent name matched the env.
Sandwich uses `Chad` and `Dave` (or `Alice` and `Dave` depending on the env
version). Check the `initial_scene` field of a previous JSONL row to see the
actual agent names.

**Episode hangs**
The RRT planner can block on infeasible actions. `--max-sim-steps 5000`
(default) limits this. Reduce to `2000` for faster debugging at the cost of
more `NO_PROGRESS` failures.
