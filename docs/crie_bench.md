# CRIE-Bench: Learned Skill Pipeline

CRIE-Bench is the unified command-line framework for the full learned-skill
lifecycle in the RoCo collaborative robotics codebase:

```
tasks / agents  →  collect  →  train  →  test  →  skills
```

All commands are available under `python -m crie_bench`.

---

## Table of contents

1. [Installation](#1-installation)
2. [Command reference](#2-command-reference)
3. [End-to-end: sandwich task with Octo](#3-end-to-end-sandwich-task-with-octo)
4. [Supported tasks and skills](#4-supported-tasks-and-skills)
5. [Octo integration details](#5-octo-integration-details)
6. [Adding a new task or model backend](#6-adding-a-new-task-or-model-backend)
7. [File reference](#7-file-reference)

---

## 1. Installation

### Runtime split

Use the Python 3.8 RoCo environment for simulator-backed collection and
rollout. Use the newer Octo/JAX environment for Octo conversion and
fine-tuning.

```bash
# RoCo simulator side
conda activate roco
pip install -e .

# Octo training side used by the current repo setup
.venv-google-genai/bin/python -m pip install -e .
```

### Optional: Octo VLA (for `--model octo`)

The verified path uses an upstream `octo-models/octo` checkout for
`scripts/finetune.py`, and the installed `octo` package in
`.venv-google-genai` for imports:

```bash
git clone https://github.com/octo-models/octo /tmp/octo-upstream
export OCTO_REPO_DIR=/tmp/octo-upstream

.venv-google-genai/bin/python -m pip install \
    tensorflow tensorflow-datasets rlds wandb gym plotly
```

### Optional: LeRobot ACT (for model=act)

```bash
pip install lerobot
```

---

## 2. Command reference

Run collection commands from the `roco` Python 3.8 environment. Run Octo
training commands from `.venv-google-genai` with `OCTO_REPO_DIR` pointing at an
upstream Octo checkout.

### `tasks` — list available environments

```bash
conda run -n roco python -m crie_bench tasks
```

Output:

```
====================================================================
  CRIE-Bench: available tasks
====================================================================

  sandwich
    Two-arm sandwich assembly: stack ingredients on a cutting board in recipe order.
    agents : Chad(ur5e_suction), Dave(humanoid)
    skills : PICK, STACK_ON, WAIT
    recipes: bacon, vegetarian, beef_patty, ham

  pack
    Pack groceries from a table into a container using two arms.
    agents : Alice(ur5e_robotiq), Bob(panda)
    skills : PICK, PLACE, PICK_AND_PLACE, WAIT
  ...
```

---

### `agents` — list robots for a task

```bash
conda run -n roco python -m crie_bench agents --task sandwich
```

Output:

```
============================================================
  Agents for task: sandwich
============================================================
  name         embodiment_id        description
  ------------ -------------------- ------------------------------
  Chad         ur5e_suction         UR5e with suction gripper
  Dave         humanoid             Humanoid torso
```

---

### `collect` — gather scripted (RRT) demonstrations

```bash
conda run -n roco python -m crie_bench collect \
    --task sandwich \
    --skills PICK STACK_ON \
    --num-episodes 200 \
    --output-dir data/subtask_demos
```

Each successful episode is saved under:

```
data/subtask_demos/sandwich/PICK/episode_0000/
    observations.npz   # ur5e_suction_qpos, ee_xpos, obj_*_xpos, ...
    actions.npz        # ctrl  shape (T, 64)
    metadata.json      # skill_name, call_args, success, num_steps
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | required | Task name |
| `--skills` | task default | Space-separated skill names |
| `--num-episodes` | 100 | How many rollout episodes |
| `--output-dir` | `data/subtask_demos` | Root output directory |
| `--render` | off | Enable MuJoCo OpenGL window |
| `--seed` | 0 | Random seed |

---

### `train` — fine-tune a learned policy

```bash
env OCTO_REPO_DIR=/tmp/octo-upstream .venv-google-genai/bin/python -m crie_bench train \
    --task sandwich \
    --skill PICK \
    --agent Chad \
    --model octo \
    --num-steps 20000 \
    --batch-size 128 \
    --output-dir checkpoints/octo
```

The train command automatically:
1. Converts demos at `data/subtask_demos/sandwich/PICK/` → a TFDS/RLDS builder dataset.
2. Writes an Octo fine-tune config file.
3. Launches upstream `octo-models/octo/scripts/finetune.py` as a subprocess.
4. Writes a `run_manifest.json` with provenance.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | required | Task name |
| `--skill` | required | Skill to train (e.g. `PICK`) |
| `--agent` | required | Agent name (e.g. `Chad`) |
| `--model` | `octo` | Model backend: `act` or `octo` |
| `--data-dir` | `data/subtask_demos` | Demo root |
| `--output-dir` | `checkpoints/octo` | Checkpoint root |
| `--pretrained` | `hf://rail-berkeley/octo-small` | Pre-trained checkpoint |
| `--num-steps` | 20000 | Fine-tuning steps |
| `--batch-size` | 128 | Training batch size |
| `--chunk-size` | 10 | Action prediction horizon |
| `--dry-run` | off | Print command without running |

---

### `test` — evaluate a trained policy

```bash
conda run -n roco python -m crie_bench test \
    --task sandwich \
    --skill PICK \
    --agent Chad \
    --checkpoint checkpoints/octo/octo_sandwich_pick_chad \
    --model octo \
    --num-episodes 20
```

Writes per-episode results and a `eval_summary.json` to `--artifact-dir`.

Use `--model mock` for a no-checkpoint smoke test using the RRT-backed
MockACTHandle:

```bash
conda run -n roco python -m crie_bench test \
    --task sandwich --skill PICK --agent Chad \
    --checkpoint "" --model mock \
    --num-episodes 1 --max-steps 20 --no-video \
    --artifact-dir /tmp/roco_doc_eval_mock
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to trained checkpoint dir |
| `--model` | `octo` | `act`, `octo`, or `mock` |
| `--uncertainty-mode` | `heuristic` | `none`, `heuristic`, `policy_metadata` |
| `--num-episodes` | 20 | Rollout episodes |
| `--max-steps` | 300 | Max env steps per episode |
| `--artifact-dir` | `artifacts/eval` | Where to save results |

---

### `skills` — view trained skill inventory

```bash
conda run -n roco python -m crie_bench skills             # all tasks
conda run -n roco python -m crie_bench skills --task sandwich
```

Output:

```
======================================================================
  Trained skill inventory — task: sandwich
======================================================================

  Task: sandwich
  policy_id                           skill          agent    type   ckpt  enabled
  ----------------------------------- -------------- -------- ------ ----  -------
  mock_sandwich_pick_chad             PICK           Chad     mock     OK  yes
  mock_sandwich_stack_on_dave         STACK_ON       Dave     mock     OK  yes
  octo_sandwich_pick_chad_v1          PICK           Chad     octo     OK  yes
  octo_sandwich_stack_on_dave_v1      STACK_ON       Dave     octo MISSING  no
```

`MISSING` means the checkpoint path in the YAML doesn't exist yet. Run
`crie_bench train` to generate it, update the `checkpoint` path if needed, then
flip `enabled: true` in `configs/skills/learned_sandwich_octo.yaml`.

---

## 3. End-to-end: sandwich task with Octo

### Step 0 — check available tasks and agents

```bash
conda run -n roco python -m crie_bench tasks
conda run -n roco python -m crie_bench agents --task sandwich
```

### Step 1 — collect demonstrations

```bash
conda run -n roco python -m crie_bench collect \
    --task sandwich \
    --skills PICK STACK_ON \
    --num-episodes 200 \
    --seed 0
```

Target: ≥ 100 successful episodes per skill for reliable fine-tuning.

### Step 2 — train PICK policy for Chad

```bash
env OCTO_REPO_DIR=/tmp/octo-upstream .venv-google-genai/bin/python -m crie_bench train \
    --task sandwich --skill PICK --agent Chad --model octo \
    --pretrained hf://rail-berkeley/octo-small \
    --num-steps 20000 --batch-size 128 --chunk-size 10
```

Checkpoint written to: `checkpoints/octo/octo_sandwich_pick_chad/`

### Step 3 — train STACK_ON policy for Dave

```bash
env OCTO_REPO_DIR=/tmp/octo-upstream .venv-google-genai/bin/python -m crie_bench train \
    --task sandwich --skill STACK_ON --agent Dave --model octo \
    --num-steps 30000 --batch-size 128
```

Checkpoint written to: `checkpoints/octo/octo_sandwich_stack_on_dave/`

### Step 4 — test PICK policy

```bash
conda run -n roco python -m crie_bench test \
    --task sandwich --skill PICK --agent Chad \
    --checkpoint checkpoints/octo/octo_sandwich_pick_chad \
    --model octo --num-episodes 20
```

Expected output (after successful training):

```
  ep 000  OK  steps=127  conf=0.84  risk=0/13
  ep 001  OK  steps=115  conf=0.81  risk=1/12
  ...
  Success rate : 85.0%  (17/20)
  Mean steps   : 118.3
  Summary saved: artifacts/eval/eval_summary.json
```

### Step 5 — enable in skill registry

Edit `configs/skills/learned_sandwich_octo.yaml`:

```yaml
- policy_id: octo_sandwich_pick_chad_v1
  ...
  enabled: true    # flip this once checkpoint is verified
```

Then verify:

```bash
conda run -n roco python -m crie_bench skills --task sandwich
```

### Using a pre-built training config file

Instead of passing all flags inline, use the pre-written YAML configs:

```bash
# Dry-run to preview the training command
env OCTO_REPO_DIR=/tmp/octo-upstream .venv-google-genai/bin/python - <<'PY'
from dataclasses import replace
from integrations.octo_roco.training.config import load_octo_training_config
from integrations.octo_roco.training.launch import launch_octo_training

cfg = load_octo_training_config("configs/training/sandwich_octo_pick_chad.yaml")
cfg = replace(cfg, dry_run=True)
launch_octo_training(cfg)
PY
```

Or load and launch from Python:

```python
from integrations.octo_roco.training.config import load_octo_training_config
from integrations.octo_roco.training.launch import launch_octo_training

cfg = load_octo_training_config("configs/training/sandwich_octo_pick_chad.yaml")
result = launch_octo_training(cfg)
print(result.checkpoint_dir)
```

### Verified one-step Octo smoke command

This command was run successfully in the current environment and exercises the
real TFDS/RLDS conversion plus upstream Octo finetune entry point:

```bash
OUT=/tmp/roco_octo_upstream_train_smoke_$(date +%Y%m%d_%H%M%S)
env OCTO_REPO_DIR=/tmp/octo-upstream .venv-google-genai/bin/python -m crie_bench train \
    --task sandwich --skill PICK --agent Chad --model octo \
    --data-dir data/subtask_demos_v2 \
    --output-dir "$OUT" \
    --num-steps 1 --batch-size 1 --chunk-size 4 \
    --save-interval 1 --eval-interval 1000 --verbose
```

Expected manifest fields:

```json
{
  "status": "completed",
  "dataset": {
    "backend": "tfds_rlds",
    "num_train_episodes": 45,
    "num_val_episodes": 5,
    "total_steps": 209
  }
}
```

---

## 4. Supported tasks and skills

| Task | Agents (embodiment) | Skills | Recipes |
|------|---------------------|--------|---------|
| `sandwich` | Chad (ur5e_suction), Dave (humanoid) | PICK, STACK_ON, WAIT | bacon, vegetarian, beef_patty, ham |
| `pack` | Alice (ur5e_robotiq), Bob (panda) | PICK, PLACE, PICK_AND_PLACE, WAIT | — |
| `cabinet` | Alice (ur5e_robotiq), Bob (panda) | PICK, PLACE, OPEN_CABINET, WAIT | — |
| `sort` | Alice (ur5e_robotiq), Bob (panda) | PICK, PLACE, WAIT | — |
| `rope` | Alice (ur5e_robotiq), Bob (panda) | PICK, PLACE, WAIT | — |
| `sweep` | Alice (ur5e_robotiq) | SWEEP, WAIT | — |

---

## 5. Octo integration details

### How a single Octo model generalises across all sandwich subtasks

The sandwich subtasks (PICK tomato, PICK cheese, STACK_ON bread_slice1, …)
differ only in **which object/target** is named.  Octo conditions on a plain
text instruction (`"PICK(object=tomato)"`) via a T5 language encoder.  Because
the instruction carries the object name as a token, the same model weights
generalise across all ingredient variants — including new recipes not seen
at training time, as long as the visual appearance is covered.

This is fundamentally different from ACT + task-ID conditioning, where a
discrete embedding gives the model no signal that
`PICK(tomato)` and `PICK(cheese)` share the same motor primitive.

### OctoHandle internals

`integrations/octo_roco/handle.py` implements `LearnedPolicyHandle`:

```python
from integrations.octo_roco.handle import OctoHandle
handle = OctoHandle.from_checkpoint("checkpoints/octo/octo_sandwich_pick_chad", spec)
chunk = handle.predict_native_chunk(obs, instruction, action_low, action_high)
# chunk.actions  shape (chunk_size, n_ctrl)
# chunk.metadata["confidence"]  float in (0, 1)
```

The handle maintains a rolling observation window of `window_size=2` frames,
resizes camera images to 256×256, and extracts a flat proprioception vector
from the active agent's `qpos + ee_xpos + ee_xquat`.

### Dataset conversion

`integrations/octo_roco/dataset_converter.py` converts the NPZ demos to RLDS:

```python
from integrations.octo_roco.dataset_converter import convert_demos
result = convert_demos(
    data_root="data/subtask_demos/sandwich/PICK",
    output_dir="data/rlds/sandwich_pick",
    task_id="sandwich",
    skill_id="PICK",
    agent_name="Chad",
    language_instruction="PICK(object=ingredient)",
)
# result.backend = "tfds_rlds"  # real TFDS/RLDS builder dataset
# or "numpy_fallback" when TensorFlow / TFDS are not installed
```

### Confidence estimation

`OctoHandle.predict_native_chunk()` fills `chunk.metadata["confidence"]` from
the variance of the predicted action chunk: `exp(-mean(var(actions)))`.
Low-variance chunks → high confidence.  This integrates directly with
`UncertaintyEstimator` in `policy_metadata` mode for CRIE-BT fallback.

---

## 6. Adding a new task or model backend

### New task

1. Add a `TaskMeta` entry to `crie_bench/registry.py`:

```python
"mytask": TaskMeta(
    task_id="mytask",
    class_path="rocobench.envs.task_mytask:MyTask",
    description="...",
    agents=(AgentMeta("Alice", "ur5e_robotiq", "..."),),
    skills=("PICK", "PLACE", "WAIT"),
),
```

2. Add or confirm the derived `DEFAULT_SKILLS_PER_TASK` entry in
   `crie_bench/registry.py`.
3. Add a `configs/skills/learned_mytask_octo.yaml` (copy from sandwich).

### New model backend

1. Implement `YourHandle(LearnedPolicyHandle)` in
   `integrations/your_model/handle.py`.
2. Add a branch in `crie_bench/train.py::train()` and
   `crie_bench/evaluate.py::_build_policy_handle()`.
3. Add the model name to `SUPPORTED_MODELS` in `crie_bench/train.py`.

---

## 7. File reference

### `crie_bench/`

| File | Purpose |
|------|---------|
| `cli.py` | Argparse CLI: tasks, agents, collect, train, test, skills |
| `registry.py` | Static task/agent metadata; `scan_trained_skills()` |
| `collect.py` | `CollectConfig` + `collect()` — scripted demo collection |
| `train.py` | `TrainConfig` + `train()` — dispatch to ACT or Octo backends |
| `evaluate.py` | `EvalConfig` + `evaluate()` — policy rollout + metrics |
| `skills_view.py` | `print_skills_table()` — checkpoint inventory display |
| `__main__.py` | `python -m crie_bench` entry point |

### `integrations/octo_roco/`

| File | Purpose |
|------|---------|
| `handle.py` | `OctoHandle(LearnedPolicyHandle)` — Octo inference wrapper |
| `dataset_converter.py` | `RoCoToRLDSConverter` — NPZ → TFDS/RLDS builder dataset |
| `training/config.py` | `OctoTrainingConfig` + preset factory functions |
| `training/launch.py` | `launch_octo_training()` — dataset convert + finetune |
| `evaluation/config.py` | `OctoEvalConfig` for rollout evaluation |

### Config files

| File | Purpose |
|------|---------|
| `configs/training/sandwich_octo_pick_chad.yaml` | Octo training config: PICK / Chad |
| `configs/training/sandwich_octo_stack_on_dave.yaml` | Octo training config: STACK_ON / Dave |
| `configs/skills/learned_sandwich_octo.yaml` | Trained skill registry for sandwich + Octo |
