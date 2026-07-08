# Learned Subtask Skills — Tutorial

This tutorial explains the full pipeline for adding, training, and running **learned subtask skills** in RoCo. A "subtask skill" is a short, self-contained motion — PICK, PLACE, STACK_ON, OPEN_CABINET — executed by a single-arm policy that replaces the scripted RRT planner.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Collecting demonstration data](#2-collecting-demonstration-data)
3. [Training an ACT policy](#3-training-an-act-policy)
4. [Registering a trained policy](#4-registering-a-trained-policy)
5. [Runtime uncertainty estimation](#5-runtime-uncertainty-estimation)
6. [Running the demo](#6-running-the-demo)
7. [Writing a new skill](#7-writing-a-new-skill)
8. [File reference](#8-file-reference)

---

## 1. Architecture overview

```
LLM plan text
    │
    ▼
SkillPlan (SkillCall per agent)
    │
    ▼
SubtaskLearnedExecutor
    ├─ LearnedPolicyRegistry  → resolves (skill, agent, embodiment, task) → LearnedPolicySpec
    ├─ BoundedPolicyHandleCache  → loads & caches the policy checkpoint
    ├─ UncertaintyEstimator  → estimates confidence from chunk metadata
    ├─ SubtaskSuccessChecker  → declares skill done when condition is met
    └─ NoProgressMonitor  → aborts if the arm stalls
          │
          ▼ (per env step)
    LearnedPolicyHandle.predict_native_chunk(obs, instruction, low, high)
          │  NativeActionChunk(actions[chunk_size, n_ctrl], metadata)
          ▼
    SubtaskEnvAdapter.step(env, action, eq_active_idxs, eq_active_vals)
          │  SimAction → env.step() → MuJoCo physics
          ▼
    next_obs, reward, done, info
```

The key interface is `LearnedPolicyHandle` (defined in `rocobench/skills/learned/policy_handle.py`):

```python
class LearnedPolicyHandle:
    def reset(self) -> None: ...
    def predict_native_chunk(
        self,
        observation,              # current EnvState
        instruction: dict,        # {"canonical": "PICK(object=apple)", ...}
        action_low: np.ndarray,   # per-joint lower bound  shape (n_ctrl,)
        action_high: np.ndarray,  # per-joint upper bound  shape (n_ctrl,)
    ) -> NativeActionChunk:       # .actions shape (chunk_size, n_ctrl)
        ...
    def health_check(self, spec) -> dict: ...
```

---

## 2. Collecting demonstration data

Use the scripted RRT planner to generate (observation, action) pairs:

```bash
python scripts/collect_subtask_demos.py \
    --task pack \
    --num_episodes 100 \
    --output_dir data/subtask_demos \
    --skills PICK PLACE
```

Each episode is saved as:

```
data/subtask_demos/pack/PICK/episode_0000/
    observations.npz   # keys: ur5e_robotiq_ee_xpos, ur5e_robotiq_qpos, obj_apple_xpos, ...
    actions.npz        # key: ctrl  shape (T, 64)
    metadata.json      # skill_name, call_args, success, num_steps
```

Tip: run many episodes (`--num_episodes 500`) to cover diverse object placements. Only episodes where `success=True` are useful for training.

---

## 3. Training an ACT policy

RoCo uses [LeRobot](https://github.com/huggingface/lerobot) as the ACT training backend (optional dependency). The `collect_subtask_demos.py` output is compatible with the LeRobot HDF5 format after a one-step conversion:

```bash
# Convert to LeRobot format
python -m integrations.lerobot_roco.convert_demos \
    --input data/subtask_demos/pack/PICK \
    --output data/lerobot_datasets/pack_pick \
    --task pack --skill PICK

# Train
python -m lerobot.scripts.train \
    policy=act \
    dataset_repo_id=data/lerobot_datasets/pack_pick \
    training.num_epochs=200 \
    training.batch_size=8 \
    output_dir=checkpoints/pack_pick_v1
```

The resulting checkpoint is a `.pt` file at `checkpoints/pack_pick_v1/last.ckpt`.

> **Without LeRobot**: run the demo using the built-in `MockACTHandle` (see §6). It drives the arm with the same RRT planner used for demo collection — no neural network needed.

---

## 4. Registering a trained policy

Create a `LearnedPolicySpec` for each (skill, agent, embodiment) combination:

```python
from rocobench.skills.learned.registry import LearnedPolicyRegistry
from rocobench.skills.learned.models import LearnedPolicySpec

registry = LearnedPolicyRegistry()

spec = LearnedPolicySpec(
    policy_id="pack_PICK_Alice_v1",
    skill_name="PICK",
    agent_name="Alice",           # display name from the LLM plan
    embodiment_id="ur5e_robotiq", # MuJoCo body name (robot_name_map_inv["Alice"])
    task_id="pack",
    checkpoint="checkpoints/pack_pick_v1/last.ckpt",
    checkpoint_revision="v1",
    policy_type="act",
    schema_hash="abc123",         # hash of the obs/action schema used at training time
    action_representation="joint_ctrl",
    cameras=("wrist_cam", "overhead_cam"),
    max_steps=300,
    execution_horizon=10,         # how many actions to pop per chunk
    success_monitor="stable",
    failure_monitors=("NO_PROGRESS", "TIMEOUT"),
    enabled=True,
)

registry.register(spec)
```

Then create a matching `LearnedPolicyHandle` backed by LeRobot ACT:

```python
from integrations.lerobot_roco.evaluation.lerobot_handle import LeRobotACTHandle
from rocobench.skills.learned.policy_handle import BoundedPolicyHandleCache

cache = BoundedPolicyHandleCache(
    loader=lambda spec: LeRobotACTHandle(spec.checkpoint, chunk_size=spec.execution_horizon),
    max_size=8,
)
```

Pass both to `SubtaskLearnedExecutor`:

```python
from rocobench.skills.learned.subtask_executor import SubtaskLearnedExecutor
from rocobench.skills.learned.config import LearnedExecutorConfig

executor = SubtaskLearnedExecutor(
    env=env,
    robots=robots,
    policy_registry=registry,
    policy_cache=cache,
    uncertainty_mode="policy_metadata",  # or "heuristic", "ensemble_variance", "none"
    stable_success_checks=2,
    max_steps=300,
    config=LearnedExecutorConfig(task_id="pack"),
)
```

### Embodiment IDs

`embodiment_id` must match what `robot_name_map_inv` resolves the agent to:

| Task         | Agent name | Embodiment ID    |
|--------------|------------|------------------|
| pack / sort  | Alice      | ur5e_robotiq     |
| pack / sort  | Bob        | panda            |
| sandwich     | Chad       | ur5e_suction     |
| sandwich     | Dave       | humanoid         |
| cabinet      | Alice      | ur5e_robotiq     |

```python
name_map_inv = getattr(env, "robot_name_map_inv", {})
embodiment_id = name_map_inv.get(agent_name, agent_name)
```

---

## 5. Runtime uncertainty estimation

`UncertaintyEstimator` reads confidence information from the policy at inference time and classifies each chunk as `low / medium / high` risk.

### Modes

| Mode                | Source                               | Notes                                         |
|---------------------|--------------------------------------|-----------------------------------------------|
| `none`              | Always reports confidence = 1.0      | Disables monitoring; fastest                  |
| `heuristic`         | Action variance within the chunk     | No model changes needed; task-agnostic        |
| `policy_metadata`   | `chunk.metadata["confidence"]`       | Policy must fill `metadata` during inference  |
| `ensemble_variance` | Multiple forward passes              | Requires ensemble wrapper; most accurate      |

### Adding policy metadata (policy_metadata mode)

In your `LearnedPolicyHandle.predict_native_chunk()`, populate the `metadata` dict:

```python
def predict_native_chunk(self, obs, instruction, low, high) -> NativeActionChunk:
    actions, logits = self.model.forward(obs)   # your model call
    confidence = float(logits.softmax(-1).max())
    return NativeActionChunk(
        actions=actions,
        metadata={"confidence": confidence, "source": "act_logits"},
    )
```

The `UncertaintyEstimator` will read `metadata["confidence"]` and set `risk_level`:

- `< 0.6` → `high`
- `< 0.8` → `medium`
- `>= 0.8` → `low`

### Integrating with CRIE-BT

The CRIE-BT behavior tree reads uncertainty from `SubtaskLearnedExecutor` and can trigger fallback or re-planning:

```python
result = executor.execute(plan, obs, artifact_dir="artifacts/")
# result.metadata["uncertainty_summary"] contains:
# {n_chunks, mean_confidence, min_confidence, high_risk_chunks, ...}
```

A high `high_risk_chunks / n_chunks` ratio signals to the BT that this skill is unreliable and should be retried with a scripted fallback.

---

## 6. Running the demo

The demo script exercises the full pipeline without a real neural-network checkpoint by substituting `MockACTHandle` — an RRT-backed policy that uses the existing scripted planner internally.

```bash
# Single skill
python scripts/demo_learned_subtask.py \
    --task pack \
    --skill PICK \
    --object apple \
    --uncertainty_mode policy_metadata \
    --max_steps 150

# All skills in sequence
python scripts/demo_learned_subtask.py \
    --task pack \
    --run_all \
    --uncertainty_mode heuristic \
    --max_steps 150
```

Expected output (pack, PICK, apple):

```
============================================================
  Learned Subtask Skill Demo
  Task            : pack
  Uncertainty mode: policy_metadata
============================================================

  Robots available: ['Alice', 'Bob']

  ✓ PICK({'object': 'apple'})
     status       : success
     sim steps    : 115
     wall time    : 17.4s
     uncertainty  : mean_conf=0.500  min_conf=0.500  high_risk_chunks=0/12
     artifacts    : artifacts/subtask_demo/pack/PICK
```

Artefacts written under `--artifact_dir` (default `artifacts/subtask_demo`):

```
artifacts/subtask_demo/pack/PICK/
    skill_call.json          # the SkillCall that was executed
    instruction.json         # canonical instruction string
    policy_spec.json         # resolved LearnedPolicySpec
    checkpoint_metadata.json # from handle.health_check()
    policy_chunks.jsonl      # one JSON line per predicted chunk
    uncertainty_trace.jsonl  # one line per chunk: step, confidence, risk_level
    uncertainty_summary.json # aggregated uncertainty stats
```

### Supported tasks and skills

| Task       | Skills                               |
|------------|--------------------------------------|
| `pack`     | PICK, PLACE, PICK_AND_PLACE, WAIT    |
| `sandwich` | PICK, STACK_ON, WAIT                 |
| `cabinet`  | PICK, PLACE, OPEN_CABINET, WAIT      |
| `sort`     | PICK, PLACE, PICK_AND_PLACE, WAIT    |
| `sweep`    | SWEEP, WAIT                          |

---

## 7. Writing a new skill

### Step 1 — Define the skill name

Add a constant to `rocobench/skills/learned/subtask_skills.py`:

```python
SKILL_HANDOFF = "HANDOFF"
```

### Step 2 — Register it in a SkillRegistry

Add a `SkillSpec` to the relevant registry builder:

```python
registry.register(SkillSpec(
    name=SKILL_HANDOFF,
    description="Pass an object from one gripper to the other.",
    required_arguments=("object",),
    supported_agents=supported,
    resource_arguments=("object",),
))
```

### Step 3 — Add a success checker

Create a class in `rocobench/skills/learned/subtask_success.py`:

```python
class HandoffSuccessChecker(SubtaskSuccessChecker):
    def _once(self, env, obs, call_args):
        obj_name = call_args.get("object", "")
        # Check that receiver has the object and giver has released
        ...
        return received and released

# Register in _CHECKER_MAP:
_CHECKER_MAP["HANDOFF"] = HandoffSuccessChecker
```

### Step 4 — Collect demonstrations

```bash
python scripts/collect_subtask_demos.py \
    --task bimanual \
    --skills HANDOFF \
    --num_episodes 200
```

### Step 5 — Train and register

Follow §3 and §4 using `skill_name="HANDOFF"`.

---

## 8. File reference

| File | Purpose |
|------|---------|
| `rocobench/skills/learned/subtask_skills.py` | Skill registries per task; `build_registry_for_task()` |
| `rocobench/skills/learned/subtask_success.py` | Per-skill success checkers; `build_success_checker()` |
| `rocobench/skills/learned/subtask_executor.py` | `SubtaskLearnedExecutor` main loop; `SubtaskEnvAdapter` |
| `rocobench/skills/learned/mock_handle.py` | `MockACTHandle` — RRT-backed demo policy |
| `rocobench/skills/learned/registry.py` | `LearnedPolicyRegistry` — resolve spec from (skill, agent, embodiment, task) |
| `rocobench/skills/learned/policy_handle.py` | `LearnedPolicyHandle` ABC; `BoundedPolicyHandleCache` |
| `rocobench/skills/learned/models.py` | `LearnedPolicySpec`, `NativeActionChunk` |
| `rocobench/skills/learned/uncertainty.py` | `UncertaintyEstimator`, `UncertaintyState` |
| `rocobench/skills/learned/config.py` | `LearnedExecutorConfig` (no_progress params, task_id) |
| `scripts/collect_subtask_demos.py` | CLI to collect scripted demonstrations |
| `scripts/demo_learned_subtask.py` | CLI to demo the learned pipeline end-to-end |
| `tests/skills/test_subtask_pipeline.py` | 48 unit tests (no MuJoCo required) |
