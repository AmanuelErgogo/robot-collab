# Phase 2 Implementation Package
## RRT Expert Demonstration Collection and LeRobotDataset v3 Generation

**Repository:** `AmanuelErgogo/robot-collab`  
**Primary task:** `PackGroceryTask`  
**Primary skill:** `PUT_OBJECT_IN_CONTAINER(object, container)`  
**Expert backend:** Phase 1 `RRTSkillExecutor` / existing `PlannedPathPolicy` and `MultiArmRRT`  
**Simulation bridge:** Phase 0 `RoCoGymEnv` and RoCo bridge  
**Dataset runtime:** Python 3.12+, LeRobot 0.5.2 / LeRobotDataset v3  
**RoCo runtime:** Python 3.8, MuJoCo 2.3.0, dm_control 1.0.8  
**Phase objective:** Generate, validate, save, resume, replay, and inspect high-quality RRT expert demonstrations in the current LeRobotDataset format without introducing learned-policy training.

---

# 1. Executive Summary

Phase 2 turns successful RoCo RRT skill executions into a training-ready LeRobotDataset.

The required end-to-end path is:

```text
Deterministic episode specification
    -> Phase 0 RoCo bridge reset
    -> Phase 1 typed SkillPlan
    -> semantic validation
    -> RRT skill preparation
    -> existing PlannedPathPolicy action buffer
    -> project every expert SimAction into the public Gym action space
    -> execute through the normal RoCoGymEnv.step(action) path
    -> record observation_t, action_t, reward/done/success
    -> save only accepted successful episodes
    -> LeRobotDataset.save_episode()
    -> LeRobotDataset.finalize()
    -> reopen and validate
    -> replay selected episodes through RoCoGymEnv
```

The dataset writer must run in the Python 3.12 LeRobot client runtime. The RRT expert planner and MuJoCo simulator remain in the Python 3.8 RoCo server runtime.

The main dataset features are:

```text
observation.images.front
observation.images.active_agent
observation.state
action
next.reward
next.done
next.success
task
```

Each saved episode represents one execution of one skill:

```text
PUT_OBJECT_IN_CONTAINER(object=<item>, container=<bin_slot>)
```

The other robot executes `WAIT()` and remains safely held.

The dataset must store exactly the action representation that a future learned policy will produce and that `RoCoGymEnv.step()` accepts. Do not store full privileged `SimAction` internals, object-specific weld IDs, RRT waypoints, target object IDs, or equality-constraint indices inside the `action` label.

Privileged expert information may be stored only in diagnostic sidecar metadata, never as an input or output feature used by the learned policy.

---

# 2. Position in the Overall Roadmap

```text
Phase 0
    RoCoBench <-> Gymnasium <-> LeRobot-compatible bridge

Phase 1
    SkillPlan + semantic validation + RRT-backed SkillExecutor

Phase 2
    RRT expert rollouts -> LeRobotDataset v3      <-- this specification

Phase 3
    Train and deliberately overfit one ACT skill

Phase 4
    Run learned-policy inference in RoCoBench

Phase 5
    Replace online RRT with LearnedSkillExecutor

Phase 6
    Multi-agent learned-skill scheduling and concurrency
```

Phase 2 must not implement policy training or inference. It should leave a clean dataset that Phase 3 can consume directly.

---

# 3. Phase 2 Readiness Gates

Codex must audit these prerequisites before implementation.

## 3.1 Phase 0 readiness

The repository should already provide:

```text
RoCoBridgeServer
RoCoGymEnv
versioned localhost protocol
GET_SPEC
RESET
STEP
RENDER
GET_STATE_DIGEST
fixed observation and action spaces
public action -> SimAction adapter
Python 3.8 server / Python 3.12 client separation
```

Required Phase 0 behavior:

- `RoCoGymEnv.reset(seed=...)` is deterministic.
- `RoCoGymEnv.step(action)` accepts the public fixed-shape action.
- Images and state have fixed shapes.
- The action interface can perform complete gripper open/close behavior.
- Passive-agent hold is implemented.
- Direct and bridged low-level actions have an equivalence test.

## 3.2 Phase 1 readiness

The repository should already provide:

```text
SkillSpec
SkillCall
SkillPlan
SkillRegistry
PackGrocerySkillPlanValidator
RRTSkillCompiler
SkillFeedbackManager
SkillExecutor
RRTSkillExecutor
PUT_OBJECT_IN_CONTAINER
WAIT
```

Required Phase 1 behavior:

- A skill plan can be constructed without an LLM.
- It can be semantically validated.
- It can be compiled into existing `LLMPathPlan` objects.
- `PlannedPathPolicy.plan(env)` can produce `action_buffer`.
- Skill-level outcome can be evaluated separately from full-task completion.

## 3.3 Mandatory action-representability gate

Before saving any data, prove:

```text
RRT SimAction
    -> ExpertActionProjector
    -> public action vector
    -> Phase 0 action adapter
    -> reconstructed SimAction behavior
```

is equivalent for:

- ordinary joint motion;
- gripper close;
- object attachment/grasp;
- object transport;
- gripper open;
- object release;
- return-home movement;
- passive-agent hold.

If RRT execution requires hidden equality-constraint changes that cannot be inferred from the public action and current simulator state, do not record the trajectory.

Implement or complete a deterministic server-side gripper/grasp state machine used by both:

```text
expert projected actions
future learned-policy actions
```

The learned action must not contain:

```text
object name
object ID
equality-constraint index
RRT waypoint
privileged grasp target
```

Collection must fail with:

```text
UNREPRESENTABLE_EXPERT_ACTION
```

rather than silently producing labels that cannot be replayed at inference time.

## 3.4 Audit outcome

Codex must produce a readiness report before editing:

```text
PASS
PARTIAL
MISSING
```

for each prerequisite.

If an API exists under a different name, reuse it rather than adding a duplicate.

If a small missing prerequisite is required for Phase 2, implement the minimum compatible extension and document it. Do not reimplement all of Phase 0 or Phase 1.

---

# 4. Goals

## 4.1 Functional goals

1. Deterministically define a sequence of expert skill episodes.
2. Reset RoCoBench with a known seed.
3. Construct the skill plan without calling an LLM.
4. Produce an RRT expert trajectory without executing it immediately.
5. Project the RRT action buffer to the public Gym action representation.
6. Execute the projected actions through the ordinary Phase 0 `STEP` path.
7. Record aligned `(observation_t, action_t)` pairs.
8. Record reward, done, and skill success generated after `action_t`.
9. Save only valid accepted episodes by default.
10. Discard failed attempts cleanly.
11. Create a LeRobotDataset v3 dataset.
12. Finalize the dataset correctly.
13. Reopen it locally after finalization.
14. Validate videos, state, actions, metadata, statistics, and task descriptions.
15. Resume collection safely into an existing compatible dataset.
16. Replay selected episodes through RoCoGymEnv.
17. Produce a human-readable collection and quality report.
18. Keep Hub upload optional and disabled by default.
19. Require no external LLM and no API key.

## 4.2 Dataset-quality goals

1. Each episode has a clear, specific natural-language task.
2. Observation and action representations are identical to future inference interfaces.
3. Frames are temporally aligned.
4. All saved episodes satisfy the skill success predicate.
5. Episode seeds and task arguments are recoverable.
6. Failed attempts do not contaminate the main behavior-cloning dataset.
7. Object, target, seed, and agent sampling can be balanced.
8. No NaN, Inf, shape drift, image corruption, or out-of-range action exists.
9. Dataset splits avoid seed leakage.
10. Selected episodes replay successfully.

## 4.3 Engineering goals

1. LeRobot imports remain exclusively in the Python 3.12 client.
2. RRT and MuJoCo remain exclusively in the Python 3.8 server.
3. Dataset code is separated from bridge and planner code.
4. LeRobot-version-specific calls are isolated.
5. Collection is transaction-like at the episode level.
6. Collection can be interrupted and resumed.
7. Dataset schema is versioned and hashed.
8. All testable logic has unit tests.
9. Simulator and video tests are clearly marked.
10. Existing Phase 0, Phase 1, and legacy RoCo behavior remains intact.

---

# 5. Non-Goals

Do not implement:

- ACT training.
- Diffusion Policy training.
- SmolVLA training.
- Learned-policy inference.
- Online replacement of RRT.
- LLM-generated training episodes.
- Human teleoperation.
- Human correction data.
- Failed/recovery demonstration training.
- Simultaneous learned control of both robots.
- A joint multi-arm policy.
- MetaWorld.
- Migration of RoCo to a new MuJoCo version.
- Native changes inside the upstream LeRobot package.
- Automatic public Hub publishing.
- Large-scale distributed collection.
- SLURM orchestration.
- Image augmentation during recording.
- Reward shaping.
- A new motion planner.
- Object-specific privileged fields in policy observations/actions.

Phase 2 may store failed-attempt metadata for future research, but failed demonstrations must not be included in the primary dataset unless explicitly requested through a separate future dataset mode.

---

# 6. Official LeRobotDataset Contract to Target

Target the currently validated LeRobot release and isolate API differences.

At the time of this specification, the official package metadata reports:

```text
LeRobot 0.5.2
Python >= 3.12
LeRobotDataset v3
```

The current public dataset flow is:

```python
dataset = LeRobotDataset.create(
    repo_id=...,
    fps=...,
    features=...,
    root=...,
    robot_type=...,
    use_videos=True,
)

dataset.add_frame(frame)
dataset.save_episode()
dataset.finalize()
```

Important requirements:

- `add_frame(frame)` requires a `task` entry.
- `save_episode()` commits the current episode.
- `clear_episode_buffer(delete_images=True)` discards an attempt.
- `resume(...)` appends to an existing local dataset and requires an explicit root.
- `finalize()` must be called after recording; otherwise Parquet writers may not have valid footers and the dataset may be unreadable.
- Dataset v3 stores tabular data in Parquet, visual data in MP4 shards, and episode/task metadata separately.
- Standard constants use:
  ```text
  action
  next.reward
  next.done
  observation.state
  observation.images.<camera>
  ```
- The writer automatically maintains:
  ```text
  timestamp
  frame_index
  episode_index
  index
  task_index
  ```

Do not manually add those default index features.

All direct LeRobot calls must live behind a `LeRobotDatasetAdapter` so later API changes affect one module.

---

# 7. Required Architecture

## 7.1 End-to-end architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ Python 3.12 LeRobot collection process                       │
│                                                              │
│ EpisodeSampler                                               │
│      ↓                                                       │
│ RoCoGymEnv.reset(seed)                                       │
│      ↓                                                       │
│ ExpertServiceClient.prepare(skill)                           │
│      ↓                                                       │
│ loop:                                                        │
│   expert_action = peek_expert_action()                       │
│   frame uses observation_t + expert_action_t                 │
│   observation_t+1 = env.step(expert_action_t)                │
│   episode transaction adds frame                             │
│      ↓                                                       │
│ quality gate                                                 │
│      ↓                                                       │
│ LeRobotDataset.add_frame / save_episode / finalize           │
└─────────────────────────────┬────────────────────────────────┘
                              │ versioned Phase 0 RPC
┌─────────────────────────────▼────────────────────────────────┐
│ Python 3.8 RoCo server                                       │
│                                                              │
│ PackGroceryTask                                              │
│ SkillPlan validator                                          │
│ RRTSkillCompiler                                             │
│ PlannedPathPolicy                                            │
│ MultiArmRRT                                                  │
│ ExpertActionProjector                                        │
│ SkillOutcomeEvaluator                                        │
│ public action adapter -> SimAction -> env.step               │
└──────────────────────────────────────────────────────────────┘
```

## 7.2 Core invariant

All demonstrations must execute through:

```python
RoCoGymEnv.step(public_action)
```

Do not execute the RRT `SimAction` directly during the recorded rollout and then store a separately projected label.

That would risk recording an action that did not actually produce the transition.

The executed action and saved action must be byte-equivalent after canonical float32 conversion.

---

# 8. Proposed Repository Layout

Adapt to existing Phase 0/Phase 1 paths where necessary.

```text
integrations/
└── lerobot_roco/
    ├── common/
    │   ├── expert_protocol.py
    │   ├── dataset_contract.py
    │   └── versioning.py
    ├── roco_runtime/
    │   ├── expert_service.py
    │   ├── expert_session.py
    │   ├── expert_action_projector.py
    │   ├── grasp_controller.py
    │   ├── skill_outcome.py
    │   └── trajectory_validation.py
    └── client/
        └── src/
            └── lerobot_roco_env/
                ├── dataset/
                │   ├── __init__.py
                │   ├── config.py
                │   ├── schema.py
                │   ├── lerobot_adapter.py
                │   ├── frame_adapter.py
                │   ├── episode_sampler.py
                │   ├── episode_transaction.py
                │   ├── collector.py
                │   ├── manifest.py
                │   ├── quality.py
                │   ├── validation.py
                │   ├── replay.py
                │   ├── splits.py
                │   └── report.py
                └── expert_client.py

scripts/
├── collect_roco_expert_dataset.py
├── validate_roco_dataset.py
├── replay_roco_dataset_episode.py
├── inspect_roco_dataset.py
└── smoke_test_roco_dataset_collection.py

tests/
└── lerobot_roco/
    ├── dataset/
    │   ├── test_schema.py
    │   ├── test_frame_adapter.py
    │   ├── test_episode_sampler.py
    │   ├── test_episode_transaction.py
    │   ├── test_manifest.py
    │   ├── test_quality.py
    │   ├── test_splits.py
    │   └── test_resume.py
    ├── expert/
    │   ├── test_expert_protocol.py
    │   ├── test_expert_action_projector.py
    │   ├── test_expert_session.py
    │   ├── test_grasp_controller.py
    │   └── test_skill_outcome.py
    └── integration/
        ├── test_collect_one_episode.py
        ├── test_failed_episode_is_discarded.py
        ├── test_dataset_finalize_and_load.py
        ├── test_dataset_video_decode.py
        ├── test_dataset_resume.py
        ├── test_dataset_replay.py
        ├── test_expert_projection_equivalence.py
        └── test_collection_determinism.py

docs/
└── phase2_expert_dataset_collection.md

PHASE2_CODEX_IMPLEMENTATION_SPEC.md
```

Do not create parallel duplicate versions of Phase 0 client/server classes. Extend the existing bridge cleanly.

---

# 9. Dependency Direction

```text
common expert protocol:
    no MuJoCo
    no Gymnasium
    no LeRobot
    Python 3.8 compatible

roco_runtime expert service:
    may import rocobench
    may import Phase 1 skill modules
    no LeRobot
    no Gymnasium dependency required

client dataset package:
    may import RoCoGymEnv
    may import LeRobotDataset
    must not import rocobench, dm_control, or MuJoCo

scripts:
    compose public package APIs
    no duplicated business logic
```

---

# 10. Episode Definition

## 10.1 One episode equals one skill

An episode begins immediately before execution of:

```text
PUT_OBJECT_IN_CONTAINER(object=<object>, container=<container>)
```

and ends when one of these occurs:

```text
SKILL_SUCCESS
RRT_PLAN_FAILED
EXECUTION_FAILED
TIMEOUT
UNREPRESENTABLE_EXPERT_ACTION
INVALID_INITIAL_STATE
QUALITY_REJECTED
BRIDGE_ERROR
```

Only `SKILL_SUCCESS` episodes are saved by default.

## 10.2 Initial supported plan

For active Alice:

```text
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

For active Bob:

```text
NAME Alice ACTION WAIT()
NAME Bob ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
```

The initial collector should default to one agent, preferably Alice, but support either agent through configuration.

## 10.3 Full-task success is not the episode predicate

`PackGroceryTask.get_reward_done()` reports success only when all grocery objects are packed.

Phase 2 episodes represent one subtask, so define:

```python
SkillOutcomeEvaluator.is_success(
    skill_call,
    observation,
) -> bool
```

For `PUT_OBJECT_IN_CONTAINER`:

- requested object is inside the requested target container/slot according to the same task helper used by Phase 1;
- object is released;
- object is stable for a configurable number of environment steps;
- no contradictory held-object state remains.

Do not duplicate or contradict Phase 1 occupancy logic.

Expose:

```text
info["is_success"]       -> full task success from Phase 0
info["skill_success"]    -> current skill success
```

Dataset feature `next.success` uses `skill_success`.

## 10.4 Stabilization tail

After the expert action buffer is exhausted, optionally execute a bounded number of safe hold actions:

```text
settle_steps = 3 to 10, configurable
```

Purpose:

- allow released object to settle;
- detect immediate slippage;
- verify stable placement.

Do not record settling frames in the primary demonstration by default.

If settling is recorded, document and mark it consistently.

---

# 11. Episode Specification Model

```python
@dataclass(frozen=True)
class ExpertEpisodeSpec:
    attempt_index: int
    seed: int
    active_agent: str
    skill_name: str
    object_name: str
    container_name: str
    task_instruction: str
```

Canonical task instruction:

```text
Put the apple in bin_front_left.
```

Optionally support a human-readable mapping:

```text
Put the apple in the front-left bin slot.
```

The mapping must be deterministic and tested.

Provide:

```python
to_dict()
canonical_id()
```

Suggested canonical ID:

```text
pack__alice__put_object_in_container__apple__bin_front_left__seed_000123
```

---

# 12. Episode Sampling

Implement a deterministic `EpisodeSampler`.

## 12.1 Inputs

```python
objects
containers
active_agents
seed_start or explicit seed list
sampling_seed
num_successful_episodes
max_attempts
balance_mode
```

## 12.2 Sampling modes

Support:

```text
round_robin
shuffled_cartesian
random
```

Default:

```text
shuffled_cartesian
```

This should balance:

- object;
- target container;
- active agent, if collecting multiple agents.

## 12.3 Initial-state validity

After reset, reject before planning if:

- object missing;
- object already packed;
- target occupied;
- object held;
- target invalid;
- object unreachable according to existing semantic checks;
- requested plan cannot make progress.

Invalid resets count as attempts but not dataset episodes.

## 12.4 Reproducibility

Given the same:

```text
collection config
sampler seed
environment version
```

the ordered episode specifications must be identical.

Save the complete planned specification list before collection begins:

```text
roco_meta/planned_episodes.jsonl
```

---

# 13. Expert Service Protocol

Extend the existing Phase 0 protocol; do not create a second socket.

## 13.1 New capabilities

Add capability flags:

```text
expert_prepare
expert_peek_action
expert_status
expert_cancel
skill_outcome
```

## 13.2 Commands

Implement:

```text
EXPERT_PREPARE
EXPERT_PEEK_ACTION
EXPERT_STATUS
EXPERT_CANCEL
```

Normal action execution remains the existing:

```text
STEP
```

## 13.3 `EXPERT_PREPARE`

Request:

```python
{
    "episode_id": "...",
    "skill_call": {
        "agent_name": "Alice",
        "skill_name": "PUT_OBJECT_IN_CONTAINER",
        "arguments": {
            "object": "apple",
            "container": "bin_front_left"
        }
    },
    "passive_calls": [
        {
            "agent_name": "Bob",
            "skill_name": "WAIT",
            "arguments": {}
        }
    ]
}
```

Server flow:

1. validate current episode;
2. create `SkillPlan`;
3. perform Phase 1 semantic validation;
4. compile to legacy plans;
5. run geometric feedback;
6. instantiate `PlannedPathPolicy`;
7. call `plan(env)` without executing;
8. collect complete expert `SimAction` buffer;
9. project every action into public action space;
10. validate representability;
11. optionally dry-run equivalence on saved simulator state;
12. restore original state;
13. create expert session.

Response:

```python
{
    "expert_session_id": "...",
    "plan_id": "...",
    "skill_call": {...},
    "num_actions": 143,
    "projected_action_shape": [8],
    "projection_report": {...},
    "rrt_metadata": {...},
    "current_state_digest": "...",
}
```

## 13.4 `EXPERT_PEEK_ACTION`

This command must not advance the simulator.

Request:

```python
{
    "episode_id": "...",
    "expert_session_id": "...",
    "expert_step_index": 0
}
```

Response:

```python
{
    "expert_step_index": 0,
    "action": np.ndarray(dtype=float32),
    "action_sha256": "...",
    "current_state_digest": "...",
    "phase": "approach|grasp|transport|release|home|unknown",
    "is_last_action": False
}
```

The phase label is diagnostic sidecar metadata only.

## 13.5 Expert-aware `STEP`

The client calls the ordinary `STEP` command with optional expert fields:

```python
{
    "episode_id": "...",
    "step_index": 0,
    "action": ...,
    "expert_session_id": "...",
    "expert_step_index": 0,
    "expert_action_sha256": "..."
}
```

Server validates:

- expert session matches;
- step index matches;
- action hash matches;
- numeric action equals expected within strict tolerance;
- current state digest matches the state used for `EXPERT_PEEK_ACTION`.

Then execute through the normal public action adapter.

After step:

- increment expert cursor;
- evaluate skill outcome;
- report:
  ```text
  expert_exhausted
  skill_success
  expert_step_index
  ```

## 13.6 `EXPERT_STATUS`

Return:

```python
{
    "prepared": True,
    "cursor": 42,
    "num_actions": 143,
    "exhausted": False,
    "skill_success": False,
    "failure": None,
}
```

## 13.7 `EXPERT_CANCEL`

Clear the current expert session without resetting the environment.

Must be idempotent.

## 13.8 State machine

```text
NO_EXPERT
    -> PREPARED
    -> EXECUTING
    -> EXHAUSTED
    -> SUCCEEDED or FAILED
    -> CANCELLED
```

Reject:

- prepare before environment reset;
- second expert prepare while active;
- peek after cancellation;
- step without matching peek/hash;
- stale expert index;
- expert action different from saved action.

---

# 14. Expert Action Projection

## 14.1 Purpose

Convert existing RRT-generated `SimAction` into the same fixed action vector used by Phase 0 and future learned policies.

```python
class ExpertActionProjector:
    def project(
        self,
        sim_action: SimAction,
        env,
        active_agent: str,
    ) -> ExpertActionProjection:
        ...
```

## 14.2 Projection model

```python
@dataclass(frozen=True)
class ExpertActionProjection:
    action: np.ndarray
    representable: bool
    reasons: Tuple[str, ...]
    hidden_effects: Tuple[str, ...]
    source_digest: str
    action_digest: str
```

## 14.3 Requirements

- output exact public action shape;
- output contiguous `float32`;
- preserve active joint targets;
- preserve gripper command;
- verify all passive controls are compatible with Phase 0 hold behavior;
- detect equality-constraint transitions;
- detect unrepresented control indices;
- detect unexpected second-agent movement;
- detect NaN/Inf;
- verify bounds;
- never clip silently;
- never drop hidden effects silently.

## 14.4 Grasp and release semantics

If the original expert `SimAction` changes weld/equality state, the public action path must reproduce this through deterministic runtime semantics.

Preferred solution:

```text
gripper close transition
    + contact / nearest valid grasp candidate
    -> attach object using server GraspController

gripper open transition
    -> release currently attached object
```

Requirements:

- same logic for expert and learned actions;
- no target object name passed in the action;
- deterministic tie-breaking;
- only graspable objects;
- maximum grasp distance;
- contact validation where available;
- one object held per gripper;
- clear failure if ambiguous;
- release on open;
- reset clears state.

Do not encode equality IDs in the LeRobot action.

## 14.5 Equivalence validation

For representative action types:

1. save simulator state;
2. execute original RRT `SimAction`;
3. save resulting state;
4. restore simulator state;
5. execute projected public action through normal adapter;
6. save resulting state;
7. compare:
   ```text
   qpos
   qvel
   ctrl
   equality state
   held object
   object pose
   reward
   done
   ```
8. restore original state.

Use explicit tolerances.

The collection service may run a lightweight projection validation on every action and a full simulator equivalence validation on:

```text
first episode of each object/target/agent combination
```

or when a new action pattern is encountered.

---

# 15. Frame Alignment

Correct behavior-cloning alignment is mandatory.

At loop iteration `t`:

```text
observation_t = current environment observation
action_t = RRT expert public action
observation_t+1, reward_t+1, done_t+1 = env.step(action_t)
save frame:
    observation_t
    action_t
    next.reward = reward_t+1
    next.done = done_t+1
    next.success = skill_success_t+1
```

Do not save `observation_t+1` paired with `action_t`.

Pseudo-code:

```python
observation = env.current_observation

while True:
    expert = expert_client.peek_action()
    action = expert.action

    next_observation, reward, terminated, truncated, info = env.step(
        action,
        expert_token=expert.token,
    )

    frame = frame_adapter.build(
        observation=observation,
        action=action,
        reward=reward,
        done=terminated or truncated,
        success=info["skill_success"],
        task=episode_spec.task_instruction,
    )
    transaction.add_frame(frame)

    observation = next_observation

    if info["skill_success"] or failure_or_timeout:
        break
```

Add a unit test that catches an intentional one-step observation shift.

---

# 16. Dataset Schema

## 16.1 Schema version

```text
roco_dataset_schema_version = "0.1.0"
```

Compute a stable SHA-256 hash over canonical schema JSON.

## 16.2 Required features

Example using current LeRobot conventions:

```python
features = {
    "observation.images.front": {
        "dtype": "video",
        "shape": (3, height, width),
        "names": ["channels", "height", "width"],
    },
    "observation.images.active_agent": {
        "dtype": "video",
        "shape": (3, height, width),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": state_field_names,
    },
    "action": {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": action_field_names,
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["reward"],
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["done"],
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["skill_success"],
    },
}
```

Use the exact camera feature shape expected by the installed LeRobot version.

The Phase 0 environment returns HWC uint8. The frame adapter should convert deterministically to the chosen writer format, likely CHW uint8 for the current video schema.

Add a round-trip test:

```text
raw HWC frame
    -> writer frame
    -> LeRobotDataset
    -> loaded tensor
```

Verify color/channel orientation with a synthetic RGB test pattern.

## 16.3 Do not declare

Do not manually declare:

```text
timestamp
frame_index
episode_index
index
task_index
task
```

The dataset writer handles default index fields and task mapping. The `task` string is supplied to `add_frame()`.

## 16.4 Robot type

Use a stable string:

```text
roco-pack-panda
roco-pack-ur5e
```

If multiple active agents/embodiments are stored in one dataset, verify the action/state dimensions are identical. If they differ, create separate datasets per embodiment.

Do not pad incompatible robot state/action dimensions merely to combine datasets in Phase 2.

## 16.5 FPS

Derive source environment step duration from the Phase 0 `RoCoEnvSpec`.

```python
source_fps = 1.0 / effective_step_duration
```

LeRobot dataset FPS is integer.

Requirements:

- if source FPS is within a strict tolerance of an integer, use the rounded integer;
- otherwise fail with `NON_INTEGER_SOURCE_FPS` unless explicit deterministic resampling is configured;
- do not use wall-clock collection speed;
- simulation may run faster than real time;
- store exact source step duration in sidecar metadata.

Do not use the video export FPS from `run_dialog.py` unless it is proven equal to the environment control rate.

---

# 17. Frame Adapter

```python
class LeRobotFrameAdapter:
    def build(
        self,
        observation,
        action,
        reward,
        done,
        success,
        task,
    ) -> Dict[str, Any]:
        ...
```

Requirements:

- validate required cameras;
- HWC RGB uint8 input;
- convert to configured writer layout;
- contiguous arrays/tensors;
- `observation.state` float32;
- `action` float32;
- reward shape `(1,)`, float32;
- done shape `(1,)`, bool;
- success shape `(1,)`, bool;
- exact feature keys;
- include `task`;
- no extra undocumented key;
- no NaN/Inf;
- no mutation of source observation/action;
- deterministic output.

A schema validator must check every frame before `dataset.add_frame()`.

---

# 18. Dataset Adapter

Hide current LeRobot API details behind:

```python
class LeRobotDatasetAdapter:
    @classmethod
    def create(...): ...
    @classmethod
    def resume(...): ...
    def add_frame(...): ...
    def save_episode(...): ...
    def discard_episode(...): ...
    def has_pending_frames(...): ...
    def finalize(...): ...
    def push_to_hub(...): ...
```

## 18.1 Create

Use:

```python
LeRobotDataset.create(
    repo_id=repo_id,
    root=root,
    fps=fps,
    robot_type=robot_type,
    features=features,
    use_videos=True,
    image_writer_threads=config.image_writer_threads,
    image_writer_processes=config.image_writer_processes,
    batch_encoding_size=config.batch_encoding_size,
    streaming_encoding=False,
)
```

Default to non-streaming encoding because simulation may produce frames faster than real time and overflow a real-time encoder queue.

Allow streaming encoding only after a dedicated load test.

## 18.2 Add frame

Call:

```python
dataset.add_frame(frame)
```

The frame must include `task`.

## 18.3 Commit

```python
dataset.save_episode()
```

Only after:

- skill success;
- quality checks;
- accepted episode;
- no pending server failure.

## 18.4 Rollback

```python
dataset.clear_episode_buffer(delete_images=True)
```

Call after:

- failed RRT planning;
- execution failure;
- timeout;
- quality rejection;
- bridge exception after frames were buffered;
- user abort before commit.

Assert no pending frames remain.

## 18.5 Finalize

Always call in a controlled shutdown path:

```python
dataset.finalize()
```

Requirements:

- idempotent wrapper;
- called on normal completion;
- called on SIGINT/SIGTERM where possible;
- called before reading;
- called before Hub push;
- status written to collection manifest;
- errors propagated clearly.

## 18.6 Resume

Use current:

```python
LeRobotDataset.resume(
    repo_id=...,
    root=explicit_root,
    ...
)
```

Before resuming:

- dataset is readable;
- schema hash matches;
- FPS matches;
- feature definitions match;
- robot type matches;
- camera shapes match;
- action/state names match;
- collection config is compatible;
- no pending episode transaction exists.

Refuse resume on mismatch.

## 18.7 Periodic checkpoints

Support:

```text
checkpoint_every_episodes
```

At a checkpoint:

1. ensure no pending episode;
2. finalize;
3. reopen dataset read-only;
4. validate counts and sample decode;
5. reopen through `resume()` for further collection.

This limits damage from hard process termination.

---

# 19. Episode Transaction

```python
class EpisodeTransaction:
    def __enter__(self): ...
    def add_frame(self, frame): ...
    def commit(self, metadata): ...
    def rollback(self, reason): ...
    def __exit__(self, exc_type, exc, tb): ...
```

Required states:

```text
NEW
RECORDING
COMMITTED
ROLLED_BACK
FAILED
```

Requirements:

- only one transaction active;
- commit exactly once;
- rollback exactly once;
- exception causes rollback;
- `save_episode()` only during commit;
- `clear_episode_buffer()` during rollback;
- attempt manifest written for every outcome;
- committed episode index returned;
- no silent commit in `__exit__`;
- post-condition checks for pending frames.

---

# 20. Collection Configuration

```python
@dataclass
class ExpertDatasetCollectionConfig:
    endpoint: str
    repo_id: str
    root: Path
    num_successful_episodes: int
    max_attempts: int

    task: str = "pack"
    skill_name: str = "PUT_OBJECT_IN_CONTAINER"
    active_agents: Tuple[str, ...] = ("Alice",)

    objects: Optional[Tuple[str, ...]] = None
    containers: Optional[Tuple[str, ...]] = None

    seed_start: int = 0
    sampler_seed: int = 0
    sampling_mode: str = "shuffled_cartesian"

    max_episode_steps: int = 500
    settle_steps: int = 5
    success_stability_steps: int = 3

    use_videos: bool = True
    image_writer_threads: int = 4
    image_writer_processes: int = 0
    batch_encoding_size: int = 1
    streaming_encoding: bool = False

    resume: bool = False
    overwrite: bool = False
    checkpoint_every_episodes: int = 10

    success_only: bool = True

    push_to_hub: bool = False
    hub_private: bool = True

    validate_after_collection: bool = True
    replay_validation_episodes: int = 3
```

Validate:

- positive episode/attempt counts;
- `max_attempts >= num_successful_episodes`;
- mutually exclusive resume/overwrite;
- root is explicit;
- task/skill supported;
- nonempty agent list;
- supported objects/containers;
- videos enabled for initial implementation;
- no public push without explicit confirmation flag.

---

# 21. Collection Loop

Pseudo-code:

```python
collector = ExpertDatasetCollector(...)

dataset = collector.create_or_resume_dataset()

try:
    for spec in sampler:
        if successful_count >= target:
            break
        if attempts >= max_attempts:
            break

        attempt = manifest.start_attempt(spec)

        observation, reset_info = env.reset(seed=spec.seed)
        initial_digest = env.get_state_digest()

        initial_validation = validator.validate_initial_state(spec, observation)
        if not initial_validation.valid:
            manifest.reject(attempt, initial_validation.code)
            continue

        expert = expert_client.prepare(spec.to_skill_plan())

        if not expert.prepared:
            manifest.reject(attempt, expert.failure_code)
            continue

        with EpisodeTransaction(dataset, manifest, attempt) as tx:
            for t in range(config.max_episode_steps):
                expert_action = expert_client.peek_action()

                action = expert_action.action.copy()
                previous_observation = observation

                observation, reward, terminated, truncated, info = env.step(
                    action,
                    expert_session_id=expert.session_id,
                    expert_step_index=expert_action.index,
                    expert_action_sha256=expert_action.sha256,
                )

                frame = frame_adapter.build(
                    observation=previous_observation,
                    action=action,
                    reward=reward,
                    done=terminated or truncated,
                    success=info["skill_success"],
                    task=spec.task_instruction,
                )

                tx.add_frame(frame)
                quality_monitor.observe(...)

                if info["skill_success"]:
                    break

                if expert_client.status().exhausted:
                    break

                if terminated or truncated:
                    break

            outcome = outcome_evaluator.evaluate(...)
            quality = quality_monitor.finalize(...)

            if outcome.success and quality.accepted:
                tx.commit(metadata=...)
                successful_count += 1
            else:
                tx.rollback(reason=...)

        expert_client.cancel()

        if checkpoint_due:
            dataset = checkpoint_and_resume(dataset)

finally:
    safely_finalize(dataset)
    env.close()
```

No LLM call is allowed.

---

# 22. Quality Gates

## 22.1 Frame-level hard checks

Reject frame/episode on:

```text
missing feature
wrong shape
wrong dtype
noncontiguous invalid image
NaN/Inf state
NaN/Inf action
action outside declared bounds
image outside uint8
camera shape drift
state/action dimension drift
expert action hash mismatch
observation/action temporal mismatch
```

## 22.2 Episode-level hard checks

Reject episode on:

```text
RRT planning failure
public-action projection failure
bridge error
expert action mismatch
timeout
zero frames
episode too short
episode exceeds max steps
skill predicate false
object not released
object unstable after settling
target occupied by wrong object
passive robot displacement beyond tolerance
unexpected second-agent action
pending writer inconsistency
```

## 22.3 Episode-level warnings

Report but do not necessarily reject:

```text
near action bounds
large joint jumps
long stationary segment
very long trajectory
camera mostly static
minor state-replay drift within tolerance
unbalanced object/target counts
```

## 22.4 Action smoothness metrics

Compute:

```text
max abs delta per action field
mean abs delta
fraction of identical consecutive actions
fraction near lower/upper bounds
```

Do not smooth expert actions during recording. Save executed actions exactly.

## 22.5 Visual checks

For selected episodes:

- save contact sheet;
- first/middle/last frame;
- verify RGB channel orientation;
- target object visible in at least one camera where practical;
- no blank/constant frames;
- no corrupt decode.

---

# 23. Sidecar Metadata

Do not modify LeRobot internal metadata files directly.

Create:

```text
dataset_root/
└── roco_meta/
    ├── schema.json
    ├── collection_config.json
    ├── environment_spec.json
    ├── software_versions.json
    ├── planned_episodes.jsonl
    ├── attempts.jsonl
    ├── episodes.jsonl
    ├── splits.json
    ├── collection_summary.json
    ├── quality_summary.json
    └── replay_summary.json
```

## 23.1 Episode record

```json
{
  "dataset_episode_index": 0,
  "canonical_episode_id": "...",
  "attempt_index": 3,
  "seed": 104,
  "active_agent": "Alice",
  "skill_name": "PUT_OBJECT_IN_CONTAINER",
  "object": "apple",
  "container": "bin_front_left",
  "task": "Put the apple in the front-left bin slot.",
  "num_frames": 143,
  "initial_state_digest": "...",
  "final_state_digest": "...",
  "skill_success": true,
  "full_task_success": false,
  "plan_id": "...",
  "expert_session_id": "...",
  "rrt_plan_count": 1,
  "projected_action_count": 143,
  "projection_schema_hash": "...",
  "quality_status": "accepted",
  "software": {
    "roco_git_commit": "...",
    "lerobot_version": "0.5.2",
    "protocol_version": "...",
    "dataset_schema_version": "0.1.0"
  }
}
```

Do not store session tokens or secrets.

## 23.2 Attempt record

Every attempt, successful or not:

```json
{
  "attempt_index": 3,
  "spec": {...},
  "status": "accepted|rejected|failed",
  "failure_code": null,
  "failure_message": null,
  "frames_buffered": 143,
  "dataset_episode_index": 0
}
```

Use append-only JSONL writes with flush/fsync or an atomic helper.

---

# 24. Dataset Splits

Create deterministic episode-level splits after collection.

## 24.1 Default ratios

Configurable:

```text
train 80%
validation 10%
test 10%
```

For very small smoke datasets, permit train-only but report it.

## 24.2 Leakage prevention

Do not place the same environment seed in multiple splits.

Prefer stratification by:

```text
object
container
active agent
```

Optional stronger generalization split modes:

```text
held-out seeds
held-out object-target combinations
held-out target positions
```

Phase 2 implements metadata splits only; it does not physically duplicate dataset files.

Save:

```json
{
  "train": [0, 1, 2],
  "validation": [3],
  "test": [4],
  "method": "stratified_seed_group",
  "split_seed": 0
}
```

Validate no overlap.

---

# 25. Replay Validation

Replay is a mandatory quality gate.

## 25.1 Replay inputs

For a saved episode:

- episode index;
- original reset seed;
- active agent;
- recorded action sequence;
- original task arguments.

## 25.2 Replay flow

1. reset RoCoGymEnv using original seed;
2. verify initial state digest;
3. execute dataset actions in order;
4. evaluate skill success;
5. compare final state and object pose;
6. record max drift.

## 25.3 Acceptance

For deterministic simulation:

```text
skill_success = true
same held/released state
target object in requested target
final qpos/object pose within tolerance
```

Exact pixel equality is not required.

## 25.4 Replay action loading

Use:

```python
LeRobotDataset(
    repo_id,
    root=root,
    episodes=[episode_index],
    download_videos=False,
)
```

Select the `action` column or iterate raw frames.

Do not rely on images for replay.

## 25.5 Replay sample size

During collection:

- replay at least first accepted episode;
- replay after each periodic checkpoint;
- replay configurable random accepted episodes at final validation.

For smoke testing, replay every episode.

---

# 26. Dataset Validation

Implement:

```bash
python scripts/validate_roco_dataset.py \
  --repo-id local/roco-pack-put \
  --root datasets/roco-pack-put
```

Checks:

1. dataset loads after finalization;
2. expected feature keys;
3. exact state/action shapes;
4. exact feature names;
5. camera decode;
6. frame count > 0;
7. episode count matches manifest;
8. task metadata exists;
9. timestamps monotonic and consistent with FPS;
10. episode frame indices valid;
11. statistics exist and are finite;
12. no NaN/Inf;
13. action bounds respected;
14. all saved episodes marked successful;
15. sidecar episode IDs unique;
16. split lists valid;
17. schema hash matches;
18. software versions recorded;
19. no pending temporary episode;
20. optional replay passes.

Produce machine-readable:

```text
roco_meta/validation_report.json
```

and console summary.

---

# 27. Train-Readiness Smoke Test

Do not train a policy, but prove the dataset can enter a training pipeline.

Required checks:

```python
dataset = LeRobotDataset(repo_id, root=root)
sample = dataset[0]
batch = next(iter(DataLoader(dataset, batch_size=2)))
```

Verify:

```text
observation.state tensor
action tensor
observation.images.<camera> tensor
task/task index metadata
finite dataset statistics
batch shapes
```

Test temporal loading:

```python
delta_timestamps = {
    "observation.images.front": [-1/fps, 0.0],
    "observation.state": [-1/fps, 0.0],
    "action": [0.0],
}
```

Use valid frame intervals and verify returned temporal dimensions.

Optionally verify that ACT configuration can infer input/output features from the dataset, without running optimization.

---

# 28. Hub Publishing

Disabled by default.

Require:

```text
--push-to-hub
```

and default:

```text
--private
```

Before push:

1. finalize;
2. validate;
3. replay required sample;
4. generate dataset card;
5. display repo ID and privacy;
6. require explicit CLI confirmation unless `--yes`.

Do not include secrets or local machine paths in uploaded metadata.

Dataset card should state:

- simulator and robots;
- task;
- expert source: RRT;
- number of episodes/frames;
- cameras;
- state/action representations;
- success-only filtering;
- known limitations;
- license;
- RoCo and LeRobot versions.

---

# 29. CLI Requirements

## 29.1 Collection

```bash
python scripts/collect_roco_expert_dataset.py \
  --endpoint tcp://127.0.0.1:5557 \
  --repo-id amanuel/roco-pack-put-alice-v0 \
  --root datasets/roco-pack-put-alice-v0 \
  --active-agent Alice \
  --num-successful-episodes 100 \
  --max-attempts 300 \
  --seed-start 0 \
  --sampling-mode shuffled_cartesian
```

Options:

```text
--endpoint
--repo-id
--root
--active-agent / --active-agents
--objects
--containers
--num-successful-episodes
--max-attempts
--seed-start
--sampler-seed
--sampling-mode
--max-episode-steps
--settle-steps
--success-stability-steps
--checkpoint-every-episodes
--resume
--overwrite
--push-to-hub
--private / --public
--yes
--log-level
```

## 29.2 Smoke collection

```bash
python scripts/smoke_test_roco_dataset_collection.py \
  --endpoint tcp://127.0.0.1:5557 \
  --root /tmp/roco-phase2-smoke \
  --episodes 2
```

Must:

- use no LLM/API key;
- collect at least one successful episode;
- finalize;
- reload;
- decode cameras;
- replay;
- delete output unless `--keep`;
- exit nonzero on failure.

## 29.3 Inspect

```bash
python scripts/inspect_roco_dataset.py \
  --repo-id local/roco-pack-put \
  --root datasets/roco-pack-put
```

Print:

```text
episodes
frames
tasks
objects
containers
agents
length distribution
feature schema
camera shapes
action/state dimensions
success count
failure-attempt count
split sizes
software versions
```

## 29.4 Replay

```bash
python scripts/replay_roco_dataset_episode.py \
  --endpoint tcp://127.0.0.1:5557 \
  --repo-id local/roco-pack-put \
  --root datasets/roco-pack-put \
  --episode-index 0 \
  --save-video artifacts/replay_episode_0.mp4
```

---

# 30. Failure Codes

At minimum:

```text
PHASE0_NOT_READY
PHASE1_NOT_READY
UNSUPPORTED_TASK
UNSUPPORTED_SKILL
INVALID_INITIAL_STATE
UNKNOWN_OBJECT
UNKNOWN_CONTAINER
TARGET_OCCUPIED
OBJECT_ALREADY_PACKED
OBJECT_HELD_BY_OTHER_AGENT
RRT_PLAN_FAILED
GEOMETRIC_VALIDATION_FAILED
UNREPRESENTABLE_EXPERT_ACTION
EXPERT_ACTION_MISMATCH
EXPERT_STATE_MISMATCH
EXPERT_SESSION_ERROR
PUBLIC_ACTION_EXECUTION_FAILED
GRASP_FAILED
RELEASE_FAILED
OBJECT_SLIPPED
SKILL_NOT_ACHIEVED
SKILL_UNSTABLE
EPISODE_TIMEOUT
FRAME_SCHEMA_ERROR
FRAME_ALIGNMENT_ERROR
ACTION_OUT_OF_BOUNDS
NONFINITE_FRAME
CAMERA_SHAPE_DRIFT
PASSIVE_ROBOT_MOVED
QUALITY_REJECTED
DATASET_SCHEMA_MISMATCH
DATASET_RESUME_FAILED
DATASET_FINALIZE_FAILED
DATASET_LOAD_FAILED
VIDEO_DECODE_FAILED
REPLAY_FAILED
MAX_ATTEMPTS_REACHED
USER_INTERRUPTED
```

Use stable codes in logs and manifests.

---

# 31. Logging

## 31.1 Collection logs

Include:

```text
attempt index
successful episode count
target count
episode spec
seed
plan ID
RRT planning time
number of expert actions
frame count
quality result
save/discard result
dataset episode index
checkpoint/finalize result
```

## 31.2 Do not log

- image bytes;
- complete large arrays;
- Hub token;
- bridge session token;
- private paths in dataset card;
- arbitrary server tracebacks at normal log level.

## 31.3 Progress summary

Example:

```text
Accepted 24/100 | Attempts 31/300 | Object apple | Target bin_front_left
Frames 143 | RRT 1.21 s | Record 3.47 s | Replay pending
```

---

# 32. Tests

## 32.1 Unit tests: schema

Test:

- feature generation from Phase 0 env spec;
- schema hash determinism;
- field names;
- incompatible camera/state/action dimensions;
- integer FPS validation;
- no default LeRobot features manually declared;
- robot-type derivation.

## 32.2 Unit tests: frame adapter

Test:

- HWC to configured writer layout;
- synthetic RGB channel pattern;
- state/action float32;
- reward/done/success shapes;
- task required;
- wrong shape;
- wrong camera;
- NaN/Inf;
- source immutability;
- exact keys.

## 32.3 Unit tests: episode sampler

Test:

- deterministic output;
- balanced combinations;
- no invalid object/container;
- agent ordering;
- canonical task text;
- max attempts;
- explicit seed list.

## 32.4 Unit tests: expert protocol

Test:

- prepare;
- peek without advance;
- expert-aware step validation;
- stale index;
- wrong hash;
- wrong state digest;
- cancel;
- second prepare rejection;
- status after exhaustion.

## 32.5 Unit tests: action projector

Use fake `SimAction` and layouts.

Test:

- ordinary movement;
- passive hold;
- gripper close;
- gripper open;
- hidden control rejection;
- equality transition detection;
- out-of-bounds;
- nonfinite;
- second-agent movement rejection;
- deterministic digest.

## 32.6 Unit tests: grasp controller

Test:

- nearest valid candidate;
- no candidate;
- ambiguous candidates;
- contact requirement;
- deterministic tie-break;
- attach;
- release;
- reset;
- cannot hold two objects;
- no privileged target argument.

## 32.7 Unit tests: transaction

Mock LeRobot adapter.

Test:

- commit calls save exactly once;
- rollback clears buffer;
- exception rolls back;
- no implicit commit;
- pending frame assertion;
- manifest written;
- commit after rollback rejected.

## 32.8 Unit tests: manifest

Test:

- atomic writes;
- unique episode IDs;
- attempt and episode linkage;
- software-version serialization;
- no secret fields;
- resume reads existing count.

## 32.9 Unit tests: quality gates

Test:

- valid episode;
- zero frames;
- timeout;
- action bound violation;
- passive movement;
- unstable success;
- frame mismatch;
- long stationary segment warning;
- action smoothness metrics.

## 32.10 Unit tests: splits

Test:

- deterministic;
- no overlap;
- no seed leakage;
- ratio behavior;
- stratification;
- small dataset handling.

## 32.11 Client/LeRobot unit tests

With a temporary local dataset:

- create;
- add one synthetic episode;
- save;
- finalize;
- reload;
- sample;
- DataLoader batch;
- video decode;
- custom `next.success`;
- resume and append;
- schema mismatch refusal.

## 32.12 Simulator integration tests

Mark:

```python
@pytest.mark.integration
@pytest.mark.mujoco
@pytest.mark.expert
```

Tests:

1. reset valid pack state;
2. prepare one expert plan;
3. expert trajectory representable;
4. execute projected action sequence;
5. skill succeeds;
6. one episode saved;
7. dataset finalizes and loads;
8. failed planning does not save;
9. timeout rolls back;
10. passive robot remains within tolerance;
11. grasp/release projection equivalence;
12. repeated seeded run produces same actions/state within tolerance.

## 32.13 LeRobot integration tests

Mark:

```python
@pytest.mark.integration
@pytest.mark.lerobot
```

Tests:

- official installed version recorded;
- current create/add/save/finalize API works;
- loaded feature tensors valid;
- videos decode;
- stats finite;
- temporal delta loading;
- resume;
- no Hub required.

## 32.14 End-to-end smoke

Collect two episodes, finalize, reload, replay both.

No external LLM or API key.

---

# 33. Crash and Interruption Handling

Handle:

```text
SIGINT
SIGTERM
KeyboardInterrupt
bridge disconnect
RRT server exception
video encoding error
dataset writer error
```

Required behavior:

1. active episode rolls back;
2. attempt marked interrupted;
3. expert session cancelled where possible;
4. dataset finalized where safe;
5. manifest flushed;
6. environment closed;
7. process exits nonzero for incomplete requested collection.

Periodic finalize/resume checkpoints reduce hard-crash exposure.

Do not claim recovery from `SIGKILL` or power loss. Document limitation.

---

# 34. Performance and Storage

Measure:

```text
RRT planning time
environment execution time
frame adaptation time
dataset add-frame time
video encoding time
save-episode time
finalization time
average bytes/frame
average bytes/episode
```

Requirements:

- no unbounded render buffer;
- do not store duplicate raw videos outside dataset by default;
- binary image handling, not JSON;
- no repeated RRT planning per action;
- plan once per episode;
- do not sleep to real time unless explicitly requested;
- collection may run faster than real time;
- memory bounded by LeRobot episode buffer and image writer configuration;
- report disk-space estimate before large collection.

---

# 35. Backward Compatibility

Required:

1. Legacy `run_dialog.py` still works.
2. Phase 0 normal Gym stepping still works without expert mode.
3. Phase 1 normal skill execution still works.
4. Expert service is opt-in.
5. Server starts without LeRobot installed.
6. Client imports without MuJoCo installed.
7. No LLM key required.
8. Existing protocol commands unchanged.
9. Protocol version changes are backward compatible or negotiated.
10. Existing dataset-independent tests continue to pass.
11. No main RoCo dependency upgrade.
12. No changes to RRT math unless a tested bug fix is unavoidable.

---

# 36. Documentation

Create:

```text
docs/phase2_expert_dataset_collection.md
```

Include:

1. objective and non-goals;
2. prerequisites;
3. two-runtime architecture;
4. expert protocol;
5. episode definition;
6. action representability requirement;
7. gripper/grasp semantics;
8. observation/action schema;
9. temporal alignment;
10. task instructions;
11. collection commands;
12. resuming;
13. validation;
14. replay;
15. splits;
16. Hub upload;
17. troubleshooting;
18. failure codes;
19. limitations;
20. transition to Phase 3.

Add a “do not collect production data until” checklist.

---

# 37. Acceptance Criteria

## Prerequisites

- [ ] Phase 0 reset/step/render/state digest works.
- [ ] Phase 1 deterministic skill plan and RRT preparation works.
- [ ] Complete pick-and-place can execute through public actions.
- [ ] Grasp/release requires no privileged action field.

## Expert generation

- [ ] No LLM is called.
- [ ] RRT plans once per episode.
- [ ] Expert action can be peeked without advancing.
- [ ] Executed action equals saved action.
- [ ] Expert action/state hashes are checked.
- [ ] Passive agent is held.
- [ ] Skill success is separate from full-task success.

## Dataset

- [ ] Current LeRobotDataset v3 API is isolated behind adapter.
- [ ] Features use standard keys.
- [ ] Every frame has correct temporal alignment.
- [ ] Failed attempts are discarded.
- [ ] Successful episodes call `save_episode()`.
- [ ] Dataset always calls `finalize()` on controlled completion.
- [ ] Finalized dataset reloads.
- [ ] Cameras decode.
- [ ] Statistics are finite.
- [ ] Resume appends compatible episodes.
- [ ] Schema mismatch blocks resume.
- [ ] Sidecar manifests are complete.

## Quality

- [ ] Every saved episode passes skill success.
- [ ] State/action dimensions never drift.
- [ ] No NaN/Inf or out-of-range action.
- [ ] Replay passes for smoke episodes.
- [ ] Split metadata has no overlap/seed leakage.
- [ ] Two-episode end-to-end smoke test passes.
- [ ] No Hub or API key required for tests.

## Compatibility

- [ ] Server does not import LeRobot.
- [ ] Client does not import rocobench.
- [ ] Legacy and Phase 0/1 workflows remain functional.
- [ ] Python 3.8 server compatibility remains.
- [ ] Python 3.12 client compatibility remains.

---

# 38. Required Implementation Order

Codex must implement in this order:

1. Read this specification completely.
2. Audit Phase 0 and Phase 1 implementation.
3. Inspect exact current LeRobot version/API.
4. Write a concise implementation plan and readiness report.
5. Add dataset schema/config/manifest models.
6. Add expert protocol models.
7. Add action projector tests with fakes.
8. Add expert session/state machine.
9. Add skill outcome evaluator.
10. Add projection-equivalence integration gate.
11. Add LeRobot dataset adapter.
12. Add frame adapter and schema tests.
13. Add episode transaction.
14. Add deterministic episode sampler.
15. Add quality monitor.
16. Add collector.
17. Add resume/checkpoint behavior.
18. Add splits.
19. Add validation and replay.
20. Add CLI scripts.
21. Add smoke and integration tests.
22. Add documentation.
23. Run separate server and client test suites.
24. Run two-episode end-to-end collection and replay.
25. Self-review for privileged labels and temporal misalignment.
26. Report exact commands, results, blocked tests, and risks.

Do not start by editing the main dialogue runner.

---

# 39. Copy-Ready Master Prompt for Codex

Paste the following into Codex from the repository root.

---

## Codex Prompt

Implement **Phase 2: RRT expert demonstration collection and LeRobotDataset v3 generation** according to `PHASE2_CODEX_IMPLEMENTATION_SPEC.md`.

### Objective

Build a professional, reproducible data-collection pipeline that:

```text
resets PackGroceryTask
constructs a deterministic Phase 1 skill plan
uses the existing RRT backend to create an expert action buffer
projects each expert SimAction into the Phase 0 public Gym action space
executes the projected action through RoCoGymEnv.step()
records aligned observation_t/action_t frames
saves only accepted successful skill episodes
creates, finalizes, validates, and replays a LeRobotDataset v3 dataset
```

Do not train or run a learned policy.

### Critical invariant

The action stored in the dataset must be exactly the public action executed through the ordinary Phase 0 `STEP` path.

Do not execute a privileged RRT `SimAction` and save a separately projected label.

Do not include object IDs, equality-constraint IDs, RRT waypoints, or target object names in the `action` feature.

If the expert trajectory is not representable through the future learned-policy action interface, reject it with `UNREPRESENTABLE_EXPERT_ACTION`.

### First actions

Before editing:

1. Read `PHASE2_CODEX_IMPLEMENTATION_SPEC.md`.
2. Inspect the implemented Phase 0 bridge and Phase 1 skill layer.
3. Inspect current:
   - `run_dialog.py`
   - `rocobench/policy.py`
   - `rocobench/subtask_plan.py`
   - `rocobench/envs/base_env.py`
   - `rocobench/envs/task_pack.py`
   - Phase 0 action/observation adapters
   - Phase 0 protocol/server/client
   - Phase 1 skill models/compiler/validator/executor.
4. Inspect the installed target LeRobot source/API, especially:
   - `LeRobotDataset.create`
   - `LeRobotDataset.resume`
   - `add_frame`
   - `save_episode`
   - `clear_episode_buffer`
   - `finalize`
   - standard constants and feature formats.
5. Produce:
   - readiness report for Phase 0 and Phase 1;
   - exact file-by-file implementation plan;
   - current LeRobot version and Python requirement;
   - identified grasp/equality-constraint representation risk.
6. Then implement. Do not stop after planning.

### Runtime separation

Keep:

```text
Python 3.8 RoCo server:
    MuJoCo
    Phase 1 RRT expert
    action projection
    skill success

Python 3.12 client:
    RoCoGymEnv
    LeRobotDataset
    collection
    validation
    replay metadata
```

No LeRobot import in the server. No rocobench import in the client.

### Supported initial scope

```text
task: PackGroceryTask
skill: PUT_OBJECT_IN_CONTAINER(object, container)
active agent: one at a time
passive agent: WAIT and safe hold
expert: existing PlannedPathPolicy/MultiArmRRT
dataset: successful skill episodes
```

Construct skill plans directly. Do not call an LLM.

### Expert protocol

Extend the Phase 0 protocol with:

```text
EXPERT_PREPARE
EXPERT_PEEK_ACTION
EXPERT_STATUS
EXPERT_CANCEL
```

Use existing `STEP` for action execution.

`EXPERT_PREPARE` must:

- create and validate SkillPlan;
- compile and geometrically validate;
- plan RRT action buffer without executing;
- project each SimAction into public action;
- validate representability;
- create expert session.

`EXPERT_PEEK_ACTION` must return the next action without advancing.

The client must then call ordinary `STEP` with the returned action and expert index/hash. The server must reject altered, stale, or state-mismatched expert actions.

### Action representation

Derive public action shape and field ordering from Phase 0.

Implement `ExpertActionProjector`.

Verify:

- active joint targets;
- gripper command;
- passive hold;
- hidden control indices;
- equality changes;
- bounds;
- finite values.

Grasp/release must work through a deterministic server-side controller triggered only by public gripper command and current physical state. Do not pass requested object identity as an action.

Add direct original-SimAction versus projected-public-action equivalence tests for motion, grasp, transport, release, and home.

### Skill success

Do not use only the full PackGroceryTask done predicate.

Implement/reuse Phase 1 task helpers so one skill succeeds when:

- requested object is in requested target;
- object is released;
- placement remains stable for configured steps.

Expose `info["skill_success"]`.

### Temporal alignment

Record:

```text
observation_t
action_t
reward after action_t
done after action_t
skill success after action_t
```

Do not pair action_t with observation_t+1.

Add a regression test that detects one-step misalignment.

### Dataset features

Use current LeRobotDataset conventions:

```text
observation.images.front
observation.images.active_agent
observation.state
action
next.reward
next.done
next.success
task
```

Use current official feature schemas for video images. Convert Phase 0 HWC uint8 frames deterministically and verify RGB/channel order through a synthetic pattern and round-trip load.

Do not manually define writer-generated index/timestamp/task-index features.

### Dataset writer

Isolate LeRobot API in `LeRobotDatasetAdapter`.

Support:

```text
create
add_frame
save_episode
clear_episode_buffer
has_pending_frames
finalize
resume
push_to_hub
```

Always finalize on controlled completion and before reading/pushing.

Use explicit root for resume.

Refuse resume if schema, FPS, robot type, field names, or dimensions differ.

### Episode transaction

Implement transaction semantics:

- add frames;
- commit only on skill success and quality acceptance;
- rollback/clear buffer on all failures;
- exceptions rollback;
- attempt manifest always written;
- no implicit commit.

### Sampling

Implement deterministic balanced sampling across:

```text
object
container
active agent
seed
```

Save planned episode specs before execution.

Reject invalid initial states before RRT planning.

### Sidecar metadata

Write `roco_meta/` without modifying LeRobot internal metadata.

Record:

```text
schema/config/env spec/software versions
all attempts
accepted episodes
seeds
skill arguments
plan IDs
frame counts
state digests
quality results
splits
validation
replay
```

No tokens/secrets.

### Validation

After collection:

1. finalize;
2. reload local LeRobotDataset;
3. validate schema/counts/tasks;
4. decode cameras;
5. verify finite stats;
6. inspect first/middle/last samples;
7. build a DataLoader batch;
8. test delta timestamps;
9. replay selected episodes through RoCoGymEnv;
10. verify skill success and state tolerance.

### Resume and checkpointing

Support:

```text
--resume
--checkpoint-every-episodes
```

At checkpoints:

- finalize;
- reopen read-only and validate;
- reopen via resume for appending.

### Required CLI tools

Implement:

```bash
python scripts/collect_roco_expert_dataset.py
python scripts/smoke_test_roco_dataset_collection.py
python scripts/validate_roco_dataset.py
python scripts/inspect_roco_dataset.py
python scripts/replay_roco_dataset_episode.py
```

Smoke test must collect, finalize, reload, decode, and replay at least one episode without an LLM, API key, or Hub access.

### Tests

Add unit tests for:

- schema;
- frame adapter;
- sampler;
- transaction;
- manifest;
- quality;
- splits;
- resume;
- expert protocol;
- action projector;
- expert session;
- grasp controller;
- skill outcome.

Add simulator integration tests for:

- one successful episode;
- failed episode discarded;
- projection equivalence;
- finalization/load;
- video decode;
- resume;
- replay;
- deterministic collection.

Use test markers and do not pretend unavailable tests passed.

### Backward compatibility

Do not break:

- legacy run_dialog;
- normal Phase 0 Gym operation;
- Phase 1 skill execution;
- existing protocol clients.

Expert mode must be opt-in.

### Completion checks

Before declaring completion:

1. run Python 3.8 server/common tests;
2. run Python 3.12 client/LeRobot tests;
3. run compileall in both runtimes;
4. run two-episode end-to-end smoke collection;
5. finalize and reload dataset;
6. decode all smoke videos;
7. replay both episodes;
8. confirm every saved episode is successful;
9. confirm failed attempt is not saved;
10. confirm exact action executed equals action stored;
11. confirm no privileged label fields;
12. confirm LeRobot DataLoader batch works;
13. confirm resume appends one episode;
14. confirm schema mismatch refuses resume;
15. report exact commands, exit codes, counts, artifacts, skipped checks, and risks.

---

# 40. Suggested `AGENTS.md` Addendum

Merge with existing instructions.

```markdown
## Phase 2 dataset collection

Phase 2 converts RRT skill executions to LeRobotDataset demonstrations.

### Core invariant

The action written to the dataset must be exactly the public action executed through RoCoGymEnv.step(). Never save a projected label while directly executing a privileged SimAction.

### Runtime boundary

- RRT/MuJoCo/expert service: Python 3.8 server.
- LeRobotDataset writer/validation: Python 3.12 client.
- Server must not import LeRobot.
- Client must not import rocobench.

### Dataset requirements

- One episode is one skill execution.
- Record observation_t with action_t.
- Save only successful accepted episodes by default.
- Use stable standard feature keys.
- Include task string in every add_frame call.
- Always finalize before reading or pushing.
- Resume only after schema compatibility checks.
- Failed attempts must clear the episode buffer.

### Privileged information

Object IDs, equality-constraint IDs, RRT paths, and expert-only target data may be sidecar diagnostics, but must never be policy observation/action features.

### Verification

No phase is complete until a saved episode replays through the same public action interface and achieves the skill.
```

---

# 41. Follow-Up Codex Review Prompt

```text
Review the Phase 2 diff as a senior robot-learning data engineer.

Do not edit first. Report findings ordered by severity with file and line references.

Focus on:

1. observation_t/action_t off-by-one alignment;
2. saving an action different from the executed action;
3. privileged RRT or object-specific information leaking into policy features;
4. hidden equality/weld effects that public actions cannot reproduce;
5. incorrect grasp/release semantics;
6. using full-task done instead of skill success;
7. failed episodes accidentally committed;
8. missing clear_episode_buffer on failures;
9. missing or late finalize;
10. resume without schema compatibility checks;
11. HWC/CHW or RGB/BGR errors;
12. incorrect FPS/timestamps;
13. nonfinite stats/actions/states;
14. seed leakage across dataset splits;
15. manifest and dataset episode-index mismatch;
16. replay that does not reset using original seed;
17. direct server execution bypassing RoCoGymEnv.step;
18. LeRobot imports in Python 3.8 code;
19. rocobench imports in Python 3.12 client;
20. tests that mock away the end-to-end contract.

After reporting findings, fix confirmed issues, rerun affected tests, and report exact results.
```

---

# 42. Follow-Up Codex Verification Prompt

```text
Verify Phase 2 end to end with no LLM and no API key.

A. Readiness
1. Print Phase 0 environment/action spec.
2. Print Phase 1 skill registry and supported skill.
3. Print installed LeRobot and Python versions.
4. Confirm server imports without LeRobot.
5. Confirm client imports without MuJoCo/rocobench.

B. Expert projection
1. Prepare one RRT expert episode.
2. Print number of original SimActions and projected public actions.
3. Verify one-to-one mapping.
4. Run motion/grasp/release projection equivalence.
5. Confirm no hidden field is silently dropped.
6. Confirm action stored equals action sent to STEP.

C. Dataset smoke
1. Collect two successful episodes.
2. Force at least one failed/rejected attempt.
3. Confirm only two episodes exist in LeRobotDataset.
4. Confirm failed attempt exists only in attempts manifest.
5. Finalize dataset.
6. Reload it.
7. Decode every camera from both episodes.
8. Print all feature shapes/dtypes.
9. Confirm stats are finite.
10. Build DataLoader batch.

D. Temporal alignment
1. For one frame, save state digest before action.
2. Verify recorded observation corresponds to that digest.
3. Apply recorded action.
4. Verify next reward/done/success corresponds to resulting transition.
5. Run off-by-one regression test.

E. Replay
1. Reset each smoke episode using stored seed.
2. Replay actions from dataset.
3. Verify skill success.
4. Report final object-pose and robot-state errors.
5. Save replay videos.

F. Resume
1. Finalize.
2. Resume with identical schema.
3. Append one successful episode.
4. Finalize and verify episode count increases by one.
5. Attempt resume with changed action dimension and confirm refusal.

G. Reporting
List:
- commands;
- environment names;
- exit codes;
- passed/failed/skipped counts;
- dataset root;
- episodes/frames/tasks;
- replay results;
- artifact paths;
- exact blocked checks and errors.

Do not claim any simulator, video, or replay result passed unless it actually ran.
```

---

# 43. Expected Phase 2 End-State

Collection:

```python
spec = sampler.next()

obs, info = env.reset(seed=spec.seed)

expert = expert_client.prepare(spec.to_skill_plan())

with EpisodeTransaction(dataset_adapter, manifest, spec) as episode:
    while not expert.finished:
        expert_action = expert_client.peek_action()

        next_obs, reward, terminated, truncated, info = env.step(
            expert_action.action,
            expert_session_id=expert.session_id,
            expert_step_index=expert_action.index,
            expert_action_sha256=expert_action.sha256,
        )

        episode.add_frame(
            frame_adapter.build(
                observation=obs,
                action=expert_action.action,
                reward=reward,
                done=terminated or truncated,
                success=info["skill_success"],
                task=spec.task_instruction,
            )
        )

        obs = next_obs

    if info["skill_success"] and quality.accepted:
        episode.commit()
    else:
        episode.rollback(reason=quality.reason)

dataset_adapter.finalize()
```

Validation:

```python
dataset = LeRobotDataset(repo_id, root=root)

sample = dataset[0]

assert "observation.state" in sample
assert "action" in sample
assert "observation.images.front" in sample
```

Replay:

```text
stored seed
    -> reset
stored public actions
    -> ordinary RoCoGymEnv.step
requested object
    -> requested target
skill_success = true
```

The core deliverable is a finalized, replayable, versioned, quality-checked LeRobotDataset whose action labels are exactly executable by the future learned-skill interface.
