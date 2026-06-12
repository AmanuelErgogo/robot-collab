# RoCo × LeRobot Engineering Roadmap
## Phases 2–8

**Repository:** `AmanuelErgogo/robot-collab`  
**Prerequisites:** Phase 0 Gym/LeRobot bridge and Phase 1 typed skill/RRT executor  
**Initial task:** `PackGroceryTask`  
**Initial skill:** `PUT_OBJECT_IN_CONTAINER(object, container)`  
**Initial policy:** ACT

# 1. Final architecture

```text
RoCo multi-agent planner
  -> typed SkillPlan
  -> semantic/resource validation
  -> executor routing
  -> learned policy inference
  -> RoCoGymEnv / RPC bridge
  -> RoCoBench simulator
  -> skill outcome monitors
  -> structured result/failure
  -> bounded replanning
```

# 2. Phase boundaries

## Phase 2 — Dataset pipeline
Record successful RRT expert trajectories as a validated, replayable LeRobotDataset v3 dataset. No policy training.

## Phase 3 — ACT training
Prove dataset-to-checkpoint learning with a debug run and controlled tiny-set overfit. No simulator rollout.

## Phase 4 — Direct inference
Run the trained policy directly against RoCoGymEnv. No high-level planner and no production SkillExecutor.

## Phase 5 — LearnedSkillExecutor
Wrap the validated rollout in the Phase 1 `SkillExecutor` contract, with policy registry, monitors, cancellation, structured results, and explicit RRT fallback.

## Phase 6 — Planner integration
Connect RoCo skill planning to learned execution, grounded feedback, rollback, bounded retries, recovery, and fallback.

## Phase 7 — Multi-agent execution
Add per-agent policies, centralized scheduling, reservations, synchronized stepping, collision monitoring, and conservative concurrency.

## Phase 8 — Benchmark release
Freeze task/variation manifests, add LeRobot-native evaluation/EnvHub packaging where feasible, baselines, metrics, CI, and release cards.

# 3. Hard gates

A later phase must not compensate for a failed prerequisite.

**Gate 2→3**
- Dataset loads with the pinned public `LeRobotDataset`.
- Observation/action alignment is proven.
- Action replay succeeds.
- Splits prevent variation leakage.
- Dataset validator passes.

**Gate 3→4**
- ACT debug training completes.
- Tiny subset overfits.
- Checkpoint reloads independently.
- Feature and action contracts match.

**Gate 4→5**
- Direct closed-loop rollout executes correctly.
- Native action units and queue semantics are proven.
- Success and failure termination are trustworthy.
- Evaluation artifacts are aligned.

**Gate 5→6**
- Learned executor returns structured outcomes.
- Policy reset/cancellation/failure paths pass.
- RRT fallback is explicit.
- Caller rollback is verified.

**Gate 6→7**
- Planner feedback is grounded in current state.
- Retries and replans are bounded.
- Failure loops terminate.
- RRT-only legacy path remains available.

**Gate 7→8**
- Sequential multi-agent mode is stable.
- Joint stepper advances the simulator exactly once per tick.
- Conflict admission is conservative.
- STOP_ALL and sequential fallback work.

# 4. Version policy

LeRobot changes quickly. Before Phase 2, create:

```text
integrations/lerobot_roco/compatibility.lock.json
```

Record:
- LeRobot version and git commit;
- Python, PyTorch, CUDA, Gymnasium, transformers, datasets versions;
- RoCo commit;
- bridge protocol;
- schema version;
- action representation.

All version-specific LeRobot imports must live in one compatibility module. Do not implement against an unpinned moving `main`.

# 5. Cross-phase schema

Create one versioned `SkillDataSchema` used by dataset, trainer, evaluator, and executor:

```text
observation feature names/shapes/dtypes
camera aliases and order
state field names/order
action field names/order
action representation and units
control frequency
task/skill/agent/embodiment IDs
bridge protocol and simulator revisions
```

Any incompatible change requires a schema version change.

# 6. Architectural rules

- Environment owns simulation and raw features.
- Dataset recorder owns synchronized episode storage.
- Trainer owns optimization only.
- Policy runner owns direct checkpoint inference.
- Skill executor owns one skill lifecycle.
- Planner integration owns routing, rollback, retry, and recovery.
- Scheduler owns multi-agent resources and concurrency.
- Benchmark owns fixed evaluation conditions and reporting.
- Use task predicates for success.
- Never silently normalize, resize, reorder, clip, retry, or fall back.
- Keep learned, fallback, and overall success separate.
- RRT remains expert, baseline, and fallback until explicitly removed.
- No unsafe network pickle, `eval`, or `exec`.
- All loops, retries, and horizons are bounded.

# 7. Reproducibility

Every run records:
- git commit and dirty flag;
- resolved config;
- all random seeds;
- dependency lock;
- hardware;
- task/skill/agent;
- schema and variation hashes;
- dataset/checkpoint revision;
- policy and processor revisions;
- success predicate version;
- failure taxonomy version;
- commands, logs, timestamps, and artifact paths.

Completed runs are immutable unless `--overwrite` is explicit.

# 8. Branch strategy

```text
phase2/lerobot-dataset-pipeline
phase3/act-training
phase4/direct-policy-rollout
phase5/learned-skill-executor
phase6/planner-learned-execution
phase7/multi-agent-learned-skills
phase8/benchmark-release
```

Each PR must include design notes, tests, commands, sample artifacts, limitations, and no unrelated formatting churn.

# 9. Official API basis

Verify the pinned revision against official LeRobot documentation/source for:
- LeRobotDataset v3;
- subtask annotations;
- ACT;
- SmolVLA;
- action representations;
- environment processors;
- adding benchmarks;
- EnvHub;
- asynchronous inference and RTC;
- training/evaluation CLIs.

Documentation examples may target a different release. Confirm exact API names before coding.
