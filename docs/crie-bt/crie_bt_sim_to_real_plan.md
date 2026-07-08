# CRIE-BT Simulator-To-Real Robot Plan

This document describes the next implementation plan for using CRIE-BT with
the RoCo simulator first, then with the real robot stack. The guiding rule is
to promote one capability at a time: scripted control, simulator control,
hardware shadow mode, supervised hardware execution, then increasingly
autonomous hardware execution.

## Current Starting Point

CRIE-BT currently has a scripted, dependency-light architecture path:

```text
ScriptedPlanner
  -> CollaborativePlan
  -> direct_feedback / bt_mediated / vlm_sarm_monitor_planner controller
  -> ScriptedSkillExecutor
  -> ProgressMonitor / UncertaintyEstimator / FailureDetector
  -> RuntimeEvent / CommunicationManager
  -> JSONL evaluation logs
```

This already supports the three ablation modes:

- `open_loop`: plan once, execute without replanning.
- `direct_feedback`: failure goes directly to the planner.
- `bt_mediated`: failure goes through the BT controller, which may retry
  locally before replanning.

The simulator adapters now target the existing RoCoBench tasks without defining
new simulator tasks. PackGrocery has a typed CRIE-BT skill path:

```text
PackGroceryCRIEPlanner
  -> one PUT_OBJECT_IN_CONTAINER subtask per unpacked PackGrocery item
  -> PackGroceryRRTExecutorAdapter
  -> PackGrocerySkillPlanValidator
  -> RRTSkillCompiler
  -> rocobench.skills.executor.RRTSkillExecutor
  -> PackGroceryTask.step(SimAction)
  -> PackGrocery postcondition, progress evidence, and fake uncertainty
```

The other task integrations reuse the legacy RoCoBench action-plan contract:

```text
LegacyActionPlanner
  -> LEGACY_ACTION_PLAN(response)
  -> LLMResponseParser
  -> env.get_task_feedback(...)
  -> rocobench.skills.executor.RRTSkillExecutor
  -> task-specific env.step(SimAction)
  -> env.get_reward_done(obs) for whole-task success
```

Supported simulator task IDs are:

```text
pack
sort
sweep
sandwich
rope
cabinet
```

The next work is to harden simulator-backed and hardware-backed executors while
keeping the same CRIE-BT planner, feedback, event, and evaluation schemas.

## Phase A: Simulator Integration

### A1. Add A RoCo SkillPlan Adapter

CRIE-BT uses `rocobench.crie_bt.types.SkillCall`, while the existing simulator
skill stack uses `rocobench.skills.models.SkillCall` and `SkillPlan`.

Add a conversion module:

```text
rocobench/crie_bt/roco_adapters.py
```

Responsibilities:

- convert `crie_bt.SkillCall` to `rocobench.skills.models.SkillCall`;
- convert one or more `PlanStep` entries to a legacy `SkillPlan`;
- insert `WAIT()` calls for passive agents;
- preserve `agent`, `skill_name`, `object`, `container`, and instruction text;
- expose conversion errors as structured `ExecutionFeedback`.

Initial target:

```text
PUT_OBJECT_IN_CONTAINER(object, container)
WAIT()
```

For tasks without typed skill registries, use `LEGACY_ACTION_PLAN(response)`
until a task-specific skill contract is justified by real planner or learned
policy requirements.

### A2. Implement A Simulator RRT Executor Adapter

Replace the placeholder `rocobench.crie_bt.executor.RRTSkillExecutor` with an
adapter around the existing simulator skill stack:

```text
CRIE-BT SkillCall
  -> RoCo SkillPlan
  -> PackGrocerySkillPlanValidator
  -> RRTSkillCompiler
  -> rocobench.skills.executor.RRTSkillExecutor
  -> SkillExecutionResult
  -> CRIE-BT ExecutionFeedback
```

The adapter should accept injected dependencies rather than constructing hidden
global state:

```python
RRTSkillExecutor(
    env=pack_env,
    agent_names=["Alice", "Bob"],
    validator=...,
    compiler=...,
    executor=...,
)
```

Map simulator statuses to CRIE-BT failure codes:

```text
success + postcondition true       -> BTStatus.SUCCESS / FailureCode.NONE
invalid plan                       -> POSTCONDITION_FAILED or UNKNOWN
motion planning failed             -> NO_PROGRESS or POSTCONDITION_FAILED
timeout                            -> TIMEOUT
unsafe contact / collision info    -> SAFETY_CONFLICT
object not at target after success -> POSTCONDITION_FAILED
```

Acceptance gate:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/crie_bt tests/skills tests/planning
```

### A3. Add Simulator Progress Evidence

Extend `ProgressMonitor` helper functions so they read real RoCo observation
objects robustly:

- `get_object_position(obs, object_name)`;
- `get_target_position(env, target_name)`;
- `get_gripper_position(obs, agent_name)`;
- `is_object_grasped(obs, agent_name, object_name)`;
- `is_object_at_target(obs, object_name, target_name)`.

For PackGrocery, prefer existing environment helpers when available:

```text
env.get_agent_held_object(...)
env.get_packed_slot_for_object(...)
env.get_slot_occupancy(...)
env.bin_slot_xposes
obs.objects[object_name]
```

The monitor must remain graceful when fields are missing. Missing evidence
should reduce certainty, not crash execution.

Acceptance gate:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py --scenario success --overwrite
```

Then run a CRIE-BT simulator smoke that uses the new RRT adapter.

### A4. Add A CRIE-BT Simulator Runner

Add:

```text
scripts/run_crie_bt_sim.py
```

Current CLI:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/run_crie_bt_sim.py \
    --task pack \
    --mode bt_mediated \
    --episodes 1 \
    --output artifacts/crie_bt/sim/pack_bt_rrt.jsonl
```

The runner:

- instantiate the RoCo simulator task in the Python 3.8 runtime;
- create the typed PackGrocery stack or generic legacy action-plan stack;
- use the existing CRIE-BT controller modes;
- write the same JSONL schema as `scripts/run_crie_bt_eval.py`;
- record artifacts for plan, feedback, BT events, simulator result, and video
  when available.

### A5. Compare Three Modes In Simulation

After the RRT adapter works, run comparable simulator ablations:

```text
direct_feedback
bt_mediated
vlm_sarm_monitor_planner
```

Suggested simulator scenarios:

- nominal PackGrocery success;
- invalid target or occupied target;
- forced motion-planning failure;
- timeout;
- no-progress detector using repeated stagnant evidence;
- optional failure injection at executor level.

Metrics:

- success rate;
- steps and simulator time;
- planner calls;
- replans;
- local retries;
- recovery rate;
- explanation count;
- unnecessary replanning proxy;
- safety stops.

### A6. Integrate Learned Policies Through The Bridge

Once the RRT path is stable, implement the learned executor adapter:

```text
CRIE-BT SkillCall
  -> bridge-compatible task instruction / policy context
  -> Phase 0 RoCoGymEnv / LeRobot policy
  -> action chunks
  -> ExecutionFeedback
```

Use the existing runtime boundary:

- RoCo simulator stays Python 3.8;
- Gymnasium/LeRobot client stays isolated;
- no `rocobench` imports from the client package;
- no Gymnasium/LeRobot imports from normal RoCo modules.

The learned executor should report policy metadata when available:

```text
confidence
entropy
log_prob
action_norm
chunk_disagreement
```

This metadata feeds `UncertaintyEstimator(mode="policy_metadata")`.

## Phase B: Real Robot Readiness

Real robot work should start only after simulator gates pass. Hardware
execution must be opt-in, supervised, and recoverable.

### B1. Define A Real-World CRIE-BT Environment Adapter

Current real-world components include:

```text
real_world/real_env.py
real_world/realur5.py
real_world/kinect.py
real_world/task_blockincup.py
real_world/runners/dialog_runner.py
```

Add a CRIE-BT adapter that presents a stable minimal interface:

```python
class RealWorldCRIEEnvAdapter:
    def reset(self): ...
    def get_observation(self): ...
    def execute_skill(self, skill_call, guarded=True): ...
    def get_progress_evidence(self, skill_call): ...
    def stop(self): ...
    def home(self): ...
```

The adapter should translate real-world observations into the same evidence
shape used by `ProgressMonitor`:

```text
objects[object].position
targets[target].position
agents[agent].gripper_position
held[agent]
slot_occupancy[target]
safety_state
```

It should not expose raw camera frames, robot IP addresses, calibration files,
or robot commands in public logs.

### B2. Start With Shadow Mode

Before moving the robot, run CRIE-BT in shadow mode:

```text
camera observation
  -> planner
  -> BT controller
  -> proposed skill
  -> communication overlay
  -> no robot motion
```

Shadow mode acceptance:

- plans are parsed and validated;
- BT events and overlays are generated;
- human can approve, reject, or correct target/object labels;
- no robot command is sent;
- logs contain enough evidence to replay the decision.

Suggested command:

```bash
python scripts/run_crie_bt_real.py \
  --task blockincup \
  --mode bt_mediated \
  --planner scripted \
  --executor shadow \
  --episodes 1 \
  --output artifacts/crie_bt/real_shadow/blockincup.jsonl
```

### B3. Add A Hardware Safety Gate

Before any real movement, add a safety gate that checks:

- robot connection health;
- emergency stop availability;
- workspace bounds;
- calibration file freshness;
- object detection confidence;
- target detection confidence;
- commanded pose inside allowed region;
- no human hand detected in robot motion region if perception supports it;
- operator confirmation required unless explicitly disabled.

Every hardware command should pass through:

```text
BT decision
  -> safety gate
  -> human approval gate
  -> robot primitive
```

The first hardware executor should refuse autonomous execution by default.

### B4. Implement Real Robot Primitive Executor

Use existing real-world primitives where possible:

```text
real_world.real_env.RealEnv.pick_and_place_primitive(...)
real_world.realur5.UR5RTDE
```

Adapter mapping:

```text
PUT_OBJECT_IN_CONTAINER(object, container)
  -> detect object pose
  -> detect/place target pose
  -> guarded pick_and_place_primitive
  -> observe result
  -> ExecutionFeedback
```

For the first real robot milestone, support one robot and one task:

```text
task: blockincup or a simple pick-place analog
agent: Bob or Alice, whichever matches the real UR5 setup
mode: bt_mediated
executor: real_primitive
planner: scripted
```

Keep human execution for the other agent if the task is collaborative.

### B5. Hardware Feedback And BT Mediation

After each primitive or partial primitive, convert real-world evidence into
CRIE-BT feedback:

```text
object moved toward target          -> CONTINUE
low detection confidence            -> EXPLAIN or REQUEST_HUMAN_INPUT
missed grasp                        -> LOCAL_RETRY if safe and retry budget remains
object slipped                      -> EXPLAIN, pause, then retry or request human input
target occupied                     -> REQUEST_REPLAN
workspace/safety conflict           -> ABORT or REQUEST_HUMAN_INPUT
timeout                             -> REQUEST_REPLAN or ABORT
```

The real robot BT policy should be stricter than simulation:

- fewer local retries;
- more human confirmation;
- more abort-on-uncertainty behavior;
- no retry after safety conflict.

### B6. Add A Real Robot Runner

Add:

```text
scripts/run_crie_bt_real.py
```

Initial CLI:

```bash
python scripts/run_crie_bt_real.py \
  --task blockincup \
  --mode bt_mediated \
  --planner scripted \
  --executor shadow \
  --require-human-confirmation \
  --output artifacts/crie_bt/real/blockincup_shadow.jsonl
```

Only later enable:

```bash
python scripts/run_crie_bt_real.py \
  --task blockincup \
  --mode bt_mediated \
  --planner scripted \
  --executor real_primitive \
  --require-human-confirmation \
  --output artifacts/crie_bt/real/blockincup_guarded.jsonl
```

Do not add a no-confirmation hardware mode until shadow and guarded runs are
stable and reviewed.

## Phase C: Sim-To-Real Evaluation Protocol

### C1. Use The Same Episode Schema

Simulation and real robot runs should share the CRIE-BT JSONL schema:

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

Real robot logs should add:

```text
hardware_mode: shadow | guarded | autonomous
human_confirmations
safety_gate_failures
perception_confidence
calibration_version
operator_interventions
```

### C2. Compare In Increasing Risk Order

Run studies in this order:

1. scripted synthetic eval;
2. simulator with scripted planner and RRT executor;
3. simulator with learned policy executor;
4. real-world shadow mode;
5. real-world guarded primitive execution;
6. real-world guarded learned execution;
7. autonomous hardware only after explicit review.

### C3. Report Separate Metrics

Do not mix simulator and real robot scores. Report separate tables:

- scripted ablation;
- simulator ablation;
- hardware shadow evaluation;
- hardware execution evaluation.

For hardware, include safety and human-intervention metrics next to success.

## Implementation Checklist

Simulator:

- [x] Add `rocobench/crie_bt/roco_adapters.py`.
- [x] Convert CRIE-BT `SkillCall` to RoCo `SkillPlan`.
- [x] Replace CRIE-BT RRT placeholder with injected simulator adapters.
- [x] Extend progress helpers for real RoCo `EnvState`.
- [x] Add `scripts/run_crie_bt_sim.py`.
- [x] Add simulator adapter tests under `tests/crie_bt/`.
- [x] Run PackGrocery RRT smoke in the `roco` environment; the current motion
  rollout completes internally but the PackGrocery postcondition remains false.

Real robot:

- [ ] Add `RealWorldCRIEEnvAdapter`.
- [ ] Add shadow executor.
- [ ] Add hardware safety gate.
- [ ] Add guarded primitive executor using existing real-world primitives.
- [ ] Add `scripts/run_crie_bt_real.py`.
- [ ] Add dry-run tests with synthetic real-world observations.
- [ ] Run shadow mode before any robot motion.
- [ ] Require operator confirmation for first hardware motion runs.

Docs and evaluation:

- [x] Extend `docs/crie-bt/crie_bt_api.md` with simulator adapter examples.
- [x] Extend `docs/crie-bt/crie_bt_evaluation.md` with simulator commands.
- [ ] Extend evaluation docs with hardware commands once the real runner exists.
- [ ] Add safety checklist for hardware runs.
- [ ] Keep real robot logs free of private IPs, credentials, and raw images by
  default.

## Near-Term Milestone

The next concrete milestone should be:

```text
BT-mediated PackGrocery RRT simulation, scripted planner, one episode,
local retry/replan decisions logged in CRIE-BT JSONL format.
```

That milestone proves CRIE-BT can control the real RoCo simulator while keeping
the ablation architecture, event schema, and feedback loop intact. Only after
that should the real-world adapter move beyond shadow mode.
