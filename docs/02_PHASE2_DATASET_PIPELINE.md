# Phase 2 — RoCo Expert Demonstration Dataset Pipeline
## RRT Rollouts → Validated LeRobotDataset v3

# 1. Objective

Build a production-quality pipeline that records successful RRT-backed RoCo skill executions into a versioned, replayable LeRobotDataset v3 dataset.

```text
deterministic variation
 -> typed SkillPlan fixture
 -> RRT expert execution
 -> synchronized pre-action observation/action records
 -> atomic episode validation
 -> LeRobotDataset v3
 -> split/quality reports
 -> action replay
```

Initial scope:
- task: `PackGroceryTask`;
- skill: `PUT_OBJECT_IN_CONTAINER`;
- active agent: Alice;
- expert: `RRTSkillExecutor`;
- action: Phase 0 absolute joint positions + gripper;
- cameras/state: Phase 0 schema.

# 2. Non-goals

Do not train a policy, call an LLM, execute a learned policy, add concurrent demonstrations, alter physics, or silently mix failed attempts into the expert dataset. Failed/recovery traces belong in a separate quarantine corpus.

# 3. Required files

```text
integrations/lerobot_roco/dataset/
  schema.py config.py episode_sampler.py expert_source.py
  recorder.py writer.py validator.py splitter.py replay.py
  manifest.py statistics.py cli.py
scripts/
  collect_roco_expert_dataset.py
  validate_roco_dataset.py
  visualize_roco_dataset.py
  replay_roco_dataset_episode.py
  create_roco_dataset_splits.py
configs/dataset/
  pack_put_object_alice.yaml
  pack_put_object_debug.yaml
tests/dataset/
docs/phase2_dataset_pipeline.md
```

LeRobot writing/loading runs in Python 3.12. The Python 3.8 simulator provides transitions through the Phase 0 bridge.

# 4. Schema contract

Implement a versioned schema:

```python
@dataclass(frozen=True)
class SkillDataSchema:
    schema_version: str
    task_id: str
    skill_id: str
    embodiment_id: str
    observation_features: Mapping[str, FeatureSpec]
    action_feature: FeatureSpec
    fps: float
    action_representation: str
    camera_aliases: Mapping[str, str]
    state_field_names: Tuple[str, ...]
    action_field_names: Tuple[str, ...]
    bridge_protocol_version: str
```

Required frame features:

```text
observation.images.front
observation.images.active_agent
observation.state
action
timestamp
frame_index
episode_index
task/task_index
```

Required episode metadata:

```text
episode_id, seed, variation_id, task_id, skill_id
canonical_skill_call, natural_language_instruction
agent_name, robot_name, object_name, target_name
success, termination_reason, frame_count
expert_backend, expert_plan_id
schema_hash, source commits and dependency revisions
```

If deterministic subtask stages are derivable from the RRT plan, annotate frames with:
`approach_object`, `grasp_object`, `lift_object`, `transport_to_target`, `lower_object`, `release_object`, `retreat`. Do not invent noisy stages.

# 5. Alignment convention

This is mandatory:

```text
sample[t].observation = observation immediately before action[t]
sample[t].action      = action actually applied after that observation
sample[t].timestamp   = simulator time for that observation
```

Add tests that fail on a one-frame offset.

# 6. Collection architecture

Create a transition observer around the existing expert executor:

```python
class TransitionObserver(Protocol):
    def before_step(self, observation, action, metadata) -> None: ...
    def after_step(self, result) -> None: ...
```

Do not reconstruct the primary dataset from old pickle files. An explicit offline migration tool may load trusted local artifacts and convert them to safe records.

Collection flow:
1. derive a deterministic variation from master seed;
2. reset environment to that exact variation;
3. build a typed skill fixture;
4. prepare and execute the RRT expert;
5. record every pre-action observation and applied public action vector;
6. confirm executor success and task predicate success;
7. validate all samples;
8. atomically commit or quarantine.

# 7. Variation sampling

Variation dimensions:
- object identity;
- target slot;
- object pose;
- distractor arrangement;
- optional agent later.

Generate a stable `variation_id` from the full canonical variation spec, not merely the seed. Save the exact variation JSON so replay does not depend on hidden RNG call order.

# 8. Atomic commit and quarantine

Stage each episode in a temporary directory. Commit only when:
- executor reports success;
- task postcondition confirms requested object in requested target;
- no nonfinite state/action;
- feature names, dtypes, and shapes match;
- actions remain in valid bounds;
- timestamps are monotonic;
- video/low-dimensional frame counts align;
- episode length is within configured limits.

Quarantine with structured codes such as:
`EXPERT_PLANNING_FAILED`, `EXPERT_EXECUTION_FAILED`, `POSTCONDITION_FAILED`, `SCHEMA_ERROR`, `TIMESTAMP_ERROR`, `VIDEO_ALIGNMENT_ERROR`.

Resume must be idempotent and must not duplicate committed variation IDs.

# 9. Splits

Split by episode/variation groups, never frames. Requirements:
- no episode crosses splits;
- no identical variation appears in multiple splits;
- save immutable episode ID lists;
- validator detects leakage.

Default:
```text
train 70%, validation 15%, test 15%
```

For Phase 3 tiny overfit, create a separate explicit subset manifest without modifying official splits.

# 10. Timestamps and frequency

Record simulator time and environment step index. Wall time is diagnostic only. Declared FPS must match the actual environment cadence. Any resampling must be explicit, versioned, and tested; avoid resampling initially.

# 11. LeRobotDataset v3 writing

Use the pinned public `LeRobotDataset` API. Do not manually write Parquet/MP4 layout unless the pinned API requires a documented converter. Record codec and video settings. Images are HWC `uint8`; state/action are fixed-shape finite numeric arrays.

# 12. Manifest

Create `dataset_manifest.json`:

```json
{
  "schema_version": "1.0",
  "schema_hash": "...",
  "dataset_revision": "...",
  "roco_commit": "...",
  "lerobot_version": "...",
  "lerobot_commit": "...",
  "bridge_protocol": "...",
  "task": "pack",
  "skill": "PUT_OBJECT_IN_CONTAINER",
  "action_representation": "absolute_joint_position_plus_gripper",
  "fps": 10,
  "episodes_attempted": 100,
  "episodes_committed": 82,
  "episodes_quarantined": 18,
  "split_manifest": "splits.json"
}
```

# 13. Validator

Produce JSON and Markdown reports checking:
- loader compatibility;
- schema/features/dtypes/shapes;
- finite values and action bounds;
- monotonic timestamps/cadence;
- episode/frame indices;
- video alignment;
- task/subtask references;
- duplicate IDs;
- split leakage;
- zero/near-zero variance;
- blank/frozen/repeated cameras;
- excessive episode length;
- final postcondition;
- metadata/provenance completeness.

# 14. Replay

Reset from saved variation, replay recorded actions, and compare:
- success;
- termination;
- final qpos/qvel/ctrl;
- object pose;
- frame count;
- skill postcondition.

Report state drift over time. Pixel identity is optional; report image similarity separately.

# 15. CLI

```bash
python scripts/collect_roco_expert_dataset.py \
  --config configs/dataset/pack_put_object_alice.yaml \
  --num-episodes 100 \
  --output-root artifacts/datasets/pack_put_object_v1 \
  --master-seed 42

python scripts/validate_roco_dataset.py --dataset-root ... --report-dir ...
python scripts/replay_roco_dataset_episode.py --dataset-root ... --episode-id ... --compare
python scripts/visualize_roco_dataset.py --dataset-root ... --episode-id ... --output ...
```

# 16. Tests

Unit:
- schema and variation hash stability;
- instruction rendering;
- frame alignment;
- split leakage;
- atomic commit/quarantine;
- resume idempotency;
- timestamp/action/video validators;
- subtask derivation;
- manifest serialization.

Integration:
- collect one episode;
- load via pinned `LeRobotDataset`;
- inspect first/last sample;
- replay;
- collect multiple episodes and split;
- interrupt/resume;
- forced expert failure quarantine.

No API key or LLM.

# 17. Acceptance criteria

- [ ] Dataset loads through public pinned LeRobotDataset.
- [ ] Pre-action alignment is tested.
- [ ] One expert episode replays successfully.
- [ ] Schema/manifest/splits are versioned.
- [ ] No variation leakage.
- [ ] Failed attempts are quarantined.
- [ ] Resume is idempotent.
- [ ] Validator has no critical errors.
- [ ] Exact dependency revisions are recorded.

# 18. Copy-ready Codex prompt

```text
Implement Phase 2 according to 02_PHASE2_DATASET_PIPELINE.md.

Inspect the actual Phase 0 bridge and Phase 1 RRTSkillExecutor. Find the exact hook exposing each observation immediately before the exact public action applied to the simulator. Reuse it; do not duplicate execution.

Use the pinned public LeRobotDataset v3 API. Implement the versioned schema, deterministic variation sampler, transition observer, atomic recorder/writer, manifest, quarantine, resume, variation-group splits, validator, replay, visualization, configs, tests, and docs.

Mandatory convention: observation[t] precedes action[t].
Use simulator timestamps.
Commit only if executor and task postcondition both succeed.
Do not train a policy or use an LLM.

Before completion:
1. collect a one-episode debug dataset;
2. load it with LeRobotDataset;
3. print feature shapes/dtypes;
4. replay the episode from the saved variation;
5. create multi-episode splits;
6. run validator and leakage checks;
7. report commands, results, and artifact paths.

Never claim validity unless the dataset loaded through the pinned LeRobot API and replay executed.
```

# 19. Review prompt

```text
Review Phase 2 for observation/action off-by-one errors, wall-clock training timestamps, premature episode commit, split leakage, unstable variation IDs, feature drift, manual LeRobot storage internals, resume duplication, failed traces mixed into expert data, action/gripper unit errors, and non-reproducible replay. Fix confirmed issues and rerun collection, loading, validation, and replay.
```

# 20. Verification prompt

```text
Run unit tests, compileall, one debug collection, LeRobotDataset loading, replay, split checks, forced interruption/resume, forced expert failure quarantine, and visualization. Report episode/frame counts, feature keys, shapes/dtypes, cadence, action bounds, replay success/state errors, split counts, and validation report. Gate fails if loading or replay fails.
```
