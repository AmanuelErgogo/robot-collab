# Phase 4 Direct Closed-Loop ACT Inference

Phase 4 runs the Phase 3 ACT checkpoint directly against the Phase 0
`RoCoGymEnv`. It does not call the planner, production `SkillExecutor`, async
inference, RTC, or any hidden RRT fallback.

## Boundary

```text
RoCoGymEnv raw observation
  pixels / agent_pos
RoCoPolicyObservationAdapter
  observation.images.* / observation.state
LeRobot saved preprocessor
  normalization, batching, device
ACT predict_action_chunk
LeRobot saved postprocessor
  action unnormalization to native units
ACTActionQueue
  explicit execution_horizon <= checkpoint chunk_size
env.step(native action)
task predicate info["is_success"]
```

The RoCo simulator remains behind the Phase 0 bridge. LeRobot is imported only
by the Phase 4 client/evaluation path.

## Commands

Start the simulator in the RoCo Python 3.8 environment:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --image-height 96 \
  --image-width 96 \
  --max-episode-steps 80 \
  --headless
```

Run one direct rollout in the Python 3.12+ LeRobot environment:

```bash
conda run --no-capture-output -n lerobot-roco \
  python scripts/rollout_roco_policy.py \
  --config configs/evaluation/act_pack_put_debug.yaml \
  --overwrite
```

Run a suite:

```bash
conda run --no-capture-output -n lerobot-roco \
  python scripts/evaluate_roco_policy.py \
  --config configs/evaluation/act_pack_put_validation.yaml \
  --overwrite
```

Inspect one rollout:

```bash
python scripts/inspect_roco_rollout.py \
  --artifact-dir artifacts/evaluation/phase4/act_pack_put_debug/episodes/episode_000000
```

Compare against explicit RRT/expert metrics:

```bash
python scripts/compare_roco_act_vs_rrt.py \
  --act-metrics artifacts/evaluation/phase4/act_pack_put_validation/metrics.json \
  --rrt-metrics path/to/rrt_metrics.json \
  --output-dir artifacts/evaluation/phase4/act_pack_put_validation/comparison
```

## Artifacts

Each episode writes:

- `episode_config.json`
- `rollout_manifest.json`
- `actions.npz`
- `states.npz`
- `policy_chunk_trace.jsonl`
- `events.jsonl`
- `video.mp4` when imageio and rendering are available
- `final_state.json`
- `result.json`

Suite runs also write:

- `suite_manifest.json`
- `metrics.json`
- `episodes.csv`
- `termination_reasons.csv`
- `latency.csv`
- `report.md`

## Acceptance Checks

- The queue trace records `chunk_id` and `chunk_offset` for every executed
  action.
- `execution_horizon` is validated to be no larger than the checkpoint
  `chunk_size`.
- Saved LeRobot postprocessor output is validated against native bridge action
  bounds before `env.step`.
- Success is accepted only from `info["is_success"]`.
- Validation and frozen test suites are separate configs.
- RRT comparison is explicit and does not run as fallback inside ACT rollout.

