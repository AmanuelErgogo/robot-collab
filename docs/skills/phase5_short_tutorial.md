# Phase 5 Short Tutorial

This tutorial checks the Phase 5 learned skill executor path and runs the
available demos around it.

Phase 5 does not put LeRobot inside the RoCo simulator runtime. The learned
executor lives under `rocobench/skills/learned`, accepts the existing Phase 1
`SkillPlan`, and talks to policies only through an injected
`LearnedPolicyHandle`.

Run commands from the repository root.

## 1. Runtime Split

Use the RoCo Python 3.8 environment for simulator-backed commands:

```bash
conda activate roco
export MUJOCO_GL=egl
```

Use the Python 3.12+ LeRobot environment only for Phase 4 client-side policy
rollout commands:

```bash
conda activate lerobot-roco
```

Do not install LeRobot into the RoCo environment. Do not import `rocobench`
from the LeRobot client package.

## 2. Inspect the Learned Skill Config

The default Phase 5 learned policy config is:

```text
configs/skills/learned_pack_put.yaml
```

Inspect it without loading LeRobot:

```bash
python - <<'PY'
from rocobench.skills.learned.config import load_learned_skill_config

config, specs = load_learned_skill_config("configs/skills/learned_pack_put.yaml")
print(config.to_dict())
print([spec.policy_id for spec in specs])
PY
```

Expected result: one enabled `PUT_OBJECT_IN_CONTAINER` policy spec for Alice
and default fallback mode `return_request`.

## 3. Run Phase 5 Tests

These tests exercise registry resolution, policy handle lifecycle, monitors,
success predicates, fallback decisions, artifact writing, cancellation, and the
executor state machine:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/skills/learned
```

Expected result: all learned executor tests pass.

Run the broader skill and evaluation regression set:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/skills \
  tests/test_legacy_parser_regression.py \
  tests/evaluation
```

The lightweight environment may skip simulator dependency checks. To verify the
PackGrocery feedback path with MuJoCo dependencies, run the relevant test in the
RoCo environment:

```bash
conda run --no-capture-output -n roco \
  pytest -q tests/skills/test_pack_task_feedback.py
```

## 4. Demo: No-LLM Skill Smoke Test

This demo does not need MuJoCo, dm_control, OpenAI, Gymnasium, or LeRobot:

```bash
python scripts/smoke_test_pack_skill.py
```

Expected output includes the canonical skill plan:

```text
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
```

and the synthetic legacy action plan:

```text
EXECUTE
NAME Alice ACTION PICK apple PLACE bin_front_left
NAME Bob ACTION WAIT
```

This verifies Phase 1 skill parsing, validation, and compilation. It does not
run a learned policy.

## 5. Demo: Real RRT-Backed Skill Execution

Run this in the RoCo Python 3.8 environment:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/smoke_test_pack_skill.py --execute
```

Expected output includes:

```text
Execution status: success
Simulation steps:
```

This runs the existing RRT-backed skill executor against `PackGroceryTask`.
It is the explicit expert/baseline path, not a hidden fallback inside learned
execution.

## 6. Demo: Direct ACT Policy Rollout

This uses the Phase 4 direct inference path to run the Phase 3 ACT checkpoint
through the Phase 0 bridge. It is useful for validating the policy side that a
real Phase 5 deployment would inject through `LearnedPolicyHandle`.

Start the bridge in a RoCo terminal:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5561 \
  --seed 0 \
  --image-height 96 \
  --image-width 96 \
  --max-episode-steps 20 \
  --headless \
  --allow-remote-shutdown \
  --request-log-level INFO
```

In a second terminal, run a short rollout in the LeRobot environment:

```bash
conda run --no-capture-output -n lerobot-roco \
  python scripts/rollout_roco_policy.py \
  --config configs/evaluation/act_pack_put_debug.yaml \
  --endpoint tcp://127.0.0.1:5561 \
  --max-steps 3 \
  --no-video \
  --overwrite
```

For a three-step smoke run, expected output includes:

```text
"termination_reason": "MAX_STEPS"
"action_bound_violations": 0
```

Inspect the generated episode:

```bash
python scripts/inspect_roco_rollout.py \
  --artifact-dir artifacts/evaluation/phase4/act_pack_put_debug/episodes/episode_000000
```

Expected result: aligned actions, states, and policy chunk trace rows.

Shut down the bridge from the LeRobot environment:

```bash
conda run --no-capture-output -n lerobot-roco python - <<'PY'
import os
import sys

client_src = os.path.join(os.getcwd(), "integrations", "lerobot_roco", "client", "src")
sys.path.insert(0, client_src)

from lerobot_roco_env.client import RemoteRoCoClient

client = RemoteRoCoClient(endpoint="tcp://127.0.0.1:5561", request_timeout_ms=5000)
try:
    client.shutdown()
finally:
    client.close()
PY
```

## 7. Artifact Locations

When `LearnedSkillExecutor.execute(..., artifact_dir=...)` is used, Phase 5
writes:

```text
skill_call.json
executor_config.json
policy_spec.json
checkpoint_metadata.json
instruction.json
state_machine.jsonl
monitor_events.jsonl
actions.npz
states.npz
policy_chunks.jsonl
result.json
fallback.json
```

The direct ACT rollout writes episodes under:

```text
artifacts/evaluation/phase4/act_pack_put_debug/episodes/
```

Use the Phase 4 inspector before trusting a rollout artifact:

```bash
python scripts/inspect_roco_rollout.py \
  --artifact-dir artifacts/evaluation/phase4/act_pack_put_debug/episodes/episode_000000
```

## 8. What This Does Not Claim

- The no-LLM smoke test does not run MuJoCo, RRT, LeRobot, or learned policy
  inference.
- The RRT-backed smoke test is an expert/baseline execution demo.
- The direct ACT rollout is closed-loop policy inference through the bridge, but
  it is not yet wired as a production `LearnedSkillExecutor` deployment.
- Do not report learned skill manipulation success unless a real
  `LearnedPolicyHandle` and environment adapter were wired and that command
  actually ran.

## 9. Troubleshooting

If `transforms3d`, `dm_control`, or MuJoCo imports fail, run simulator-backed
tests in the `roco` environment.

If the bridge port is busy, choose another localhost endpoint and use the same
endpoint in both bridge and rollout commands.

Headless EGL warnings can appear during startup. Treat them as non-fatal only
when the command continues and the rollout/test result is successful.

If rollout fails with missing LeRobot, PyTorch, or processor dependencies, run
the rollout command in `lerobot-roco`, not `roco`.

