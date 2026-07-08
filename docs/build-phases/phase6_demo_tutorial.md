# Phase 6 Demo Tutorial

This tutorial has two demo paths:

- A real simulator-backed PackGrocery/RRT demo that uses actual MuJoCo/RRT
  execution.
- A faster canned-controller demo that uses fake executors to exercise failure
  recovery cases that are hard to force deterministically in the simulator.

## 0. Real Simulator Demo

This is the real demo path: actual `PackGroceryTask`, actual RRT compilation,
actual RRT execution, actual MuJoCo simulator state, and Phase 6 event/metric
recording. The planner response is canned to avoid a live LLM dependency.

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario success \
  --output-dir artifacts/planning/phase6_real_pack/success \
  --overwrite
```

Expected result:

```text
"success": true
"success_check": "execution"
```

The summary also reports `apple_postcondition_met`. In this workspace the legacy
RRT executor can report motion execution success even when the one-object pack
postcondition is not met, so the demo keeps those signals separate.

You can also run an invalid-plan recovery demo against the real simulator:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario invalid_then_success \
  --output-dir artifacts/planning/phase6_real_pack/invalid_then_success \
  --overwrite
```

And a real-controller budget exhaustion path:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario budget_exhaustion \
  --output-dir artifacts/planning/phase6_real_pack/budget_exhaustion \
  --max-plan-rounds 1 \
  --overwrite
```

For a strict one-object postcondition experiment, add
`--success-check postcondition`. Only treat that as passed if it exits with code
0.

Commands run successfully in this workspace:

```text
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario success \
  --output-dir artifacts/planning/phase6_real_pack/success \
  --overwrite
success: true
total_sim_steps: 37
apple_postcondition_met: false
video: artifacts/planning/phase6_real_pack/success/executor_artifacts/6b494b7d92d5_rrt/execute.mp4

conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario invalid_then_success \
  --output-dir artifacts/planning/phase6_real_pack/invalid_then_success \
  --overwrite
success: true
invalid_plans: 1
total_sim_steps: 37

conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/demo_phase6_real_pack.py \
  --scenario budget_exhaustion \
  --output-dir artifacts/planning/phase6_real_pack/budget_exhaustion \
  --max-plan-rounds 1 \
  --overwrite
success: false
final event: BUDGET_EXHAUSTED
```

Artifacts:

```text
artifacts/planning/phase6_real_pack/<scenario>/
  events.jsonl
  result.json
  feedback.txt
  summary.json
  executor_artifacts/
```

This real simulator demo is RRT-only. It does not claim a working learned ACT
rollout; the existing Phase 4 debug rollout artifact in this workspace
terminated with `ACTION_OUT_OF_BOUNDS` before stepping.

The `apple_postcondition_met: false` result is a useful limitation: the legacy
RRT executor's `success` status means it generated and executed the requested
motion sequence, not that the object-level packing predicate was satisfied.
Phase 6 records those separately instead of hiding the gap.

## 1. Canned Failure-Recovery Demo

This demo also runs without a live LLM. It uses canned typed planner responses
and fake executors, but it runs the real Phase 6 parser, validator, router,
controller, budgets, feedback renderer, rollback logic, failure-loop detector,
event log, and metrics.

## 2. Run The Canned Demo

From the repository root:

```bash
conda run --no-capture-output -n roco \
  python scripts/demo_phase6_planner.py \
  --scenario all \
  --output-dir artifacts/planning/phase6_demo
```

Scenarios:

- `success`: learned backend succeeds.
- `invalid_replan`: planner targets an occupied slot, receives grounded
  feedback, and replans.
- `slippage_replan`: learned execution changes the current state, feedback is
  rendered from that failed current state, and the next plan uses the new state.
- `inference_fallback`: learned inference fails, state is rolled back, and RRT
  fallback is explicit.
- `repeated_loop`: repeated failure fingerprint forces fallback.
- `budget_exhaustion`: all-wait plans consume the one planner round and stop.

Expected result: all scenarios except `budget_exhaustion` report
`"success": true`; `budget_exhaustion` reports `"success": false` and ends with
`BUDGET_EXHAUSTED`.

## 3. Inspect Artifacts

Each scenario writes:

```text
artifacts/planning/phase6_demo/<scenario>/
  events.jsonl
  result.json
  feedback.txt
  prompts.txt
  executor_artifacts/   # present when the scenario executes a backend
```

Useful checks:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/planning/phase6_demo")
for result_path in sorted(root.glob("*/result.json")):
    result = json.loads(result_path.read_text())
    print(result_path.parent.name, result["success"], result["metrics"])
PY
```

For rollback evidence:

```bash
python - <<'PY'
import json
from pathlib import Path

events = Path("artifacts/planning/phase6_demo/inference_fallback/events.jsonl")
for line in events.read_text().splitlines():
    event = json.loads(line)
    if event["event_type"] == "STATE_RESTORED":
        print(event["metadata"])
PY
```

Expected rollback metadata includes `"equal": true`.

## 4. Run Automated Phase 6 Tests

Fast Phase 6-only tests:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/planning
```

Neighboring skill regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/planning tests/skills tests/test_legacy_parser_regression.py
```

These tests do not require a live LLM or LeRobot.

Commands run successfully in this workspace:

```text
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/planning
10 passed

conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/planning tests/skills tests/test_legacy_parser_regression.py
52 passed
```

## 5. Optional Simulator-Backed RRT Check

This verifies the existing Phase 1 RRT-backed skill path in the RoCo simulator
environment. It is not a learned-policy test.

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  pytest -q tests/integration/test_pack_skill_smoke.py
```

For an actual execution attempt:

```bash
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/smoke_test_pack_skill.py --execute
```

Only count these as passed if the commands exit with code 0 on your machine.

Commands run successfully in this workspace:

```text
conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  pytest -q tests/integration/test_pack_skill_smoke.py
1 passed

conda run --no-capture-output -n roco env MUJOCO_GL=egl \
  python scripts/smoke_test_pack_skill.py --execute
Execution status: success
Simulation steps: 37
```

## 6. Optional LeRobot Boundary Check

Phase 6 does not import LeRobot. To confirm the local pinned LeRobot environment:

```bash
conda run --no-capture-output -n lerobot python - <<'PY'
import lerobot
print(getattr(lerobot, "__version__", "unknown"))
PY
```

This workspace currently has LeRobot `0.3.4` in the `lerobot` env, matching
`integrations/lerobot_roco/compatibility.lock.json`.

LeRobot-side checks run in this workspace:

```text
conda run --no-capture-output -n lerobot \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/lerobot_roco --ignore=tests/lerobot_roco/integration
31 passed

conda run --no-capture-output -n lerobot \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/lerobot_roco/integration
2 skipped
```

## 7. Reading The Output

Look for these event sequences:

- Success: `OBSERVED -> PROMPTED -> PLAN_PARSED -> EXECUTOR_SELECTED ->
  STATE_SNAPSHOTTED -> EXECUTION_STARTED -> SKILL_SUCCEEDED -> TASK_SUCCEEDED`
- Inference fallback: `EXECUTION_FAILED -> STATE_RESTORED ->
  FEEDBACK_RENDERED -> FALLBACK_STARTED -> SKILL_SUCCEEDED`
- Budget exhaustion: `PLAN_REJECTED -> BUDGET_EXHAUSTED`

Metrics keep learned, RRT, fallback, and overall success separate in
`result.json`.
