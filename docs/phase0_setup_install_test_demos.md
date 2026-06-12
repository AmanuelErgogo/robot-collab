# Phase 0 Setup, Install, and Test Demos

This runbook sets up and exercises the RoCoBench to Gymnasium/LeRobot Phase 0
bridge.

The bridge uses two separate Python environments:

- RoCo runtime: Python 3.8, MuJoCo, dm_control, owns simulation.
- Client runtime: Python 3.12+, Gymnasium/LeRobot, owns the Gym API.

Run every command from the repository root unless stated otherwise.

## 1. Prerequisites

Install Conda or Miniconda first. On Linux headless machines, use a working
MuJoCo rendering backend, usually EGL:

```bash
export MUJOCO_GL=egl
```

If EGL is unavailable, use an X display or another MuJoCo-supported backend.
Rendering demos require actual RGB rendering; fake-backed unit tests do not.

## 2. RoCo Runtime Install

Create the legacy simulator environment:

```bash
conda create -n roco python=3.8
conda activate roco
```

Install MuJoCo, dm_control, RoCo dependencies, and bridge transport packages.
Use `python -m pip` so packages are installed into the same interpreter that
will run the demo:

```bash
python -m pip install mujoco==2.3.0
python -m pip install dm_control==1.0.8
python -m pip install -r requirements-phase0-roco.txt
```

Install the repository in editable mode if you want imports to work outside the
repo root:

```bash
pip install -e .
```

Quick import check:

```bash
python -c "from rocobench.envs.task_pack import PackGroceryTask; print('RoCo OK')"
```

## 3. Client Runtime Install

Create the Gymnasium/LeRobot-side environment:

```bash
conda create -n lerobot-roco python=3.12
conda activate lerobot-roco
```

Install the client package:

```bash
python -m pip install -e "integrations/lerobot_roco/client[test]"
```

Install LeRobot support when validating LeRobot preprocessing:

```bash
python -m pip install -e "integrations/lerobot_roco/client[lerobot,test]"
```

The client package currently accepts `lerobot>=0.5.1,<0.6`, matching the
published PyPI release line available for Python 3.12+. If pip cannot resolve
LeRobot, update pip first and retry:

```bash
python -m pip install --upgrade pip
python -m pip install -e "integrations/lerobot_roco/client[lerobot,test]"
```

Quick import check:

```bash
python -c "from lerobot_roco_env import RoCoGymEnv, make_env; print('Client OK')"
```

## 4. Run the RoCo Bridge Server

Use the RoCo Python 3.8 environment:

```bash
conda activate roco
export MUJOCO_GL=egl
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --image-height 256 \
  --image-width 256 \
  --headless
```

For Bob as the active agent:

```bash
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Bob \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --headless
```

Leave this process running while using the client commands below.

### Live Preview Window

Non-headless mode only means the server does not force `MUJOCO_GL=egl`; it does
not automatically open a simulator viewer. The bridge normally renders
offscreen RGB arrays for Gym/LeRobot.

To see frames live, start the server with `--live-view` from a desktop session
that can open GUI windows:

```bash
conda activate roco
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --live-view \
  --live-view-camera front
```

The preview updates when the client sends `reset`, `step`, or `render`
requests. In another terminal, run:

```bash
conda activate lerobot-roco
python scripts/smoke_test_roco_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --steps 100
```

Press `q` or `Esc` in the preview window to close it. Use
`--live-view-camera active_agent` to watch the active-agent camera alias.

## 5. Diagnostics Demo

Use the Python 3.12 client environment in a second terminal:

```bash
conda activate lerobot-roco
python scripts/smoke_test_roco_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --diagnostics-only
```

Expected output includes:

- `PING` response;
- `HELLO` protocol negotiation;
- task name `pack`;
- action shape and state shape from `GET_SPEC`;
- camera aliases and image dimensions.

## 6. Safe Hold-Action Demo

This demo resets the environment, runs safe hold actions, renders once, and
checks `info["is_success"]`.

```bash
python scripts/smoke_test_roco_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --steps 5
```

Save artifacts:

```bash
python scripts/smoke_test_roco_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --steps 5 \
  --artifacts-dir artifacts/phase0
```

Artifact examples:

- `artifacts/phase0/env_spec.json`
- `artifacts/phase0/reset_info.json`
- `artifacts/phase0/render_000.png`

Random actions are available but should only be used for diagnostics:

```bash
python scripts/smoke_test_roco_bridge.py --steps 5 --random-actions
```

## 7. Python Gym Demo

Use this in the client environment while the server is running:

```bash
python - <<'PY'
from lerobot_roco_env import RoCoGymEnv

env = RoCoGymEnv(endpoint="tcp://127.0.0.1:5557", active_agent="Alice")
obs, info = env.reset(seed=0)

print("keys:", obs.keys())
print("agent_pos:", obs["agent_pos"].shape, obs["agent_pos"].dtype)
print("cameras:", {k: v.shape for k, v in obs["pixels"].items()})
print("is_success:", info["is_success"])

obs, reward, terminated, truncated, info = env.step(env.hold_action())
print("transition:", reward, terminated, truncated, info["is_success"])
print("render:", env.render().shape)

env.close()
PY
```

## 8. LeRobot Compatibility Demo

Use the client environment with LeRobot installed:

```bash
conda activate lerobot-roco
python scripts/smoke_test_lerobot_preprocessing.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice
```

The report prints:

- Python/Gymnasium/LeRobot versions;
- raw observation keys;
- raw image and state shapes;
- verified LeRobot dataset/policy feature keys;
- verified mapping from `pixels` to `observation.images.<camera>`;
- verified mapping from `agent_pos` to `observation.state`;
- verified action feature shape.

## 9. Native SimAction Replay Demo

This is a debug-only correctness path for trusted local RoCo artifacts. It does
not transfer pickle over RPC.

First create a real trusted local action file in the RoCo environment. Do not
use the literal placeholder `path/to/actions.pkl`. The recommended format is
`.npz`, because it contains only safe arrays and can be replayed from the client
environment without importing `rocobench`.

```bash
conda activate roco
python scripts/create_roco_hold_action.py \
  --active-agent Alice \
  --seed 0 \
  --output artifacts/phase0/hold_action.npz \
  --headless
```

Then start the server with debug commands enabled:

```bash
conda activate roco
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --enable-debug-commands \
  --headless
```

Replay the trusted local `.npz` file from the client environment:

```bash
conda activate lerobot-roco
python scripts/replay_roco_actions_through_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --action-file artifacts/phase0/hold_action.npz
```

Legacy `.pkl` files containing `SimAction` objects are still supported, but run
pickle replay from the RoCo environment because `pickle.load(...)` needs the
original `SimAction` class and its dependencies.

Only safe array payloads are sent over the bridge.

## 10. Unit Tests

Run common/server fake-backed tests in the RoCo environment:

```bash
conda activate roco
python -m pytest tests/lerobot_roco -m "not client and not integration"
```

Run client fake-backed tests in the Python 3.12 client environment:

```bash
conda activate lerobot-roco
python -m pytest tests/lerobot_roco -m "client and not integration"
```

If your global pytest plugins are incompatible with the active Python version,
disable plugin autoload:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/lerobot_roco -m "not integration"
```

Compile new Python modules:

```bash
python -m compileall integrations/lerobot_roco scripts tests/lerobot_roco
```

## 11. Integration Tests

MuJoCo bridge tests are marked:

```bash
python -m pytest tests/lerobot_roco/integration -m "integration and mujoco"
```

LeRobot tests are marked:

```bash
python -m pytest tests/lerobot_roco/integration -m "integration and lerobot"
```

These tests require the correct environment and, for live bridge checks, a
running server.

## 12. Common Problems

### `ModuleNotFoundError: No module named 'dm_control'`

You are not in the RoCo Python 3.8 environment, or the simulator dependencies
were not installed.

### `ModuleNotFoundError: No module named 'numpy'`

The RoCo dependency file has not been installed in the active environment, or
`pip` pointed at a different Python interpreter. Activate the RoCo environment
and install with `python -m pip`:

```bash
conda activate roco
python -V
python -m pip install -r requirements-phase0-roco.txt
```

For a minimal unblock, install the repository's pinned NumPy version:

```bash
python -m pip install numpy==1.22.0
```

### `ModuleNotFoundError: No module named 'open3d'`

RoCo imports Open3D from `rocobench/envs/env_utils.py`. Install the Phase 0
RoCo requirements again:

```bash
conda activate roco
python -m pip install -r requirements-phase0-roco.txt
```

For a minimal unblock:

```bash
python -m pip install open3d==0.19.0
```

### `ModuleNotFoundError: No module named 'gymnasium'`

You are not in the client Python 3.12 environment, or the client package was not
installed with its dependencies.

### `pyzmq is required for bridge transport`

Install transport dependencies in the active environment:

```bash
pip install pyzmq msgpack
```

### Rendering fails on a headless machine

Try:

```bash
export MUJOCO_GL=egl
```

Then restart the server. If EGL is not available on the machine, use an X
display or another MuJoCo-supported backend.

If an import check prints `libEGL warning` or `failed to create dri2 screen` but
still ends with `RoCo OK`, the Python import succeeded. Treat those warnings as
graphics backend diagnostics. Continue to the server smoke test; only change the
rendering backend if reset/render/server startup fails.

### Client times out

Check that the server is running and that both sides use the same endpoint:

```bash
tcp://127.0.0.1:5557
```

### `EPISODE_ALREADY_ACTIVE`

The server accepts one active episode at a time. Close the client environment or
send `CLOSE_EPISODE`, then reset again.

## 13. Expected Minimal Success

A successful demo run should prove:

- `GET_SPEC` returns task/action/state/camera metadata;
- `reset(seed=0)` returns `pixels`, `agent_pos`, and `info["is_success"]`;
- a hold action executes through `env.step(...)`;
- passive robot controls are included server-side;
- `render()` returns an HWC `uint8` RGB image;
- `env.close()` is idempotent.
