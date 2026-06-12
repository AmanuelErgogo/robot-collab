# Phase 0 LeRobot RoCo Bridge

## Objective

Phase 0 validates that a Python 3.12+ Gymnasium/LeRobot client can reset,
observe, step, render, and close the legacy Python 3.8 RoCoBench
`PackGroceryTask` through a stable localhost protocol. It does not implement
skill planning, dataset recording, policy training, or LeRobot source-tree
registration.

## Architecture

```text
Python 3.12+ client runtime
  RoCoGymEnv
  Gymnasium spaces
  RemoteRoCoClient
  LeRobot-compatible raw keys: pixels, agent_pos

ZeroMQ REQ/REP over localhost
  protocol_version = 0.1
  msgpack metadata
  explicit NumPy ndarray encoding

Python 3.8 RoCo runtime
  RoCoBridgeServer
  PackGroceryTask
  RoCoActionAdapter -> SimAction
  RoCoObservationAdapter
  episode state machine
```

The server owns simulator state. The client owns the Gymnasium API. Normal
`rocobench` imports remain unchanged and do not import Gymnasium or LeRobot.

For a command-oriented setup and demo runbook, see
[`phase0_setup_install_test_demos.md`](phase0_setup_install_test_demos.md).

## Protocol

Every request contains:

```python
{
    "protocol_version": "0.1",
    "request_id": "<uuid>",
    "command": "RESET",
    "payload": {...},
}
```

Implemented commands:

- `PING`
- `HELLO`
- `GET_SPEC`
- `RESET`
- `STEP`
- `RENDER`
- `GET_STATE_DIGEST`
- `CLOSE_EPISODE`
- `SHUTDOWN`
- `STEP_NATIVE_ACTION`, debug-only and disabled by default

Errors return stable codes such as `INVALID_ACTION_SHAPE`,
`STEP_INDEX_MISMATCH`, `ACTION_OUT_OF_BOUNDS`, and
`DEBUG_COMMAND_DISABLED`.

## Serialization

Arrays are encoded as:

```python
{
    "__ndarray__": True,
    "dtype": "float32",
    "shape": [3],
    "data": b"...",
}
```

The decoder rejects object dtypes, malformed shapes, mismatched byte counts,
and oversized payloads. Pickle is not used over RPC.

## Action Schema

Phase 0 exposes one active agent at a time, defaulting to `Alice`.

The action mode is:

```text
absolute_joint_position_plus_gripper
```

The server derives the action layout from `SimRobot` and MuJoCo model metadata:

- active robot joint actuator targets;
- one active gripper actuator target;
- bounds from `model.actuator_ctrlrange`;
- field names from task robot config.

Passive robots are held every step by appending their current joint qpos targets
and gripper control value to the generated `SimAction`.

## Repository Inspection Findings

`SimAction` is the legacy low-level simulator action type. It contains:

- `ctrl_idxs`: actuator control indices, stored as `np.int32`;
- `ctrl_vals`: absolute actuator target values, stored as `np.float32`;
- `qpos_idxs`: qpos indices used for action error computation, stored as
  `np.int32`;
- `qpos_target`: target qpos values used by `qpos_error`, stored as
  `np.float32`;
- optional `eq_active_idxs` and `eq_active_vals` for weld/equality state.

`MujocoSimEnv.step(action)` validates that control indices and values are
present, clears the save buffer, saves the initial state, temporarily disables
contact margins, writes `data.ctrl[ctrl_idxs] = ctrl_vals` on every internal
physics step, applies equality state when provided, renders according to
`render_freq`, stops early when `action.compute_error(...)` is below
`error_threshold`, restores contact margins, returns
`(next_obs, reward, done, info)`, and increments `env.timestep`.

`MujocoSimEnv.reset(...)` optionally reloads the XML, resets the physics
keyframe/home pose, calls task randomization when enabled, clears render and
save buffers, renders all configured cameras once, returns `get_obs()`, and
sets `timestep = 0`. The bridge follows the existing deterministic order:

```python
env.seed(np_seed=seed)
env.reset(reload=True)
```

`PackGroceryTask.get_reward_done(...)` returns reward `1` and `done=True` only
when every grocery item is in contact with `bin_inside` or aligned with the bin
within `align_threshold`; otherwise reward is `0` and `done=False`.

Available cameras in `task_pack.xml` are:

- `face_panda`
- `face_ur5e`
- `top_cam`
- `right_cam`
- `left_cam`
- `teaser`
- `video`

The default RoCo render camera list in `MujocoSimEnv` is:

- `face_panda`
- `face_ur5e`
- `top_cam`
- `right_cam`
- `left_cam`
- `teaser`

`PackGroceryTask` maps agents to robots as:

- `Alice -> ur5e_robotiq`
- `Bob -> panda`

Alice's configured controlled joints are the seven UR5e joints/base joint in
`UR5E_ROBOTIQ_CONSTANTS["ik_joint_names"]`; the gripper actuator is
`robotiq_fingers_actuator`, with XML control range `0..255`.

Bob's controlled joints are the eight Panda arm/base joints in
`PANDA_CONSTANTS["ik_joint_names"]`; `PANDA_CONSTANTS["all_joint_names"]` also
contains finger joints, so Phase 0 uses `ik_joint_names` for controlled
joint-position state to avoid duplicating gripper state. Bob's gripper actuator
is `panda_gripper_actuator`, with XML control range `0..255`.

Exact qpos indices, control indices, and all actuator ranges are derived at
runtime from `SimRobot`, dm_control named qvel indexing, and
`model.actuator_ctrlrange`; the bridge does not hard-code those indices.
Velocity indices are derived by joint name instead of assuming qpos and qvel
addresses are identical. `PlannedPathPolicy.map_qpos_to_ctrl(...)` builds
absolute joint-position controls from the planned multi-robot qpos vector,
uses the robots' joint control indices, sets qpos targets for error checking,
and appends gripper controls when a robot must keep grasping an in-hand object.

Environment state is saved through `SimSaveData`, which deep-copies qpos, qvel,
ctrl, xpos, xquat, equality activity, body positions, and body quaternions.
The state digest command hashes contiguous qpos/qvel/ctrl arrays including
their dtype and shape.

## Observation Schema

The Gym observation is:

```python
{
    "pixels": {
        "front": np.ndarray[H, W, 3],
        "active_agent": np.ndarray[H, W, 3],
    },
    "agent_pos": np.ndarray[state_dim],
}
```

Default camera aliases for `PackGroceryTask`:

- `front -> teaser`
- `active_agent -> face_ur5e` for Alice
- `active_agent -> face_panda` for Bob

`agent_pos` contains controlled active-agent joint positions, matching joint
velocities, and the gripper control value. It does not include arbitrary object
positions.

## Success Semantics

The server uses existing `PackGroceryTask.get_reward_done(...)` behavior:

```python
is_success = bool(done or reward > 0)
```

`info["is_success"]` is present on reset and every step.

## Setup

RoCo runtime:

```bash
conda create -n roco python=3.8
conda activate roco
pip install mujoco==2.3.0 dm_control==1.0.8
pip install -r requirements-phase0-roco.txt
```

Gymnasium/LeRobot runtime:

```bash
conda create -n lerobot-roco python=3.12
conda activate lerobot-roco
pip install -e "integrations/lerobot_roco/client[lerobot,test]"
```

## Server Startup

```bash
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --headless
```

Use `MUJOCO_GL=egl` for Linux headless rendering when EGL is available. The
server sets this automatically when `--headless` is passed, but rendering still
depends on the machine having a usable MuJoCo backend.

## Smoke Tests

Diagnostics:

```bash
python scripts/smoke_test_roco_bridge.py --diagnostics-only
```

Reset, hold-step, and render:

```bash
python scripts/smoke_test_roco_bridge.py --steps 5
```

LeRobot compatibility report:

```bash
python scripts/smoke_test_lerobot_preprocessing.py
```

Action replay through the debug native path:

```bash
python scripts/start_roco_bridge.py --enable-debug-commands
python scripts/replay_roco_actions_through_bridge.py --action-file path/to/actions.pkl
```

The replay script may load trusted local pickle artifacts, but only safe array
payloads are transferred over RPC.

## Tests

Common/server fake-backed tests:

```bash
python -m pytest tests/lerobot_roco -m "not client and not integration"
```

Client tests in Python 3.12:

```bash
python -m pytest tests/lerobot_roco -m "client and not integration"
```

Integration tests are marked:

```text
integration
mujoco
lerobot
```

They require the correct runtime and a live bridge.

## Security

- Binds to loopback by default.
- Non-loopback bind requires an explicit unsafe flag.
- No pickle, `eval`, or `exec` over RPC.
- Payload size, array dtype, shape, and byte count are validated.
- Episode ID and step index are validated.
- Remote shutdown is disabled by default.
- Debug native actions are disabled by default.
- Large arrays and session tokens are not logged.

## Known Limitations

- Only `task=pack` is supported.
- Only one active agent is controlled through the public Gym action space.
- Vectorized and async environments are not implemented.
- Auto-starting the server from the client is intentionally not implemented.
- Full LeRobot preprocessing validation must be run in a Python 3.12
  environment with the selected LeRobot version installed.
- Full rendering validation requires a working MuJoCo render backend.

## Phase 0 Passed Checklist

- Server starts in the RoCo runtime.
- `PING`, `HELLO`, and `GET_SPEC` work.
- `reset(seed=...)` returns `pixels`, `agent_pos`, and `is_success`.
- Hold actions step without exception.
- Passive robot controls are included in generated `SimAction` payloads.
- `render()` returns HWC `uint8` RGB.
- Same-seed state digest can be compared.
- Debug native action replay is available only when explicitly enabled.
- Client package imports do not require `rocobench`.
- Normal RoCo imports do not require Gymnasium or LeRobot.

## Phase 1 Path

Phase 1 can add named skills and an RRT-backed skill executor above this
transport. Later learned policies can call:

```python
action = policy.select_action(observation)
observation, reward, terminated, truncated, info = env.step(action)
```
