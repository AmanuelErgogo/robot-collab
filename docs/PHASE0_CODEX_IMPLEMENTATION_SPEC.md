# Phase 0 Implementation Package
## RoCoBench ↔ Gymnasium ↔ LeRobot Compatibility Spike

**Repository:** `AmanuelErgogo/robot-collab`  
**Primary task:** `PackGroceryTask`  
**Primary controlled agent:** configurable; default `Alice`  
**RoCo runtime:** Python 3.8, MuJoCo 2.3.0, dm_control 1.0.8  
**LeRobot client runtime:** Python 3.12+, current supported LeRobot release  
**Phase objective:** Prove that a LeRobot-compatible `gym.Env` can reset, observe, step, render, and close a RoCoBench task through a stable interface without importing LeRobot into the legacy RoCo process.

---

# 1. Executive Summary

Phase 0 is a compatibility and interface-validation phase. It is not a skill-planning phase, dataset phase, or learned-policy phase.

The implementation must prove this end-to-end path:

```text
LeRobot / Gymnasium process (Python 3.12+)
    -> RoCoGymEnv.reset()
    -> local versioned RPC request
    -> RoCo server process (Python 3.8)
    -> PackGroceryTask.reset()
    -> camera images + proprioceptive state
    -> LeRobot-standard raw observation keys

LeRobot / Gymnasium process
    -> continuous NumPy action
    -> local RPC request
    -> action adapter
    -> RoCo SimAction
    -> PackGroceryTask.step()
    -> observation, reward, terminated, truncated, info["is_success"]
```

The current RoCo repository is based on Python 3.8 with MuJoCo 2.3.0 and dm_control 1.0.8. Current LeRobot requires Python 3.12 or newer. Therefore, Phase 0 must use two isolated runtimes connected by a small local transport protocol.

Do not install current LeRobot into the RoCo environment. Do not upgrade RoCo’s simulation stack in Phase 0.

The expected result is:

```python
env = RoCoGymEnv(...)
obs, info = env.reset(seed=0)

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

assert "pixels" in obs
assert "agent_pos" in obs
assert "is_success" in info

env.close()
```

A second verification must replay valid low-level actions produced by the existing RoCo controller through the new bridge and confirm that the bridged environment reaches the same simulator state as direct execution.

---

# 2. Why a Two-Runtime Design Is Required

## 2.1 Current repository constraint

RoCo’s documented environment uses:

```text
Python 3.8
MuJoCo 2.3.0
dm_control 1.0.8
```

Its core environment accepts a `SimAction`, advances physics internally, and returns:

```python
next_obs, reward, done, info
```

The repository is not a Gymnasium environment and does not expose fixed `observation_space` or `action_space` objects.

## 2.2 Current LeRobot constraint

Current LeRobot expects Python 3.12 or newer and evaluates policies through Gymnasium-compatible environments. A benchmark environment should expose:

```python
reset() -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

It also expects:

```text
task
task_description
_max_episode_steps
info["is_success"]
```

Recommended raw observation keys are:

```text
pixels       -> one HWC uint8 image or a dict of HWC uint8 images
agent_pos    -> one fixed-shape proprioceptive state vector
```

## 2.3 Required conclusion

Use:

```text
roco runtime process:
    Python 3.8
    MuJoCo/dm_control
    owns simulator state

client environment process:
    Python 3.12+
    Gymnasium/LeRobot
    owns Gym API and future policy inference
```

Connect them over localhost with a small versioned request-response protocol.

This design also gives later phases a clean place to run:

- LeRobot policy inference;
- LeRobot preprocessing/postprocessing;
- dataset recording;
- evaluation CLI integration;
- action-chunk execution.

---

# 3. Phase 0 Goals

## 3.1 Functional goals

1. Start a RoCo simulation server from the existing Python 3.8 environment.
2. Connect to it from a Python 3.12+ client.
3. Query a machine-readable environment specification.
4. Reset `PackGroceryTask` deterministically by seed.
5. Return fixed-shape camera observations.
6. Return a fixed-shape proprioceptive state vector.
7. Accept a fixed-shape continuous NumPy action.
8. Convert that action into the repository’s existing `SimAction`.
9. Step the existing RoCo environment.
10. Return Gymnasium-compatible transition values.
11. Report `info["is_success"]` on reset and every step.
12. Support `render()` as an RGB NumPy array.
13. Support clean shutdown and cleanup.
14. Verify compatibility using Gymnasium’s environment checker where practical.
15. Verify LeRobot’s observation preprocessing recognizes `pixels` and `agent_pos`.
16. Verify an existing RoCo-generated low-level action can be serialized, transferred, reconstructed, and executed without semantic change.
17. Preserve all legacy RoCo behavior.

## 3.2 Engineering goals

1. Keep protocol, simulator adaptation, and Gymnasium wrapper separated.
2. Use a versioned and self-describing protocol.
3. Avoid unsafe `pickle` across the process boundary.
4. Bind to localhost by default.
5. Validate shapes, dtypes, ranges, sequence numbers, and episode state.
6. Make tests independent of external LLM APIs.
7. Make most tests independent of MuJoCo through fakes.
8. Make the bridge reusable by later Phase 1 and learned-skill work.
9. Produce clear diagnostics for dependency, rendering, connection, and shape failures.
10. Keep the initial implementation limited to one task and one actively controlled robot.

---

# 4. Non-Goals

Do not implement any of the following in Phase 0:

- Phase 1 skill grammar or `SkillPlan`.
- LLM prompting changes.
- LeRobotDataset recording.
- ACT, Diffusion Policy, SmolVLA, or any learned-policy model.
- Policy training.
- Native registration inside LeRobot’s source tree.
- Publishing an EnvHub repository.
- Multiple simultaneous Gym clients.
- Multi-process vectorization.
- Concurrent control of both robots by a learned policy.
- Action chunks.
- Domain randomization beyond current RoCo reset behavior.
- New robot controllers.
- End-effector delta control.
- Torque control.
- Replacement of `PlannedPathPolicy` or `MultiArmRRT`.
- Upgrading MuJoCo, dm_control, or Python in the RoCo environment.
- Rewriting `MujocoSimEnv`.
- Completing the full grocery-packing task with random actions.
- Faking a successful task for the sake of a smoke test.

Phase 0 should establish the transport and environment contracts that later phases will use.

---

# 5. Existing Code That Must Be Inspected

Before editing, inspect at least:

```text
README.md
requirements.txt
setup.py
run_dialog.py

rocobench/__init__.py
rocobench/policy.py
rocobench/subtask_plan.py

rocobench/envs/__init__.py
rocobench/envs/base_env.py
rocobench/envs/env_utils.py
rocobench/envs/robot.py
rocobench/envs/task_pack.py
rocobench/envs/task_pack.xml
rocobench/envs/constants.py
```

Codex must identify and document:

1. `SimAction` fields and validation behavior.
2. `MujocoSimEnv.step(...)` semantics.
3. `MujocoSimEnv.reset(...)` and `seed(...)` behavior.
4. `PackGroceryTask.get_reward_done(...)`.
5. Available camera names and image shapes.
6. `SimRobot` joint and control index fields.
7. Gripper control index and control range for Alice and Bob.
8. How `PlannedPathPolicy.map_qpos_to_ctrl(...)` constructs actions.
9. How passive robots are held during existing multi-arm actions.
10. Whether `all_joint_names` includes gripper joints.
11. Whether action control values are absolute targets.
12. How environment state is saved and restored.

Do not guess action dimensions or actuator limits. Derive them at runtime from the active robot and MuJoCo model.

---

# 6. Required Architecture

## 6.1 Process architecture

```text
┌──────────────────────────────────────────────────────┐
│ LeRobot / Gymnasium runtime — Python 3.12+           │
│                                                      │
│ RoCoGymEnv                                           │
│  ├── Gymnasium spaces                                │
│  ├── request validation                              │
│  ├── RemoteRoCoClient                                │
│  └── LeRobot observation-key compatibility           │
└───────────────────────┬──────────────────────────────┘
                        │ localhost RPC
                        │ versioned metadata + arrays
┌───────────────────────▼──────────────────────────────┐
│ RoCo runtime — Python 3.8                            │
│                                                      │
│ RoCoBridgeServer                                     │
│  ├── PackGroceryTask                                 │
│  ├── RoCoObservationAdapter                          │
│  ├── RoCoActionAdapter                               │
│  ├── episode state machine                           │
│  └── SimAction -> env.step()                         │
└──────────────────────────────────────────────────────┘
```

## 6.2 Required repository layout

Create or adapt this layout:

```text
integrations/
└── lerobot_roco/
    ├── README.md
    ├── common/
    │   ├── __init__.py
    │   ├── protocol.py
    │   ├── serialization.py
    │   ├── errors.py
    │   └── types.py
    ├── roco_runtime/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── env_factory.py
    │   ├── observation_adapter.py
    │   ├── action_adapter.py
    │   ├── episode.py
    │   ├── server.py
    │   └── cli.py
    └── client/
        ├── pyproject.toml
        ├── README.md
        └── src/
            └── lerobot_roco_env/
                ├── __init__.py
                ├── client.py
                ├── config.py
                ├── env.py
                ├── envhub.py
                └── validation.py

scripts/
├── start_roco_bridge.py
├── smoke_test_roco_bridge.py
├── smoke_test_lerobot_preprocessing.py
└── replay_roco_actions_through_bridge.py

tests/
└── lerobot_roco/
    ├── test_protocol.py
    ├── test_serialization.py
    ├── test_action_layout.py
    ├── test_observation_layout.py
    ├── test_episode_state_machine.py
    ├── test_server_dispatch.py
    ├── test_remote_client.py
    ├── test_gym_env.py
    ├── test_error_handling.py
    └── integration/
        ├── test_pack_bridge_reset_step.py
        ├── test_pack_bridge_render.py
        ├── test_action_equivalence.py
        └── test_lerobot_preprocessing.py

requirements-phase0-roco.txt
docs/
└── phase0_lerobot_roco_bridge.md
pytest.ini
```

If the existing repository structure strongly favors another layout, use it consistently and document the difference.

## 6.3 Dependency boundaries

```text
common:
    Python 3.8-compatible
    no MuJoCo
    no Gymnasium
    no LeRobot

roco_runtime:
    Python 3.8
    may import rocobench
    must not import Gymnasium or LeRobot

client:
    Python 3.12+
    may import Gymnasium and optionally LeRobot
    must not import rocobench or dm_control

tests:
    unit tests should use fakes whenever possible
```

The client and server must communicate only through protocol messages and NumPy-compatible data.

---

# 7. Transport and Serialization

## 7.1 Recommended transport

Use ZeroMQ request-reply over localhost:

```text
server: REP
client: REQ
default endpoint: tcp://127.0.0.1:5557
```

Recommended dependencies:

### RoCo runtime

```text
pyzmq
msgpack
```

### Client runtime

```text
pyzmq
msgpack
numpy
gymnasium
```

Do not use Python `pickle` over the network boundary.

If repository inspection reveals an existing safe transport utility, it may be reused, but the same protocol, validation, and security requirements still apply.

## 7.2 NumPy serialization

Implement explicit NumPy serialization:

```python
{
    "__ndarray__": True,
    "dtype": "float32",
    "shape": [14],
    "data": <binary bytes>
}
```

Requirements:

- preserve dtype exactly;
- preserve shape exactly;
- reject object dtype;
- reject dimensions outside a configured maximum;
- reject payloads larger than a configured maximum;
- validate byte length against dtype and shape;
- use C-contiguous data;
- do not allow arbitrary class reconstruction.

Images should remain `uint8`.

## 7.3 Protocol envelope

Every request:

```python
{
    "protocol_version": "0.1",
    "request_id": "<uuid>",
    "command": "RESET",
    "payload": {...}
}
```

Every success response:

```python
{
    "protocol_version": "0.1",
    "request_id": "<same uuid>",
    "ok": True,
    "payload": {...}
}
```

Every failure response:

```python
{
    "protocol_version": "0.1",
    "request_id": "<same uuid>",
    "ok": False,
    "error": {
        "code": "INVALID_ACTION_SHAPE",
        "message": "Expected action shape (8,), received (7,).",
        "details": {...},
        "retryable": False
    }
}
```

Requirements:

- verify response `request_id`;
- reject unsupported protocol versions;
- reject unknown commands;
- return stable error codes;
- never expose full local tracebacks to clients by default;
- log server tracebacks locally;
- use client timeouts;
- use a maximum payload size;
- bind to `127.0.0.1` by default;
- require an explicit unsafe flag to bind to non-loopback interfaces.

## 7.4 Protocol commands

Implement:

```text
PING
HELLO
GET_SPEC
RESET
STEP
RENDER
GET_STATE_DIGEST
CLOSE_EPISODE
SHUTDOWN
```

### `PING`

Response:

```python
{
    "server_time": ...,
    "status": "ready"
}
```

### `HELLO`

Request may contain:

```python
{
    "client_name": "lerobot_roco_env",
    "client_version": "...",
    "supported_protocol_versions": ["0.1"]
}
```

Response:

```python
{
    "selected_protocol_version": "0.1",
    "server_name": "roco-bridge",
    "python_version": "3.8.x",
    "task": "pack",
    "capabilities": [
        "reset",
        "step",
        "render",
        "state_digest"
    ]
}
```

### `GET_SPEC`

Returns the complete environment specification described below.

### `RESET`

Request:

```python
{
    "seed": 0,
    "options": {
        "active_agent": "Alice"
    }
}
```

Response:

```python
{
    "observation": {...},
    "info": {...},
    "episode_id": "...",
    "step_index": 0
}
```

### `STEP`

Request:

```python
{
    "episode_id": "...",
    "step_index": 0,
    "action": np.ndarray
}
```

Response:

```python
{
    "observation": {...},
    "reward": 0.0,
    "terminated": False,
    "truncated": False,
    "info": {...},
    "episode_id": "...",
    "step_index": 1
}
```

Reject stale, skipped, or duplicated step indices.

### `RENDER`

Return one `uint8` HWC image from the configured render camera.

### `GET_STATE_DIGEST`

Return deterministic hashes and selected state vectors for equivalence testing:

```python
{
    "qpos_sha256": "...",
    "qvel_sha256": "...",
    "ctrl_sha256": "...",
    "qpos": np.ndarray,
    "qvel": np.ndarray,
    "ctrl": np.ndarray,
    "timestep": 3
}
```

The full arrays may be disabled outside test/debug mode.

### `CLOSE_EPISODE`

Close the active episode without shutting down the server.

### `SHUTDOWN`

Only allow when:

- server started with `--allow-remote-shutdown`;
- request originates through the existing localhost endpoint;
- optional session token matches.

---

# 8. Environment Specification

Create typed protocol models using Python 3.8-compatible dataclasses.

## 8.1 `ArraySpec`

```python
@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None
    field_names: Tuple[str, ...] = ()
```

## 8.2 `CameraSpec`

```python
@dataclass(frozen=True)
class CameraSpec:
    name: str
    height: int
    width: int
    channels: int
    dtype: str = "uint8"
```

## 8.3 `RoCoEnvSpec`

```python
@dataclass(frozen=True)
class RoCoEnvSpec:
    protocol_version: str
    task: str
    task_description: str
    active_agent: str
    passive_agents: Tuple[str, ...]
    max_episode_steps: int
    effective_fps: float
    cameras: Tuple[CameraSpec, ...]
    observation_state: ArraySpec
    action: ArraySpec
    action_mode: str
    success_semantics: str
    metadata: Mapping[str, Any]
```

The server must return this specification before reset. The client must construct Gymnasium spaces only from this specification.

## 8.4 Required metadata

Include:

```text
robot model/name
agent-to-robot mapping
joint names
joint control indices
joint qpos indices
gripper control index
state field names
action field names
action control ranges
camera names
sim_forward_steps
physics timestep
effective environment-step duration
randomization enabled
reward/success description
RoCo git commit if available
```

No client-side hard-coded action dimension is allowed.

---

# 9. Initial Task and Control Scope

## 9.1 Supported task

Only:

```text
PackGroceryTask
```

Support should be selected by:

```bash
--task pack
```

Fail clearly for unsupported tasks:

```text
Phase 0 bridge currently supports only task=pack.
```

## 9.2 Active agent

Default:

```text
Alice
```

Allow:

```bash
--active-agent Alice
--active-agent Bob
```

The bridge exposes a single-agent action space. Other agents remain passive and are held safely.

## 9.3 Why single active agent

This phase validates:

- observation transport;
- fixed-dimensional control;
- Gymnasium stepping;
- LeRobot preprocessing;
- simulator compatibility.

It does not solve simultaneous multi-agent learned control. The specification must, however, preserve enough metadata to extend to a joint or per-agent action layout later.

---

# 10. Action Representation

## 10.1 Phase 0 action mode

Use:

```text
absolute_joint_position_plus_gripper
```

The fixed action vector contains:

```text
active robot joint actuator targets
+
one gripper actuator target
```

The exact dimension and field ordering must be derived from `SimRobot` and the MuJoCo model.

Example conceptual layout only:

```text
[
  joint_1_target,
  joint_2_target,
  joint_3_target,
  joint_4_target,
  joint_5_target,
  joint_6_target,
  joint_7_target,
  gripper_target
]
```

Do not hard-code seven joints. Derive the dimension.

## 10.2 Action bounds

Build `gym.spaces.Box` from MuJoCo actuator control ranges:

```python
model.actuator_ctrlrange[control_indices]
```

Requirements:

- finite bounds where available;
- preserve native units;
- `dtype=np.float32`;
- reject NaN and Inf;
- configurable clipping policy:
  - default `reject`;
  - optional `clip` for diagnostics;
- include clipping/rejection information in `info`.

Do not normalize actions to `[-1, 1]` in Phase 0. LeRobot processors can handle normalization later. Keeping native controls simplifies equivalence with existing RoCo actions.

## 10.3 `ActionLayout`

Implement:

```python
@dataclass(frozen=True)
class ActionLayout:
    active_agent: str
    active_robot_name: str
    joint_ctrl_indices: Tuple[int, ...]
    joint_qpos_indices: Tuple[int, ...]
    gripper_ctrl_index: int
    field_names: Tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
```

Validate:

```text
len(joint_ctrl_indices) == len(joint_qpos_indices)
action_dim == len(joint_ctrl_indices) + 1
field_names length == action_dim
```

## 10.4 Creating `SimAction`

`RoCoActionAdapter.to_sim_action(action)` must:

1. validate shape;
2. convert to contiguous `float32`;
3. validate bounds;
4. split joints and gripper;
5. construct active-agent controls;
6. construct passive-agent hold controls;
7. create `qpos_idxs` and `qpos_target` suitable for RoCo’s error computation;
8. preserve existing environment expectations for gripper/equality constraints;
9. return a valid `SimAction`.

Pseudo-structure:

```python
SimAction(
    ctrl_idxs=<all controlled actuator indices>,
    ctrl_vals=<active targets + passive hold targets>,
    qpos_idxs=<controlled joint qpos indices>,
    qpos_target=<corresponding joint targets>,
    eq_active_idxs=None,
    eq_active_vals=None,
)
```

The exact merge logic must be based on existing `PlannedPathPolicy.map_qpos_to_ctrl(...)` and robot metadata.

## 10.5 Passive robot behavior

Passive robots must not be left uncontrolled.

At reset, record passive hold targets. On every action:

- command passive arm joints to their hold targets;
- preserve passive gripper state;
- do not change weld/equality state unless required by existing environment semantics.

Add a test proving active-agent actions do not materially move the passive robot beyond a configurable tolerance.

## 10.6 Native `SimAction` codec for equivalence tests

Add an internal test/debug codec:

```python
encode_sim_action(sim_action) -> Dict[str, np.ndarray]
decode_sim_action(payload) -> SimAction
```

This is not the public Gym action space. It exists to verify that existing RRT-generated `SimAction` objects survive serialization and bridged execution exactly.

---

# 11. Observation Representation

## 11.1 Required raw Gym observation

Return:

```python
{
    "pixels": {
        "front": np.ndarray[H, W, 3],  # uint8
        "active_agent": np.ndarray[H, W, 3],  # uint8, if available
    },
    "agent_pos": np.ndarray[state_dim],  # float32
}
```

Camera aliases may map to actual RoCo camera names through configuration.

Recommended default for pack:

```text
front -> teaser
active_agent -> face_panda or face_ur5e, based on active robot
```

Codex must inspect task camera names and create a deterministic mapping. If only one reliable camera is available, return:

```python
{"pixels": image, "agent_pos": state}
```

but multiple-camera dict output is preferred.

## 11.2 Image requirements

- HWC layout;
- exactly three channels;
- `np.uint8`;
- fixed dimensions throughout an episode and across resets;
- contiguous array;
- no alpha channel;
- no BGR conversion unless source is actually BGR;
- validate values in `[0, 255]`;
- camera order is deterministic;
- do not append images to RoCo’s long-lived render buffer merely to generate an observation, if avoidable.

Prefer a direct one-frame rendering helper rather than `render_all_cameras()` if the latter mutates video buffers.

## 11.3 Proprioceptive state

Default state vector:

```text
active-agent joint positions
+
active-agent joint velocities
+
active-agent gripper control/state
```

Codex must inspect whether the gripper is already represented in the joint arrays and avoid accidental duplication. The final field names must be explicit in `RoCoEnvSpec`.

Do not include arbitrary object positions in `agent_pos` during Phase 0. Future phases can add an `environment_state` field or rely on vision.

The output must be:

```python
np.ndarray(shape=(state_dim,), dtype=np.float32)
```

## 11.4 `ObservationLayout`

Implement:

```python
@dataclass(frozen=True)
class ObservationLayout:
    active_agent: str
    state_field_names: Tuple[str, ...]
    camera_aliases: Mapping[str, str]
    state_dim: int
```

## 11.5 Observation adapter

`RoCoObservationAdapter.format(obs)` must:

1. read active robot state from `EnvState`;
2. construct deterministic state order;
3. render configured cameras;
4. validate shape/dtype;
5. return only protocol-safe NumPy/scalar structures;
6. avoid returning dataclasses, MuJoCo objects, sets, or named indexers.

---

# 12. Gymnasium Wrapper

Implement `RoCoGymEnv(gym.Env)` only in the Python 3.12 client package.

## 12.1 Required attributes

```python
metadata = {
    "render_modes": ["rgb_array"],
    "render_fps": <from server spec>
}

self.task = "pack"
self.task_description = "Pack grocery items into the grocery bin."
self._max_episode_steps = <from spec>
self.observation_space = spaces.Dict(...)
self.action_space = spaces.Box(...)
```

## 12.2 Constructor

Suggested signature:

```python
class RoCoGymEnv(gym.Env):
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        active_agent: str = "Alice",
        render_mode: Optional[str] = "rgb_array",
        request_timeout_ms: int = 10_000,
        auto_start_server: bool = False,
        server_command: Optional[Sequence[str]] = None,
        max_episode_steps: Optional[int] = None,
    ):
        ...
```

Phase 0 should support manual server startup first. `auto_start_server` may be implemented if clean and testable, but is not required.

## 12.3 Reset

```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
):
```

Requirements:

- call `super().reset(seed=seed)`;
- close any prior remote episode cleanly;
- send `RESET`;
- validate response against spaces;
- set local episode ID and step index;
- return:
  ```python
  observation, {"is_success": False, ...}
  ```
- preserve server-provided diagnostics;
- never report success immediately unless the actual environment is successful.

## 12.4 Step

```python
def step(self, action):
```

Requirements:

- require an active episode;
- coerce only safe array-like input;
- require exact shape;
- convert to `float32`;
- send current `episode_id` and `step_index`;
- validate response;
- increment local step count;
- map RoCo `done` to `terminated`;
- compute `truncated` from max episode steps;
- always include:
  ```python
  info["is_success"]
  ```
- if server reports environment success:
  ```python
  terminated = True
  ```
- do not silently reset.

## 12.5 Reward and success

Use existing RoCo task reward/done semantics:

```python
is_success = bool(done or reward > 0)
```

If `PackGroceryTask.get_reward_done(...)` has more precise semantics, use that existing definition and document it.

Do not introduce shaped rewards in Phase 0.

## 12.6 Render

`render()` should request or return the latest configured RGB image.

Return:

```python
np.ndarray[H, W, 3], dtype=uint8
```

Do not open a GUI window.

## 12.7 Close

`close()` must:

- send `CLOSE_EPISODE` when appropriate;
- close sockets;
- terminate an auto-started server if that feature was used;
- be idempotent;
- not raise on a previously lost connection during interpreter shutdown.

## 12.8 Environment checker

Run:

```python
from gymnasium.utils.env_checker import check_env
check_env(env, skip_render_check=False)
```

If an external-server environment causes one specific checker incompatibility, document and isolate it. Do not broadly disable checks.

---

# 13. LeRobot Compatibility Validation

Phase 0 is not complete with only a Gymnasium wrapper. It must prove that current LeRobot recognizes the raw observation contract.

## 13.1 Preprocessing smoke test

In the Python 3.12 LeRobot environment:

1. create `RoCoGymEnv`;
2. call `reset`;
3. pass raw observation through the current LeRobot observation preprocessing utility;
4. verify mappings:

```text
pixels dict -> observation.images.<camera>
agent_pos   -> observation.state
```

5. verify tensor/image shapes expected by the installed LeRobot version;
6. create a dummy action matching the action feature shape;
7. step the environment.

Do not hard-code private LeRobot internals without first inspecting the installed version. Prefer public or documented utilities. Isolate version-specific calls in one compatibility module.

## 13.2 Compatibility report

Add:

```python
get_lerobot_compatibility_report()
```

or a script that prints:

```text
LeRobot version
Python version
Gymnasium version
raw observation keys
preprocessed keys
camera shapes
state shape
action shape
task description
is_success presence
```

## 13.3 EnvHub-shaped entry point

Add an `env.py`-compatible function in the client package:

```python
def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Optional[Any] = None,
):
```

Phase 0 requirements:

- support only `n_envs=1`;
- reject `use_async_envs=True`;
- return a single `gym.Env`;
- clearly state that vectorization is not yet implemented.

This prepares the package for EnvHub without publishing it.

Do not modify LeRobot’s `src/lerobot/envs/configs.py` or `factory.py` in Phase 0.

---

# 14. Server Episode State Machine

Implement explicit states:

```text
STARTING
READY
EPISODE_ACTIVE
EPISODE_TERMINATED
CLOSED
ERROR
```

Allowed transitions:

```text
STARTING -> READY
READY -> EPISODE_ACTIVE       via RESET
EPISODE_ACTIVE -> EPISODE_ACTIVE via STEP
EPISODE_ACTIVE -> EPISODE_TERMINATED via success/truncation
EPISODE_ACTIVE -> READY       via CLOSE_EPISODE
EPISODE_TERMINATED -> EPISODE_ACTIVE via RESET
EPISODE_TERMINATED -> READY   via CLOSE_EPISODE
READY -> CLOSED               via SHUTDOWN/server stop
```

Reject:

- `STEP` before `RESET`;
- `STEP` after termination;
- wrong episode ID;
- stale step index;
- reset with unsupported agent;
- simultaneous second client session.

Include state and episode identifiers in error details.

---

# 15. Determinism and Seeding

## 15.1 Reset contract

For the same:

```text
task
active_agent
seed
configuration
```

the initial simulator state and formatted observation should be reproducible within normal rendering determinism limits.

Server reset flow should explicitly use the repository’s existing APIs in the correct order, based on inspection, for example:

```python
env.seed(np_seed=seed)
env.reset(reload=True)
obs = env.get_obs()
```

Do not assume this order; confirm it against current behavior.

## 15.2 State digest

Use SHA-256 over canonical contiguous bytes for:

```text
qpos
qvel
ctrl
```

Include dtype and shape in the hash input.

Tests:

- same seed yields same state digest;
- different seed should generally change randomized object state;
- repeated reset must not leak prior episode controls or equality state.

Camera hashes should not be the only determinism criterion.

---

# 16. Existing-Action Equivalence Test

This is the most important Phase 0 correctness test.

## 16.1 Purpose

Prove that the bridge does not alter RoCo’s low-level action semantics.

## 16.2 Test design

1. Instantiate two identically seeded `PackGroceryTask` environments:
   - direct environment A;
   - server-backed environment B.
2. Generate or construct one valid existing `SimAction`.
   Preferred sources:
   - a `PlannedPathPolicy` action buffer from a canned legacy plan;
   - or a controlled hold/small-joint-target `SimAction` built using existing policy utilities.
3. Execute action directly in A.
4. Encode the same `SimAction` through the test/debug native-action codec.
5. Send it through a debug-only `STEP_NATIVE_ACTION` command or replay utility to B.
6. Compare:
   ```text
   qpos
   qvel
   ctrl
   reward
   done
   selected object poses
   ```
7. Use explicit tolerances for floating-point state.
8. Repeat for:
   - joint movement;
   - gripper open/close if safe;
   - passive-agent hold.

The public Gym action path must also be tested separately.

## 16.3 Debug command restrictions

If implementing `STEP_NATIVE_ACTION`:

- enable only with `--enable-debug-commands`;
- bind to localhost;
- never expose through default production configuration;
- validate all indices against the model;
- do not accept arbitrary equality-constraint indices unless required for the test.

---

# 17. Required CLI Tools

## 17.1 Start server

```bash
python scripts/start_roco_bridge.py \
  --task pack \
  --active-agent Alice \
  --endpoint tcp://127.0.0.1:5557 \
  --seed 0 \
  --headless
```

Options:

```text
--task
--active-agent
--endpoint
--seed
--image-height
--image-width
--cameras
--max-episode-steps
--request-log-level
--headless
--enable-debug-commands
--allow-remote-shutdown
--session-token
```

Reject non-loopback endpoint unless an explicit unsafe flag is provided.

## 17.2 Basic smoke test

From the Python 3.12 client environment:

```bash
python scripts/smoke_test_roco_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --active-agent Alice \
  --steps 5
```

It must:

1. connect;
2. negotiate protocol;
3. print environment spec;
4. reset;
5. validate observation against spaces;
6. execute safe hold actions, not arbitrary damaging random actions by default;
7. render one image;
8. verify `is_success` exists;
9. close cleanly;
10. exit nonzero on failure.

Optional:

```bash
--random-actions
```

must sample within bounds but should be clearly labeled potentially unstable.

## 17.3 LeRobot preprocessing test

```bash
python scripts/smoke_test_lerobot_preprocessing.py \
  --endpoint tcp://127.0.0.1:5557
```

It must print raw and preprocessed feature keys and shapes.

## 17.4 Action replay test

```bash
python scripts/replay_roco_actions_through_bridge.py \
  --endpoint tcp://127.0.0.1:5557 \
  --action-file path/to/actions.pkl
```

Important:

- action files are local trusted RoCo artifacts;
- loading an existing pickle file is allowed only in this offline local script;
- never transfer pickle bytes over RPC;
- convert loaded `SimAction` objects to safe array payloads;
- reject objects of unexpected types.

---

# 18. Configuration

Use dataclasses and environment variables only where appropriate.

## 18.1 Server config

```python
@dataclass
class RoCoBridgeServerConfig:
    endpoint: str = "tcp://127.0.0.1:5557"
    task: str = "pack"
    active_agent: str = "Alice"
    seed: int = 0
    image_height: int = 256
    image_width: int = 256
    camera_aliases: Mapping[str, str] = ...
    max_episode_steps: int = 300
    request_timeout_ms: int = 30_000
    max_payload_bytes: int = 64 * 1024 * 1024
    enable_debug_commands: bool = False
    allow_remote_shutdown: bool = False
    session_token: Optional[str] = None
```

## 18.2 Client config

```python
@dataclass
class RoCoGymConfig:
    endpoint: str = "tcp://127.0.0.1:5557"
    active_agent: str = "Alice"
    render_mode: Optional[str] = "rgb_array"
    request_timeout_ms: int = 30_000
    max_episode_steps: Optional[int] = None
    action_out_of_bounds: str = "reject"
```

Validate all values in `__post_init__`.

---

# 19. Error Codes

At minimum:

```text
UNSUPPORTED_PROTOCOL
UNKNOWN_COMMAND
INVALID_REQUEST
PAYLOAD_TOO_LARGE
SERIALIZATION_ERROR
SERVER_NOT_READY
EPISODE_NOT_ACTIVE
EPISODE_ALREADY_ACTIVE
EPISODE_TERMINATED
EPISODE_ID_MISMATCH
STEP_INDEX_MISMATCH
UNSUPPORTED_TASK
UNKNOWN_AGENT
INVALID_ACTION_SHAPE
INVALID_ACTION_DTYPE
NONFINITE_ACTION
ACTION_OUT_OF_BOUNDS
INVALID_CONTROL_INDEX
OBSERVATION_SHAPE_MISMATCH
RENDER_FAILED
RESET_FAILED
STEP_FAILED
CONNECTION_TIMEOUT
SERVER_DISCONNECTED
INTERNAL_SERVER_ERROR
DEBUG_COMMAND_DISABLED
UNAUTHORIZED_SHUTDOWN
```

Client exceptions should map codes into typed exceptions:

```python
RoCoBridgeError
RoCoConnectionError
RoCoProtocolError
RoCoEpisodeError
RoCoActionError
RoCoObservationError
```

---

# 20. Testing Requirements

## 20.1 Test environments

Use two test categories.

### Server/common unit tests

Run in Python 3.8 RoCo environment:

```bash
python -m pytest tests/lerobot_roco -m "not client and not integration"
```

### Client unit tests

Run in Python 3.12 client environment:

```bash
python -m pytest tests/lerobot_roco -m "client and not integration"
```

If maintaining one test tree across two interpreters becomes fragile, colocate client tests inside the client package and document commands.

## 20.2 Unit tests: protocol

Test:

- valid envelope;
- request ID preservation;
- protocol mismatch;
- unknown command;
- stable error serialization;
- payload size limit;
- missing required keys;
- unexpected keys according to chosen strictness.

## 20.3 Unit tests: NumPy serialization

Test:

- scalar;
- float32 vector;
- int32 vector;
- uint8 image;
- nested dict of images;
- noncontiguous array;
- zero-length array where allowed;
- object dtype rejection;
- malformed shape;
- mismatched byte count;
- oversized array;
- round-trip exact dtype and values.

## 20.4 Unit tests: action layout

Use fake model/robot metadata.

Test:

- derived joint indices;
- gripper index;
- bounds;
- field names;
- exact action dimension;
- active/passive control merge;
- invalid shape;
- NaN/Inf;
- out-of-bounds reject;
- optional clipping;
- passive hold values;
- `SimAction` dtype requirements.

## 20.5 Unit tests: observation layout

Use fake `EnvState` and renderer.

Test:

- deterministic state ordering;
- float32 state;
- uint8 HWC images;
- camera aliases;
- missing camera;
- wrong image dimensions;
- no accidental state-field duplication;
- arrays are contiguous;
- no MuJoCo/dataclass objects leak into payload.

## 20.6 Unit tests: episode state machine

Test every legal and illegal transition, including:

- step before reset;
- reset;
- normal step;
- stale index;
- wrong episode ID;
- step after termination;
- close;
- second reset;
- shutdown.

## 20.7 Unit tests: server dispatch

Use fake environment factory.

Test:

- PING;
- HELLO;
- GET_SPEC;
- RESET;
- STEP;
- RENDER;
- CLOSE_EPISODE;
- server-side exception mapping;
- request logging does not include binary image data or secrets.

## 20.8 Unit tests: remote client

Use an in-process fake ZeroMQ server or transport abstraction.

Test:

- timeout;
- malformed response;
- mismatched request ID;
- server error mapping;
- reconnect policy;
- close idempotence.

## 20.9 Unit tests: Gym wrapper

Use fake client.

Test:

- spaces built from spec;
- reset contract;
- step contract;
- action conversion;
- terminated/truncated mapping;
- `is_success` always present;
- render;
- close;
- checker-compatible seed behavior;
- step before reset error.

## 20.10 MuJoCo integration tests

Mark:

```python
@pytest.mark.integration
@pytest.mark.mujoco
```

Tests:

1. `PackGroceryTask` server starts.
2. Client obtains spec.
3. Reset produces valid observation.
4. Hold action steps without exception.
5. Active robot control changes as expected.
6. Passive robot remains within tolerance.
7. Same seed state digest matches.
8. Different seed changes randomized state.
9. Render returns valid image.
10. Repeated reset does not leak state.
11. Action-equivalence test passes.
12. Client close and server shutdown are clean.

## 20.11 LeRobot integration test

Mark:

```python
@pytest.mark.integration
@pytest.mark.lerobot
```

Run in Python 3.12 with LeRobot installed.

Test:

- raw keys are accepted;
- preprocessing creates image/state keys;
- task description is available;
- one dummy/hold action can be postprocessed or directly sent according to current public interfaces;
- no policy checkpoint is required.

## 20.12 Gymnasium checker

Run in integration tests where possible:

```python
check_env(env)
```

Document any justified exclusions.

---

# 21. Dependency Files and Environment Setup

## 21.1 RoCo runtime requirements

Create:

```text
requirements-phase0-roco.txt
```

Suggested content, adjusted to compatible versions after installation testing:

```text
-r requirements.txt
pyzmq==<Python-3.8-compatible-version>
msgpack==<Python-3.8-compatible-version>
pytest==7.4.4
```

Do not update the main RoCo requirements unless necessary.

## 21.2 Client package

Create `integrations/lerobot_roco/client/pyproject.toml`:

```toml
[project]
name = "lerobot-roco-env"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy",
    "gymnasium",
    "pyzmq",
    "msgpack",
]

[project.optional-dependencies]
lerobot = [
    "lerobot>=0.5.2,<0.6",
]
test = [
    "pytest",
]
```

Before pinning, inspect the currently installed/target LeRobot version and current packaging metadata. If the validated version differs, pin the exact tested range and document it.

## 21.3 Setup documentation

Document two environments explicitly.

Example:

```bash
conda create -n roco python=3.8
conda activate roco
pip install mujoco==2.3.0 dm_control==1.0.8
pip install -r requirements-phase0-roco.txt
```

```bash
conda create -n lerobot-roco python=3.12
conda activate lerobot-roco
pip install -e "integrations/lerobot_roco/client[lerobot,test]"
```

Do not imply that one environment can satisfy both stacks.

---

# 22. Headless Rendering Requirements

Support Linux headless execution.

Codex must inspect the current rendering mechanism and document tested settings, such as:

```text
MUJOCO_GL=egl
```

or a software-rendering fallback where supported.

Requirements:

- do not silently assume an X display;
- fail with a diagnostic naming the attempted rendering backend;
- basic unit tests must not require rendering;
- rendering integration tests may be skipped with a precise reason when no backend is available;
- server should start in a state-only diagnostic mode only if explicitly configured;
- LeRobot compatibility acceptance still requires at least one actual RGB render in a suitable environment.

---

# 23. Logging and Diagnostics

## 23.1 Server logs

Include:

```text
protocol version
endpoint
task
active agent
episode ID
seed
request ID
command
step index
latency
action shape
reward
terminated/truncated
error code
```

Do not log:

- full image bytes;
- full large arrays;
- session token;
- API keys;
- arbitrary client payloads.

## 23.2 Client logs

At debug level:

```text
connect
HELLO result
GET_SPEC summary
reset latency
step latency
server errors
close
```

## 23.3 Health report

Add command:

```bash
python scripts/smoke_test_roco_bridge.py --diagnostics-only
```

Print:

- Python versions;
- package versions;
- endpoint status;
- protocol version;
- task;
- action/state/image shapes;
- render backend;
- server PID if available.

---

# 24. Artifacts

The smoke and integration tools should optionally save:

```text
artifacts/phase0/
├── env_spec.json
├── reset_info.json
├── transition_000.json
├── render_000.png
├── state_digest_before.json
├── state_digest_after.json
├── compatibility_report.json
└── action_equivalence_report.json
```

Do not serialize images directly into JSON.

`action_equivalence_report.json` should contain:

```json
{
  "passed": true,
  "qpos_max_abs_error": 0.0,
  "qvel_max_abs_error": 0.0,
  "ctrl_max_abs_error": 0.0,
  "reward_equal": true,
  "done_equal": true
}
```

---

# 25. Backward Compatibility

Required guarantees:

1. `run_dialog.py` remains behaviorally unchanged.
2. Existing LLM prompting remains unchanged.
3. Existing parser and feedback code remain unchanged.
4. Existing `PlannedPathPolicy` remains unchanged unless a tiny, backward-compatible helper extraction is essential.
5. Existing `MujocoSimEnv.step(...)` signature remains unchanged.
6. Existing `SimAction` remains the simulator action type.
7. Existing task constructors continue to work.
8. Main dependency environment remains Python 3.8.
9. No LeRobot or Gymnasium import occurs when importing normal RoCo modules.
10. Bridge code is opt-in.
11. Existing API-key behavior must not affect bridge tests.
12. Unit test collection must not import LLM clients or require key files.

Prefer adapters over modifying core classes.

---

# 26. Documentation Requirements

Create:

```text
docs/phase0_lerobot_roco_bridge.md
```

Include:

1. objective and non-goals;
2. two-runtime rationale;
3. architecture diagram;
4. protocol commands;
5. observation schema;
6. action schema;
7. passive-agent behavior;
8. task success semantics;
9. environment setup;
10. server startup;
11. client smoke test;
12. LeRobot preprocessing test;
13. action-equivalence test;
14. headless rendering;
15. troubleshooting;
16. test commands;
17. security considerations;
18. known limitations;
19. how Phase 1 will build on this;
20. how a future learned policy will call `select_action()` and `env.step()`.

Include a concise “Phase 0 passed” checklist.

---

# 27. Security Requirements

Even though this is a local research bridge:

- bind to loopback by default;
- no network pickle;
- no `eval` or `exec`;
- payload size limit;
- array shape and byte-count validation;
- explicit non-loopback override;
- shutdown disabled by default;
- optional session token;
- no shell command from RPC;
- no arbitrary file path from RPC;
- no dynamic import from client payload;
- no arbitrary MuJoCo index writes;
- debug native action command disabled by default;
- no sensitive values in logs.

---

# 28. Performance Requirements

Phase 0 is correctness-first, but measure:

```text
reset latency
render latency
step RPC latency
serialization latency
payload size per observation
```

Acceptance guidelines:

- no unbounded memory growth over 100 hold steps;
- no accumulating render buffers from observation-only rendering;
- one request outstanding per client;
- arrays transferred in binary form, not JSON lists;
- no repeated environment reconstruction on every step;
- environment reconstruction on reset only if required by current task behavior.

Add a lightweight 100-step stability script or test.

---

# 29. Acceptance Criteria

Phase 0 is complete only when all applicable boxes are satisfied.

## Architecture

- [ ] RoCo and LeRobot run in separate Python environments.
- [ ] No LeRobot import exists in RoCo runtime code.
- [ ] No `rocobench` import exists in client package.
- [ ] Protocol is versioned.
- [ ] Network serialization does not use pickle.

## Environment behavior

- [ ] Server starts with `PackGroceryTask`.
- [ ] Client retrieves environment spec.
- [ ] `reset(seed=...)` works.
- [ ] Observation contains valid `pixels` and `agent_pos`.
- [ ] Action space is derived from robot/model metadata.
- [ ] A valid hold action executes.
- [ ] Passive robot remains held.
- [ ] `render()` returns HWC uint8 RGB.
- [ ] `info["is_success"]` is present on reset and every step.
- [ ] `terminated` and `truncated` semantics are correct.
- [ ] `close()` is idempotent.

## Correctness

- [ ] Same-seed state digest is reproducible.
- [ ] Action shape/range validation works.
- [ ] Existing `SimAction` equivalence test passes.
- [ ] Repeated reset does not leak prior episode state.
- [ ] Stale episode and step IDs are rejected.

## LeRobot

- [ ] Current LeRobot preprocessing recognizes raw keys.
- [ ] Preprocessed image and state shapes are correct.
- [ ] `task_description` is available.
- [ ] EnvHub-shaped `make_env()` returns one environment.
- [ ] No policy checkpoint is needed for the compatibility test.

## Quality

- [ ] Unit tests pass in both runtimes.
- [ ] Integration tests are clearly marked.
- [ ] Smoke scripts exit nonzero on failure.
- [ ] Core legacy tests or import smoke checks still pass.
- [ ] Documentation is complete.
- [ ] Exact commands and results are reported.
- [ ] Blocked rendering/simulator tests are reported honestly.

---

# 30. Required Implementation Order

Codex should work in this order:

1. Inspect repository and current runtime constraints.
2. Confirm exact robot/control/state metadata for `PackGroceryTask`.
3. Write a concise implementation plan.
4. Add common protocol types and serialization.
5. Add protocol/serialization tests.
6. Add server configuration and episode state machine.
7. Add fake-environment server dispatch tests.
8. Add action-layout and action-adapter code.
9. Add observation-layout and observation-adapter code.
10. Add RoCo environment factory.
11. Add actual server and CLI.
12. Add Python 3.12 client package.
13. Add Gymnasium wrapper.
14. Add client/Gym unit tests with fakes.
15. Add server/client smoke scripts.
16. Add MuJoCo reset/step/render integration tests.
17. Add native-action equivalence test.
18. Add LeRobot preprocessing smoke test.
19. Run Gymnasium checker.
20. Add documentation and environment setup.
21. Review for legacy regressions and import leakage.
22. Report commands, results, skipped tests, and remaining risks.

Do not begin by modifying `run_dialog.py`. Ideally, Phase 0 should not modify it at all.

---

# 31. Copy-Ready Master Prompt for Codex

Paste the following into Codex from the repository root.

---

## Codex Prompt

Implement **Phase 0: RoCoBench ↔ Gymnasium ↔ LeRobot compatibility** according to `PHASE0_CODEX_IMPLEMENTATION_SPEC.md`.

### Objective

Prove that current LeRobot can interact with `PackGroceryTask` through a standard Gymnasium environment while preserving the existing RoCo simulator and controller semantics.

The current RoCo stack uses Python 3.8, MuJoCo 2.3.0, and dm_control 1.0.8. Current LeRobot requires Python 3.12+. Therefore, implement a two-process architecture:

```text
Python 3.12+:
    RoCoGymEnv
    Gymnasium
    LeRobot preprocessing/future policy inference

localhost RPC

Python 3.8:
    RoCoBridgeServer
    PackGroceryTask
    SimAction
    MuJoCo/dm_control
```

Do not install or import LeRobot in the RoCo runtime. Do not upgrade RoCo’s simulation dependencies in this phase.

### First actions

Before editing:

1. Read `PHASE0_CODEX_IMPLEMENTATION_SPEC.md` fully.
2. Inspect:
   - `README.md`
   - `requirements.txt`
   - `run_dialog.py`
   - `rocobench/policy.py`
   - `rocobench/subtask_plan.py`
   - `rocobench/envs/base_env.py`
   - `rocobench/envs/env_utils.py`
   - `rocobench/envs/robot.py`
   - `rocobench/envs/task_pack.py`
   - `rocobench/envs/task_pack.xml`
   - all relevant `__init__.py` files.
3. Determine and report:
   - exact `SimAction` semantics;
   - Alice/Bob robot models;
   - joint qpos/control indices;
   - gripper control indices and ranges;
   - available camera names;
   - reset/seed behavior;
   - reward/done behavior;
   - effective environment-step duration.
4. Produce a concise file-by-file plan with risks.
5. Then implement the complete phase. Do not stop after planning.

### Mandatory architecture

Create a versioned localhost request-response bridge.

Recommended transport:

```text
ZeroMQ REQ/REP
msgpack metadata
explicit binary NumPy serialization
```

Do not use pickle over the RPC boundary.

The server owns the simulator. The client owns the Gymnasium environment.

Keep:

```text
common protocol code
RoCo runtime adapter code
Python 3.12 client/Gym code
```

in separate modules with clean dependency boundaries.

### Supported scope

Support only:

```text
task = pack
one active agent at a time
default active agent = Alice
```

The passive robot must be actively held at safe control targets.

Expose a fixed action vector:

```text
absolute active-robot joint actuator targets
+
one gripper actuator target
```

Derive action dimension, field names, indices, and bounds from the actual robot and MuJoCo model. Do not hard-code dimensions.

Create a `SimAction` through an action adapter and execute the existing:

```python
env.step(sim_action)
```

Do not replace or modify the simulator controller semantics.

### Observation contract

Return:

```python
{
    "pixels": {
        "<camera_alias>": np.ndarray[H, W, 3],  # uint8
        ...
    },
    "agent_pos": np.ndarray[state_dim],         # float32
}
```

Use fixed camera shapes and deterministic state ordering.

The state should contain active-agent joint position, velocity, and gripper state/control without accidental duplication. Document exact field names in the environment specification.

Do not send MuJoCo objects, dataclasses, sets, or named indexers over RPC.

### Gymnasium contract

Implement a Python 3.12 `RoCoGymEnv(gym.Env)` with:

```text
task
task_description
_max_episode_steps
observation_space
action_space
reset
step
render
close
```

`reset()` returns:

```python
observation, info
```

`step()` returns:

```python
observation, reward, terminated, truncated, info
```

`info["is_success"]` must exist on reset and every step.

Use existing RoCo reward/done semantics. Do not fake success.

### Protocol

Implement:

```text
PING
HELLO
GET_SPEC
RESET
STEP
RENDER
GET_STATE_DIGEST
CLOSE_EPISODE
SHUTDOWN
```

Use:

```text
protocol_version
request_id
command
payload
```

and stable typed error codes.

Validate:

- protocol version;
- request ID;
- episode ID;
- step index;
- payload size;
- ndarray shape/dtype/bytes;
- action shape/range/finite values;
- command legality in current episode state.

Bind to `127.0.0.1` by default. Require explicit unsafe configuration for non-loopback binding. Disable remote shutdown by default.

### Environment specification

The server must expose a complete machine-readable spec containing:

- task and task description;
- active/passive agents;
- max steps and effective FPS;
- camera names/shapes;
- state shape/dtype/field names;
- action shape/dtype/field names/bounds;
- robot name/model;
- joint and control indices;
- gripper index;
- simulator timing;
- success semantics.

The Gym client must build spaces from this spec, not hard-coded values.

### Existing-action equivalence

Add a debug/test path proving that an existing RoCo `SimAction` executed directly and the same action serialized through the bridge produce equivalent simulator results.

Compare:

```text
qpos
qvel
ctrl
reward
done
selected object poses
```

Use explicit tolerances.

Do not transfer pickle. A local offline replay script may load trusted existing `.pkl` action files, convert them to safe arrays, and send only safe payloads.

### LeRobot validation

In a Python 3.12 environment with the current supported LeRobot version:

1. reset `RoCoGymEnv`;
2. run the raw observation through LeRobot’s current documented preprocessing path;
3. confirm:
   ```text
   pixels -> observation image key(s)
   agent_pos -> observation.state
   ```
4. print and assert feature shapes;
5. step one safe hold action;
6. require no policy checkpoint.

Isolate version-specific LeRobot imports in one compatibility module.

Also provide an EnvHub-shaped:

```python
make_env(n_envs=1, use_async_envs=False, cfg=None)
```

that returns a single Gym environment. Reject unsupported vector/async options. Do not modify LeRobot’s internal factory/config files in Phase 0.

### Tests

Add comprehensive tests for:

- protocol envelopes;
- NumPy serialization;
- payload limits;
- action layout and bounds;
- passive-agent hold;
- observation layout and images;
- episode state machine;
- server dispatch with fakes;
- remote client errors/timeouts;
- Gym reset/step/render/close;
- deterministic seed/state digest;
- repeated reset;
- MuJoCo bridge smoke test;
- action equivalence;
- LeRobot preprocessing;
- Gymnasium environment checker where practical.

Mark simulator tests:

```python
@pytest.mark.integration
@pytest.mark.mujoco
```

Mark LeRobot tests:

```python
@pytest.mark.integration
@pytest.mark.lerobot
```

Most unit tests must run without MuJoCo, rendering, an LLM, or API keys.

### Required scripts

Implement:

```bash
python scripts/start_roco_bridge.py ...
python scripts/smoke_test_roco_bridge.py ...
python scripts/smoke_test_lerobot_preprocessing.py ...
python scripts/replay_roco_actions_through_bridge.py ...
```

The normal smoke test should use a safe hold action by default.

### Backward compatibility

Do not change behavior in:

- `run_dialog.py`;
- prompting;
- parser;
- feedback;
- `PlannedPathPolicy`;
- `MultiArmRRT`;
- existing environment stepping.

Avoid importing Gymnasium/LeRobot from normal RoCo modules. The bridge must be opt-in.

### Dependency management

Create a Python 3.8 bridge requirements file and a separate Python 3.12 client package.

Do not claim one environment supports both stacks.

Pin versions only after validating compatibility. Document exact commands.

### Verification before completion

Run and report:

1. Python 3.8 server/common unit tests.
2. Python 3.12 client unit tests.
3. `python -m compileall` on new Python modules in both compatible environments.
4. server startup and PING/HELLO/GET_SPEC.
5. reset and at least five safe hold steps.
6. render validation.
7. deterministic same-seed state digest.
8. repeated-reset state-leak test.
9. action-equivalence test.
10. Gymnasium checker.
11. LeRobot preprocessing smoke test.
12. import smoke test proving normal RoCo imports do not require Gymnasium or LeRobot.

If MuJoCo, EGL, Python 3.12, or LeRobot is unavailable, complete all possible work and tests, show the exact blocked commands and errors, and do not claim those tests passed.

### Definition of done

Do not declare completion until:

- all available unit tests pass;
- smoke scripts have meaningful nonzero failure exits;
- protocol is safe and versioned;
- action/state layouts are derived, not guessed;
- `is_success` is always present;
- passive-agent hold is tested;
- direct-vs-bridge action equivalence is tested;
- current LeRobot preprocessing is validated or explicitly reported blocked;
- documentation covers setup, architecture, commands, troubleshooting, security, and limitations;
- final report lists files changed, commands, results, skipped/blocked checks, and remaining risks.

---

# 32. Suggested Repository `AGENTS.md`

Add or merge these instructions at the repository root.

```markdown
# AGENTS.md

## Repository purpose

This repository implements RoCo and RoCoBench, a Python 3.8 MuJoCo/dm_control multi-robot collaboration system.

## Phase 0 objective

Implement a compatibility bridge allowing a Python 3.12+ Gymnasium/LeRobot client to control a Python 3.8 RoCoBench simulator process.

## Runtime separation

- RoCo runtime remains Python 3.8.
- Current LeRobot client runs in Python 3.12+.
- Do not install LeRobot into the RoCo environment.
- Do not import `rocobench` from the client package.
- Do not import Gymnasium or LeRobot from normal RoCo modules.
- Communicate through the versioned local protocol only.

## Architecture rules

- Simulator state is owned by the server.
- Gym episode state is mirrored and validated by the client.
- Derive action/state dimensions from runtime metadata.
- Use `SimAction` and existing `env.step()` semantics.
- Keep one active agent and hold passive agents in Phase 0.
- Use LeRobot-standard raw keys `pixels` and `agent_pos`.
- Always return `info["is_success"]`.
- Keep protocol, serialization, simulator adapters, and Gym wrapper separate.

## Security

- No pickle over RPC.
- No `eval` or `exec`.
- Bind to localhost by default.
- Validate protocol version, payload size, ndarray dtype/shape, action bounds, episode ID, and step index.
- Disable debug commands and remote shutdown by default.
- Do not log binary arrays, tokens, or secrets.

## Compatibility

- Preserve Python 3.8 syntax in shared/server modules.
- Client package may use Python 3.12 syntax only inside the isolated client package.
- Do not modify legacy LLM behavior.
- Avoid unrelated refactors.

## Verification

Run server/common tests in the RoCo environment and client/LeRobot tests in the Python 3.12 environment.

Do not claim MuJoCo, rendering, Gym checker, or LeRobot integration passed unless each command actually ran.
```

---

# 33. Follow-Up Review Prompt for Codex

Run this in a separate Codex turn after implementation.

```text
Review the complete Phase 0 diff as a senior robotics infrastructure and ML systems engineer.

Do not edit first. Produce findings ordered by severity with exact file and line references.

Focus on:

1. accidental LeRobot/Gymnasium imports in the Python 3.8 RoCo path;
2. accidental rocobench imports in the Python 3.12 client package;
3. unsafe serialization, pickle, eval, or arbitrary reconstruction;
4. protocol version/request/episode/step validation gaps;
5. incorrect action dimension, ordering, bounds, or gripper handling;
6. passive robot not being held;
7. mismatch between SimAction qpos targets and control targets;
8. observation state duplication or nondeterministic field ordering;
9. incorrect image channel order, shape, dtype, or render-buffer growth;
10. incorrect terminated/truncated/is_success semantics;
11. reset/seed/state leakage;
12. socket timeout, close, reconnect, or shutdown leaks;
13. debug command exposure;
14. Python 3.8 syntax incompatibilities in common/server code;
15. reliance on private LeRobot internals without isolation;
16. tests that pass only because they mock away the contract being tested;
17. missing direct-vs-bridge action equivalence;
18. legacy RoCo regressions.

After presenting findings, fix all confirmed issues, run the relevant tests in each runtime, and report the results and remaining limitations.
```

---

# 34. Follow-Up Verification Prompt for Codex

```text
Verify Phase 0 end to end with no external LLM and no API key.

Use the correct separate environments.

A. RoCo Python 3.8 runtime
1. Run server/common unit tests.
2. Run compileall on common and roco_runtime modules.
3. Start the bridge for PackGroceryTask on localhost.
4. Verify PING, HELLO, and GET_SPEC.
5. Save the environment spec.

B. Python 3.12 Gym/LeRobot runtime
1. Run client unit tests.
2. Run compileall on the client package.
3. Connect RoCoGymEnv.
4. Run Gymnasium check_env where supported.
5. Reset twice with the same seed and compare state digests.
6. Reset with a different seed and inspect expected randomized changes.
7. Execute five safe hold actions.
8. Verify action/state/image shapes and dtypes on every transition.
9. Verify info["is_success"] is present.
10. Render and save one RGB image.
11. Verify close is idempotent.

C. Correctness
1. Run the direct-vs-bridge native SimAction equivalence test.
2. Report max absolute qpos, qvel, and ctrl differences.
3. Verify passive robot movement remains below tolerance.
4. Run 100 safe steps and check for memory/render-buffer growth.

D. LeRobot
1. Print installed LeRobot version.
2. Run the current documented preprocessing path.
3. Show raw keys and resulting LeRobot feature keys.
4. Show image, state, and action shapes.
5. Call the local EnvHub-shaped make_env(n_envs=1).
6. Do not use a model checkpoint.

E. Regression
1. Import normal rocobench modules without Gymnasium or LeRobot installed.
2. Confirm run_dialog CLI help still works.
3. Confirm no existing core behavior was modified unintentionally.

Report:
- every command;
- exit code;
- passed/failed/skipped counts;
- artifact paths;
- exact blocked command and error for any unavailable dependency.

Do not describe a test as passed unless it actually executed.
```

---

# 35. Expected Phase 0 End-State

Server:

```python
server = RoCoBridgeServer(
    config=RoCoBridgeServerConfig(
        task="pack",
        active_agent="Alice",
    )
)
server.serve_forever()
```

Client:

```python
from lerobot_roco_env import RoCoGymEnv

env = RoCoGymEnv(
    endpoint="tcp://127.0.0.1:5557",
    active_agent="Alice",
)

obs, info = env.reset(seed=0)

assert obs["agent_pos"].dtype == np.float32
assert all(image.dtype == np.uint8 for image in obs["pixels"].values())
assert info["is_success"] is False

action = env.hold_action()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

LeRobot compatibility:

```text
raw Gym observation
    -> pixels
    -> agent_pos
    -> LeRobot preprocessing
    -> observation.images.<camera>
    -> observation.state
```

This establishes the contract needed by later work:

```text
Phase 1:
    named skills and RRT-backed SkillExecutor

Later:
    LeRobot policy.select_action(...)
        -> RoCoGymEnv.step(action)
        -> RoCo simulator
```

The key Phase 0 result is not task success. It is a verified, deterministic, safe, fixed-shape, LeRobot-compatible interaction loop with equivalent low-level simulator semantics.
