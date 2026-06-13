# Tutorial: How to Create a New Task in RoCoBench

This tutorial explains how to create a new task in the `robot-collab` / RoCoBench codebase.

RoCoBench tasks are not simple Gym environments. A task combines:

1. a MuJoCo XML scene,
2. a Python class that subclasses `MujocoSimEnv`,
3. object/robot/task-specific helper methods,
4. LLM-facing prompts,
5. success and feedback logic,
6. registration in the runner.

The typical task structure is:

```text
rocobench/envs/task_<task_name>.xml
rocobench/envs/task_<task_name>.py
rocobench/envs/__init__.py
run_dialog.py
```

Existing examples live in `rocobench/envs/`, including `task_pack.py`, `task_sort.py`, `task_sandwich.py`, and their matching XML files.

---

## 1. What a RoCoBench task must provide

Every task should define:

```text
Simulation layer
    - MuJoCo XML scene
    - robots
    - movable objects
    - target sites
    - cameras
    - collision geoms

Robot-execution layer
    - grasp sites
    - target poses
    - object joints
    - robot reach ranges
    - allowed collisions

LLM-planning layer
    - task context
    - action prompt
    - observation description
    - agent-specific prompt
    - task-specific feedback

Evaluation layer
    - reward
    - done/success condition
```

All concrete tasks inherit from:

```python
MujocoSimEnv
```

The base class handles common MuJoCo loading, reset, stepping, rendering, observations, and simulation state management. Your task class fills in the task-specific parts.

---

## 2. Example task: Block Bin Task

We will create a simple task:

> Alice and Bob must place two colored blocks into matching bins.

Example goal:

```text
Put red_block into red_bin.
Put blue_block into blue_bin.
```

We will create:

```text
rocobench/envs/task_block_bin.xml
rocobench/envs/task_block_bin.py
```

and register it as:

```bash
python run_dialog.py --task block_bin
```

---

# Part A — Create the MuJoCo XML scene

Create:

```text
rocobench/envs/task_block_bin.xml
```

For a real implementation, copy an existing XML such as `task_pack.xml` or `task_sort.xml` and simplify it. This is safer than writing the whole MuJoCo scene from zero because the robot definitions, asset includes, cameras, table, and keyframes are already configured in existing task XML files.

Your XML must include:

```text
1. Robots
2. Table/workspace
3. Movable objects
4. Target bins/sites
5. Cameras
6. Keyframe/home pose
7. Object free joints
8. Grasp sites
```

Important naming conventions:

```text
object body name:
    red_block

object joint:
    red_block_joint

object grasp site:
    red_block_top

target site:
    red_bin
```

The Python task will later call methods such as:

```python
get_object_joint_name("red_block")  # red_block_joint
get_grasp_site("red_block")         # red_block_top
get_target_pos("Alice", "red_bin")  # site position
```

A simplified object/target section may look conceptually like:

```xml
<body name="red_block" pos="0.35 0.25 0.05">
    <joint name="red_block_joint" type="free"/>
    <geom name="red_block_geom"
          type="box"
          size="0.025 0.025 0.025"
          rgba="1 0 0 1"
          mass="0.05"/>
    <site name="red_block_top"
          pos="0 0 0.04"
          size="0.01"
          rgba="1 1 0 1"/>
</body>

<body name="blue_block" pos="0.35 -0.25 0.05">
    <joint name="blue_block_joint" type="free"/>
    <geom name="blue_block_geom"
          type="box"
          size="0.025 0.025 0.025"
          rgba="0 0 1 1"
          mass="0.05"/>
    <site name="blue_block_top"
          pos="0 0 0.04"
          size="0.01"
          rgba="1 1 0 1"/>
</body>

<site name="red_bin"
      pos="0.75 0.25 0.05"
      size="0.04"
      rgba="1 0 0 0.5"/>

<site name="blue_bin"
      pos="0.75 -0.25 0.05"
      size="0.04"
      rgba="0 0 1 0.5"/>
```

In practice, copy a working task XML and modify object/target sections incrementally.

---

# Part B — Create the Python task file

Create:

```text
rocobench/envs/task_block_bin.py
```

Start with imports:

```python
from typing import Dict, Tuple, Optional
import numpy as np

from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import (
    UR5E_ROBOTIQ_CONSTANTS,
    PANDA_CONSTANTS,
)
```

---

## 1. Define task constants

```python
BLOCK_BIN_OBJECTS = [
    "red_block",
    "blue_block",
]

BLOCK_BIN_TARGETS = {
    "red_block": "red_bin",
    "blue_block": "blue_bin",
}

BLOCK_BIN_SITES = [
    "red_bin",
    "blue_bin",
]
```

---

## 2. Define the task context

```python
BLOCK_BIN_TASK_CONTEXT = """
Two robots, Alice and Bob, must place colored blocks into their matching bins.
The red block should go to the red bin.
The blue block should go to the blue bin.
The robots should coordinate so they do not both try to manipulate the same block.
"""
```

---

## 3. Define the action prompt

```python
BLOCK_BIN_ACTION_SPACE = """
[Action Options]

1) PICK <object>
   Pick one block.

2) PLACE <object> <target>
   Place the held block at the target bin.

3) PICK <object> PLACE <target>
   Pick the object and place it at the target bin.

4) WAIT
   Do nothing while the other robot acts.

[Valid Objects]
red_block
blue_block

[Valid Targets]
red_bin
blue_bin

[Action Output Instruction]
After discussion, output exactly one ACTION per robot:

EXECUTE
NAME Alice ACTION <action>
NAME Bob ACTION <action>

Examples:
EXECUTE
NAME Alice ACTION PICK red_block PLACE red_bin
NAME Bob ACTION PICK blue_block PLACE blue_bin

or:

EXECUTE
NAME Alice ACTION PICK red_block PLACE red_bin
NAME Bob ACTION WAIT
"""
```

If you later switch to the skill-based interface, this would become:

```text
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=red_block, container=red_bin)
```

But for creating a normal RoCo task, use the existing `PICK`, `PLACE`, and `WAIT` format first.

---

# Part C — Implement the task class

```python
class BlockBinTask(MujocoSimEnv):
    def __init__(
        self,
        filepath: str = "rocobench/envs/task_block_bin.xml",
        **kwargs,
    ):
        self.robot_names = ["ur5e_robotiq", "panda"]

        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
        }

        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda",
        }

        self.robots = {}

        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()
        panda_config = PANDA_CONSTANTS.copy()

        self.object_names = BLOCK_BIN_OBJECTS
        self.target_names = list(BLOCK_BIN_TARGETS.values())

        super(BlockBinTask, self).__init__(
            filepath=filepath,
            task_objects=BLOCK_BIN_OBJECTS,
            agent_configs={
                "ur5e_robotiq": robotiq_config,
                "panda": panda_config,
            },
            **kwargs,
        )

        self.target_xposes = {}
        for target in self.target_names:
            self.target_xposes[target] = self.physics.data.site(target).xpos.copy()

        self.robots[self.robot_name_map["ur5e_robotiq"]] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )

        self.robots[self.robot_name_map["panda"]] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )

        self.success_threshold = 0.08
```

---

# Part D — Implement required geometry methods

These methods connect language names like `red_block` and `red_bin` to MuJoCo positions.

## 1. Target position

```python
def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]:
    if target_name in self.object_names:
        site_name = f"{target_name}_top"
    elif target_name in self.target_names:
        site_name = target_name
    else:
        return None

    try:
        return self.physics.data.site(site_name).xpos.copy()
    except KeyError:
        print(f"KeyError: site {site_name} not found")
        return None
```

## 2. Target orientation

For a simple top-down placement task, return a default orientation:

```python
def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
    return np.array([1, 0, 0, 0])
```

## 3. Grasp site

```python
def get_grasp_site(self, obj_name: str) -> Optional[str]:
    if obj_name in self.object_names:
        return f"{obj_name}_top"
    return None
```

## 4. Object joint name

```python
def get_object_joint_name(self, obj_name: str) -> str:
    return f"{obj_name}_joint"
```

## 5. Robot name helpers

```python
def get_robot_name(self, agent_name):
    return self.robot_name_map_inv[agent_name]

def get_agent_name(self, robot_name):
    return self.robot_name_map[robot_name]
```

## 6. Reach ranges

These are used by the feedback manager to reject unreachable plans before motion planning.

```python
def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
    if robot_name == "ur5e_robotiq" or robot_name == "Alice":
        return {
            "x": (-1.3, 1.6),
            "y": (-0.4, 1.5),
            "z": (0.0, 1.0),
        }

    if robot_name == "panda" or robot_name == "Bob":
        return {
            "x": (-1.3, 1.6),
            "y": (0.0, 1.5),
            "z": (0.0, 1.0),
        }

    raise NotImplementedError(robot_name)
```

---

# Part E — Implement reset/randomization

`MujocoSimEnv.reset()` calls `sample_initial_scene()` when randomization is enabled. Use it to place blocks in random table positions.

```python
def sample_initial_scene(self):
    base_positions = {
        "red_block": np.array([0.35, 0.25, 0.05]),
        "blue_block": np.array([0.35, -0.25, 0.05]),
    }

    for obj_name, base_pos in base_positions.items():
        noise = self.random_state.uniform(
            low=np.array([-0.05, -0.05, 0.0]),
            high=np.array([0.05, 0.05, 0.0]),
        )

        new_pos = base_pos + noise
        new_quat = np.array([1, 0, 0, 0])

        self.reset_body_pose(
            body_name=obj_name,
            pos=new_pos,
            quat=new_quat,
        )
```

If `reset_body_pose()` is not available in your exact repo state, copy the object-reset style from existing tasks.

---

# Part F — Implement observation descriptions for the LLM

## 1. Describe one object

```python
def describe_object(self, obs: EnvState, name: str) -> str:
    obj_state = obs.objects[name]
    x, y, z = obj_state.xpos

    target_name = BLOCK_BIN_TARGETS[name]
    target_pos = self.target_xposes[target_name]
    dist = np.linalg.norm(obj_state.xpos[:2] - target_pos[:2])

    status = "at target" if dist < self.success_threshold else "not at target"

    return (
        f"{name}: position=({x:.2f}, {y:.2f}, {z:.2f}), "
        f"target={target_name}, status={status}"
    )
```

## 2. Describe robot state

```python
def describe_robot_state(self, obs: EnvState, robot_name: str) -> str:
    robot_state = getattr(obs, robot_name)
    x, y, z = robot_state.ee_xpos

    contacts = [c for c in robot_state.contacts if c in self.object_names]
    holding = contacts[0] if len(contacts) > 0 else "nothing"

    agent_name = self.robot_name_map[robot_name]

    return (
        f"{agent_name}'s gripper is at ({x:.2f}, {y:.2f}, {z:.2f}) "
        f"and is holding/contacting {holding}."
    )
```

## 3. Describe full observation

```python
def describe_obs(self, obs: EnvState):
    description = "[Scene]\n"

    description += "\n[Objects]\n"
    for obj_name in self.object_names:
        description += self.describe_object(obs, obj_name) + "\n"

    description += "\n[Targets]\n"
    for target_name, xpos in self.target_xposes.items():
        x, y, z = xpos
        description += f"{target_name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"

    description += "\n[Robots]\n"
    for robot_name in self.robot_names:
        description += self.describe_robot_state(obs, robot_name) + "\n"

    return description
```

---

# Part G — Implement prompts

```python
def describe_task_context(self):
    return BLOCK_BIN_TASK_CONTEXT

def get_action_prompt(self) -> str:
    return BLOCK_BIN_ACTION_SPACE

def get_agent_prompt(self, obs: EnvState, agent_name: str):
    robot_name = self.get_robot_name(agent_name)
    other_agent = "Alice" if agent_name == "Bob" else "Bob"

    return f"""
You are {agent_name}, controlling robot {robot_name}.

{BLOCK_BIN_TASK_CONTEXT}

Current scene:
{self.describe_obs(obs)}

You should coordinate with {other_agent}.
You can pick and place blocks, but avoid selecting the same block as the other robot.
Use only the allowed action format.
"""
```

---

# Part H — Implement task feedback

Task feedback catches task-level invalid actions.

```python
def get_task_feedback(self, llm_plan, pose_dict):
    feedback = ""

    action_str = llm_plan.action_strs.get(llm_plan.agent_name, "")

    for obj_name, correct_target in BLOCK_BIN_TARGETS.items():
        if obj_name in action_str and "PLACE" in action_str:
            if correct_target not in action_str:
                feedback += (
                    f"{obj_name} should be placed in {correct_target}, "
                    f"not another target.\n"
                )

    if feedback:
        return False, feedback

    return True, ""
```

You can start simple. Later, make this stricter by inspecting parsed plan fields instead of raw strings.

---

# Part I — Implement allowed objects and collisions

## 1. Graspable objects

```python
def get_graspable_objects(self):
    return {
        "Alice": self.object_names.copy(),
        "Bob": self.object_names.copy(),
    }
```

## 2. Allowed collisions

Start conservative and copy/adapt the pattern from an existing task.

```python
def get_allowed_collision_pairs(self):
    allowed_pairs = []

    for obj in self.object_names:
        allowed_pairs.append(("ur5e_robotiq", obj))
        allowed_pairs.append(("panda", obj))

    for obj in self.object_names:
        allowed_pairs.append((obj, "table"))

    return allowed_pairs
```

The exact collision names must match your XML geoms/bodies. Inspect contacts during a debug run if unsure.

---

# Part J — Implement success condition

```python
def get_reward_done(self, obs: EnvState):
    all_done = True

    for obj_name, target_name in BLOCK_BIN_TARGETS.items():
        obj_pos = obs.objects[obj_name].xpos
        target_pos = self.target_xposes[target_name]

        dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])

        if dist > self.success_threshold:
            all_done = False
            break

    reward = 1 if all_done else 0
    done = all_done

    return reward, done
```

For stronger evaluation, also check object height, stability, release, and target-region containment.

---

# Part K — Export and register the task

Edit:

```text
rocobench/envs/__init__.py
```

Add:

```python
from .task_block_bin import BlockBinTask
```

Edit:

```text
run_dialog.py
```

Add the task to imports if needed:

```python
from rocobench.envs import BlockBinTask
```

Then add it to `TASK_NAME_MAP`:

```python
TASK_NAME_MAP = {
    "sort": SortOneBlockTask,
    "cabinet": CabinetTask,
    "rope": MoveRopeTask,
    "sweep": SweepTask,
    "sandwich": MakeSandwichTask,
    "pack": PackGroceryTask,
    "block_bin": BlockBinTask,
}
```

Now run:

```bash
python run_dialog.py --task block_bin
```

---

# Part L — Test without the LLM first

Create:

```text
scripts/test_block_bin_env.py
```

```python
from rocobench.envs import BlockBinTask

def main():
    env = BlockBinTask(
        np_seed=0,
        render_freq=1,
        render_size=256,
    )

    obs = env.reset()
    print(env.describe_task_context())
    print(env.describe_obs(obs))
    print(env.get_action_prompt())

    reward, done = env.get_reward_done(obs)
    print("Initial reward:", reward)
    print("Initial done:", done)

if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/test_block_bin_env.py
```

This verifies XML loading, robots, objects, target sites, observations, prompts, and success checks.

---

# Part M — Test target and grasp helpers

```python
from rocobench.envs import BlockBinTask

env = BlockBinTask(np_seed=0)
obs = env.reset()

for obj in ["red_block", "blue_block"]:
    print(obj, "joint:", env.get_object_joint_name(obj))
    print(obj, "grasp site:", env.get_grasp_site(obj))
    print(obj, "grasp pos:", env.get_target_pos("Alice", obj))

for target in ["red_bin", "blue_bin"]:
    print(target, "target pos:", env.get_target_pos("Alice", target))
```

Common errors:
- site not found;
- joint not found;
- target position is `None`;
- object missing from `obs.objects`.

These usually mean the XML name and Python name do not match.

---

# Part N — Test with the planner

After direct environment loading works:

```bash
python run_dialog.py   --task block_bin   --output_mode action_only   --comm_mode dialog   --num_runs 1   --max_runner_steps 5
```

Expected plan:

```text
EXECUTE
NAME Alice ACTION PICK red_block PLACE red_bin
NAME Bob ACTION PICK blue_block PLACE blue_bin
```

If parsing fails, check:
- exact object names;
- exact target names;
- `get_action_prompt()`;
- `get_agent_prompt()`;
- use of `PICK`, `PLACE`, `WAIT`.

---

# Part O — Minimal complete `task_block_bin.py`

Below is a compact reference. You still need a valid XML scene.

```python
from typing import Dict, Tuple, Optional
import numpy as np

from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import (
    UR5E_ROBOTIQ_CONSTANTS,
    PANDA_CONSTANTS,
)


BLOCK_BIN_OBJECTS = [
    "red_block",
    "blue_block",
]

BLOCK_BIN_TARGETS = {
    "red_block": "red_bin",
    "blue_block": "blue_bin",
}

BLOCK_BIN_TASK_CONTEXT = """
Two robots, Alice and Bob, must place colored blocks into their matching bins.
The red block should go to the red bin.
The blue block should go to the blue bin.
Coordinate so both robots do not manipulate the same block.
"""

BLOCK_BIN_ACTION_SPACE = """
[Action Options]

1) PICK <object>
2) PLACE <object> <target>
3) PICK <object> PLACE <target>
4) WAIT

[Valid Objects]
red_block
blue_block

[Valid Targets]
red_bin
blue_bin

[Action Output Instruction]
After discussion, output exactly one ACTION per robot:

EXECUTE
NAME Alice ACTION <action>
NAME Bob ACTION <action>

Example:
EXECUTE
NAME Alice ACTION PICK red_block PLACE red_bin
NAME Bob ACTION PICK blue_block PLACE blue_bin
"""


class BlockBinTask(MujocoSimEnv):
    def __init__(
        self,
        filepath: str = "rocobench/envs/task_block_bin.xml",
        **kwargs,
    ):
        self.robot_names = ["ur5e_robotiq", "panda"]

        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
        }

        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda",
        }

        self.robots = {}

        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()
        panda_config = PANDA_CONSTANTS.copy()

        self.object_names = BLOCK_BIN_OBJECTS
        self.target_names = list(BLOCK_BIN_TARGETS.values())

        super(BlockBinTask, self).__init__(
            filepath=filepath,
            task_objects=BLOCK_BIN_OBJECTS,
            agent_configs={
                "ur5e_robotiq": robotiq_config,
                "panda": panda_config,
            },
            **kwargs,
        )

        self.target_xposes = {}
        for target in self.target_names:
            self.target_xposes[target] = self.physics.data.site(target).xpos.copy()

        self.robots[self.robot_name_map["ur5e_robotiq"]] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )

        self.robots[self.robot_name_map["panda"]] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )

        self.success_threshold = 0.08

    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]:
        if target_name in self.object_names:
            site_name = f"{target_name}_top"
        elif target_name in self.target_names:
            site_name = target_name
        else:
            return None

        try:
            return self.physics.data.site(site_name).xpos.copy()
        except KeyError:
            print(f"KeyError: site {site_name} not found")
            return None

    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        return np.array([1, 0, 0, 0])

    def get_grasp_site(self, obj_name: str) -> Optional[str]:
        if obj_name in self.object_names:
            return f"{obj_name}_top"
        return None

    def get_object_joint_name(self, obj_name: str) -> str:
        return f"{obj_name}_joint"

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]

    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == "Alice":
            return {
                "x": (-1.3, 1.6),
                "y": (-0.4, 1.5),
                "z": (0.0, 1.0),
            }

        if robot_name == "panda" or robot_name == "Bob":
            return {
                "x": (-1.3, 1.6),
                "y": (0.0, 1.5),
                "z": (0.0, 1.0),
            }

        raise NotImplementedError(robot_name)

    def get_graspable_objects(self):
        return {
            "Alice": self.object_names.copy(),
            "Bob": self.object_names.copy(),
        }

    def describe_object(self, obs: EnvState, name: str) -> str:
        obj_state = obs.objects[name]
        x, y, z = obj_state.xpos

        target_name = BLOCK_BIN_TARGETS[name]
        target_pos = self.target_xposes[target_name]
        dist = np.linalg.norm(obj_state.xpos[:2] - target_pos[:2])
        status = "at target" if dist < self.success_threshold else "not at target"

        return (
            f"{name}: position=({x:.2f}, {y:.2f}, {z:.2f}), "
            f"target={target_name}, status={status}"
        )

    def describe_robot_state(self, obs: EnvState, robot_name: str) -> str:
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos

        contacts = [c for c in robot_state.contacts if c in self.object_names]
        holding = contacts[0] if len(contacts) > 0 else "nothing"

        agent_name = self.robot_name_map[robot_name]

        return (
            f"{agent_name}'s gripper is at ({x:.2f}, {y:.2f}, {z:.2f}) "
            f"and is holding/contacting {holding}."
        )

    def describe_obs(self, obs: EnvState):
        description = "[Scene]\n"

        description += "\n[Objects]\n"
        for obj_name in self.object_names:
            description += self.describe_object(obs, obj_name) + "\n"

        description += "\n[Targets]\n"
        for target_name, xpos in self.target_xposes.items():
            x, y, z = xpos
            description += f"{target_name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"

        description += "\n[Robots]\n"
        for robot_name in self.robot_names:
            description += self.describe_robot_state(obs, robot_name) + "\n"

        return description

    def describe_task_context(self):
        return BLOCK_BIN_TASK_CONTEXT

    def get_action_prompt(self) -> str:
        return BLOCK_BIN_ACTION_SPACE

    def get_agent_prompt(self, obs: EnvState, agent_name: str):
        robot_name = self.get_robot_name(agent_name)
        other_agent = "Alice" if agent_name == "Bob" else "Bob"

        return f"""
You are {agent_name}, controlling robot {robot_name}.

{BLOCK_BIN_TASK_CONTEXT}

Current scene:
{self.describe_obs(obs)}

You should coordinate with {other_agent}.
Use only the allowed action format.
"""

    def get_task_feedback(self, llm_plan, pose_dict):
        feedback = ""
        action_str = llm_plan.action_strs.get(llm_plan.agent_name, "")

        for obj_name, correct_target in BLOCK_BIN_TARGETS.items():
            if obj_name in action_str and "PLACE" in action_str:
                if correct_target not in action_str:
                    feedback += (
                        f"{obj_name} should be placed in {correct_target}, "
                        f"not another target.\n"
                    )

        if feedback:
            return False, feedback

        return True, ""

    def get_reward_done(self, obs: EnvState):
        all_done = True

        for obj_name, target_name in BLOCK_BIN_TARGETS.items():
            obj_pos = obs.objects[obj_name].xpos
            target_pos = self.target_xposes[target_name]
            dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])

            if dist > self.success_threshold:
                all_done = False
                break

        reward = 1 if all_done else 0
        done = all_done

        return reward, done
```

---

# Common mistakes

1. **XML and Python names do not match.**  
   If Python expects `red_block_top` but XML defines `redblock_top`, target/grasp lookup fails.

2. **Missing free joint.**  
   Movable objects need a free joint such as `red_block_joint`.

3. **Missing grasp site.**  
   If `get_grasp_site()` returns `red_block_top`, the XML must define that site.

4. **Bad target names in prompt.**  
   Use exact symbolic names like `red_bin`, not natural phrases like `red bin`.

5. **Weak success predicate.**  
   Distance-only success may be too permissive. Add release/stability checks for serious experiments.

6. **Missing registration.**  
   If the task is not in `TASK_NAME_MAP`, `--task block_bin` will fail.

---

# Final checklist

```text
[ ] Create task_<name>.xml
[ ] Add robots, objects, free joints, sites, cameras, targets
[ ] Create task_<name>.py
[ ] Subclass MujocoSimEnv
[ ] Define robot_name_map and robot_name_map_inv
[ ] Pass XML, task_objects, and agent_configs to super().__init__
[ ] Create SimRobot objects
[ ] Implement sample_initial_scene
[ ] Implement get_target_pos
[ ] Implement get_target_quat
[ ] Implement get_grasp_site
[ ] Implement get_object_joint_name
[ ] Implement get_robot_reach_range
[ ] Implement get_graspable_objects
[ ] Implement describe_task_context
[ ] Implement describe_obs
[ ] Implement get_action_prompt
[ ] Implement get_agent_prompt
[ ] Implement get_task_feedback
[ ] Implement get_reward_done
[ ] Export task in rocobench/envs/__init__.py
[ ] Add task to TASK_NAME_MAP in run_dialog.py
[ ] Test XML/env loading without LLM
[ ] Test prompts
[ ] Test target/grasp helpers
[ ] Test reward/done
[ ] Run planner with --task <name>
```

# Recommended development order

```text
1. Copy an existing task XML and Python file.
2. Rename class, object names, and target names.
3. Make reset and observation work.
4. Make target/grasp helpers work.
5. Make reward/done work.
6. Make action prompt parse correctly.
7. Run one RRT action.
8. Then run full LLM dialogue.
```

Do not start by tuning prompts. First make sure the MuJoCo task, names, sites, and success predicate work.
