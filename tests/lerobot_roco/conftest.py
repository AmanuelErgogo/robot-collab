import os
import sys

import numpy as np


CLIENT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "integrations", "lerobot_roco", "client", "src")
)
if CLIENT_SRC not in sys.path:
    sys.path.insert(0, CLIENT_SRC)


class FakeRobot:
    def __init__(self, joint_ctrl, joint_qpos, grasp_idx):
        self.joint_idxs_in_ctrl = list(joint_ctrl)
        self.joint_idxs_in_qpos = list(joint_qpos)
        self.grasp_idx = int(grasp_idx)


class FakeModel:
    def __init__(self):
        self.actuator_ctrlrange = np.asarray(
            [
                [-1.0, 1.0],
                [-2.0, 2.0],
                [0.0, 255.0],
                [-3.0, 3.0],
                [-4.0, 4.0],
                [0.0, 255.0],
            ],
            dtype=np.float32,
        )
        self.nu = 6
        self.neq = 0
        self.eq_active = np.zeros((0,), dtype=np.int32)


class FakeData:
    def __init__(self):
        self.qpos = np.asarray([0.1, 0.2, -0.3, -0.4], dtype=np.float64)
        self.qvel = np.asarray([0.01, 0.02, -0.03, -0.04], dtype=np.float64)
        self.ctrl = np.asarray([0.1, 0.2, 30.0, -0.3, -0.4, 40.0], dtype=np.float64)


class FakePhysics:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()
        self.named = type(
            "Named",
            (),
            {
                "data": type(
                    "NamedData",
                    (),
                    {
                        "qvel": type(
                            "NamedQvel",
                            (),
                            {
                                "_convert_key": staticmethod(
                                    lambda key: {
                                        "alice_j0": slice(2, 3),
                                        "alice_j1": slice(3, 4),
                                        "bob_j0": slice(0, 1),
                                        "bob_j1": slice(1, 2),
                                    }[key]
                                )
                            },
                        )()
                    },
                )()
            },
        )()

    def timestep(self):
        return 0.002

    def render(self, camera_id, height, width):
        base = 10 if camera_id == "teaser" else 20
        return np.full((height, width, 3), base, dtype=np.uint8)


class FakeEnv:
    def __init__(self):
        self.physics = FakePhysics()
        self.robots = {
            "Alice": FakeRobot([0, 1], [0, 1], 2),
            "Bob": FakeRobot([3, 4], [2, 3], 5),
        }
        self.robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}
        self.agent_configs = {
            "ur5e_robotiq": {
                "ik_joint_names": ["alice_j0", "alice_j1"],
                "grasp_actuator": "alice_gripper",
            },
            "panda": {
                "ik_joint_names": ["bob_j0", "bob_j1"],
                "grasp_actuator": "bob_gripper",
            },
        }
        self.render_cameras = ["teaser", "face_ur5e"]
        self.sim_forward_steps = 100
        self.render_freq = 20
        self.sim_save_freq = 100
        self.randomize_init = True
        self.timestep = 0
        self.last_action = None
        self.seed_value = None

    def seed(self, np_seed):
        self.seed_value = np_seed

    def reset(self, reload=False):
        self.timestep = 0
        self.physics.data = FakeData()
        return object()

    def step(self, action, verbose=False):
        self.last_action = action
        self.physics.data.ctrl[np.asarray(action.ctrl_idxs, dtype=np.int32)] = np.asarray(action.ctrl_vals)
        self.timestep += 1
        return object(), 0.0, False, {}

    def get_reward_done(self, obs):
        return 0.0, False
