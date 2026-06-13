from types import SimpleNamespace

import numpy as np

from integrations.lerobot_roco.dataset.recorder import RocoTransitionObserver, public_action_from_sim_action
from integrations.lerobot_roco.dataset.schema import default_schema_for_tests


class FakeRobot:
    joint_idxs_in_ctrl = [0]
    joint_idxs_in_qpos = [0]
    grasp_idx = 1


class FakeModel:
    actuator_ctrlrange = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)


class FakeData:
    def __init__(self):
        self.qpos = np.asarray([0.0], dtype=np.float64)
        self.qvel = np.asarray([0.0], dtype=np.float64)
        self.ctrl = np.asarray([0.0, 0.0], dtype=np.float64)
        self.time = 0.0


class FakePhysics:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()
        self.named = SimpleNamespace(
            data=SimpleNamespace(
                qvel=SimpleNamespace(_convert_key=lambda key: slice(0, 1))
            )
        )

    def render(self, camera_id, height, width):
        del camera_id
        value = int(round(float(self.data.qpos[0]) * 10.0))
        return np.full((height, width, 3), value, dtype=np.uint8)


class FakeEnv:
    def __init__(self):
        self.physics = FakePhysics()
        self.robots = {"Alice": FakeRobot()}
        self.robot_name_map_inv = {"Alice": "fake_robot"}
        self.agent_configs = {
            "fake_robot": {
                "ik_joint_names": ["joint0"],
                "grasp_actuator": "gripper",
            }
        }
        self.timestep = 0

    def step(self, action, verbose=False):
        del verbose
        self.physics.data.ctrl[np.asarray(action.ctrl_idxs, dtype=np.int64)] = action.ctrl_vals
        self.physics.data.qpos[0] = action.ctrl_vals[0]
        self.physics.data.time += 0.1
        self.timestep += 1
        return object(), 0.0, False, {}


def test_observer_records_pre_action_observation_and_applied_action():
    env = FakeEnv()
    schema = default_schema_for_tests(state_dim=3, action_dim=2, image_shape=(4, 5, 3), fps=10)
    observer = RocoTransitionObserver(
        env,
        schema,
        active_agent="Alice",
        episode_index=0,
        camera_aliases={"front": "teaser", "active_agent": "face"},
        image_height=4,
        image_width=5,
    )
    sim_action = SimpleNamespace(
        ctrl_idxs=np.asarray([0, 1], dtype=np.int32),
        ctrl_vals=np.asarray([0.75, 0.25], dtype=np.float32),
    )

    observer.before_step("obs0", sim_action, {"plan_index": 0})
    env.step(sim_action)

    frame = observer.frames[0]
    features = frame.to_feature_dict()
    assert features["observation.state"][0] == 0.0
    assert features["action"].tolist() == [0.75, 0.25]
    assert features["timestamp"] == 0.0
    assert env.physics.data.qpos[0] == 0.75


def test_public_action_fills_sparse_gripper_from_current_control():
    env = FakeEnv()
    env.physics.data.ctrl[1] = 0.5
    observer = RocoTransitionObserver(
        env,
        default_schema_for_tests(state_dim=3, action_dim=2, image_shape=(4, 5, 3), fps=10),
        active_agent="Alice",
        episode_index=0,
        camera_aliases={"front": "teaser", "active_agent": "face"},
        image_height=4,
        image_width=5,
    )
    sim_action = SimpleNamespace(
        ctrl_idxs=np.asarray([0], dtype=np.int32),
        ctrl_vals=np.asarray([0.25], dtype=np.float32),
        qpos_idxs=np.asarray([0], dtype=np.int32),
        qpos_target=np.asarray([0.25], dtype=np.float32),
    )

    action = public_action_from_sim_action(sim_action, observer.action_layout, env=env)

    assert action.tolist() == [0.25, 0.5]
