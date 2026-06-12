"""Observation formatting for LeRobot-compatible raw Gym observations."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoObservationError


@dataclass(frozen=True)
class ObservationLayout:
    active_agent: str
    active_robot_name: str
    joint_names: Tuple[str, ...]
    joint_qpos_indices: Tuple[int, ...]
    joint_qvel_indices: Tuple[int, ...]
    gripper_ctrl_index: int
    gripper_name: str
    state_field_names: Tuple[str, ...]
    camera_aliases: Mapping[str, str]
    state_dim: int


def build_observation_layout(env: Any, active_agent: str, camera_aliases: Mapping[str, str]) -> ObservationLayout:
    robot = env.robots[active_agent]
    if hasattr(env, "robot_name_map_inv"):
        robot_name = str(env.robot_name_map_inv[active_agent])
    else:
        robot_name = str(getattr(robot, "name", active_agent))
    configs = getattr(env, "agent_configs", {})
    if robot_name in configs:
        joint_names = tuple(str(x) for x in configs[robot_name]["ik_joint_names"])
        gripper_name = str(configs[robot_name]["grasp_actuator"])
    else:
        joint_names = tuple("joint_%d" % i for i in range(len(robot.joint_idxs_in_qpos)))
        gripper_name = str(getattr(robot, "grasp_actuator", "gripper"))
    qpos_fields = tuple("qpos.%s" % name for name in joint_names)
    qvel_fields = tuple("qvel.%s" % name for name in joint_names)
    qvel_indices = []
    for fallback_idx, joint_name in zip(robot.joint_idxs_in_qpos, joint_names):
        try:
            qvel_slice = env.physics.named.data.qvel._convert_key(joint_name)
            if int(qvel_slice.stop - qvel_slice.start) != 1:
                raise ValueError("Only single-DoF joints are supported.")
            qvel_indices.append(int(qvel_slice.start))
        except Exception:
            qvel_indices.append(int(fallback_idx))
    state_field_names = qpos_fields + qvel_fields + ("ctrl.%s" % gripper_name,)
    return ObservationLayout(
        active_agent=active_agent,
        active_robot_name=robot_name,
        joint_names=joint_names,
        joint_qpos_indices=tuple(int(x) for x in robot.joint_idxs_in_qpos),
        joint_qvel_indices=tuple(qvel_indices),
        gripper_ctrl_index=int(robot.grasp_idx),
        gripper_name=gripper_name,
        state_field_names=state_field_names,
        camera_aliases=dict(camera_aliases),
        state_dim=len(state_field_names),
    )


class RoCoObservationAdapter:
    def __init__(self, env: Any, active_agent: str, camera_aliases: Mapping[str, str], image_height: int, image_width: int) -> None:
        self.env = env
        self.layout = build_observation_layout(env, active_agent, camera_aliases)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def _agent_pos(self) -> np.ndarray:
        qpos = np.asarray(self.env.physics.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.env.physics.data.qvel, dtype=np.float32)
        ctrl = np.asarray(self.env.physics.data.ctrl, dtype=np.float32)
        qpos_values = qpos[list(self.layout.joint_qpos_indices)].astype(np.float32)
        qvel_values = qvel[list(self.layout.joint_qvel_indices)].astype(np.float32)
        gripper = np.asarray([ctrl[self.layout.gripper_ctrl_index]], dtype=np.float32)
        state = np.concatenate([qpos_values, qvel_values, gripper]).astype(np.float32)
        if state.shape != (self.layout.state_dim,):
            raise RoCoObservationError(
                "Agent state shape mismatch.",
                code=ErrorCode.OBSERVATION_SHAPE_MISMATCH,
                details={"expected": [self.layout.state_dim], "received": list(state.shape)},
            )
        return np.ascontiguousarray(state, dtype=np.float32)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        try:
            image = self.env.physics.render(
                camera_id=camera_name,
                height=self.image_height,
                width=self.image_width,
            )
        except Exception as exc:
            raise RoCoObservationError(
                "Failed to render camera.",
                code=ErrorCode.RENDER_FAILED,
                details={"camera": camera_name},
            ) from exc
        arr = np.asarray(image)
        expected = (self.image_height, self.image_width, 3)
        if arr.shape != expected:
            raise RoCoObservationError(
                "Rendered image shape mismatch.",
                code=ErrorCode.OBSERVATION_SHAPE_MISMATCH,
                details={"camera": camera_name, "expected": list(expected), "received": list(arr.shape)},
            )
        if arr.dtype != np.uint8:
            raise RoCoObservationError(
                "Rendered image dtype mismatch.",
                code=ErrorCode.OBSERVATION_SHAPE_MISMATCH,
                details={"camera": camera_name, "expected": "uint8", "received": str(arr.dtype)},
            )
        return np.ascontiguousarray(arr, dtype=np.uint8)

    def format(self, obs: Any) -> Dict[str, Any]:
        pixels: Dict[str, np.ndarray] = {}
        for alias in sorted(self.layout.camera_aliases.keys()):
            camera_name = self.layout.camera_aliases[alias]
            pixels[alias] = self._render_camera(camera_name)
        return {
            "pixels": pixels,
            "agent_pos": self._agent_pos(),
        }

    def render(self) -> np.ndarray:
        camera_name = self.layout.camera_aliases.get("front")
        if camera_name is None:
            camera_name = next(iter(self.layout.camera_aliases.values()))
        return self._render_camera(camera_name)
