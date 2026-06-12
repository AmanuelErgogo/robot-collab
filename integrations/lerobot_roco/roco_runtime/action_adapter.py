"""Action layout and conversion to RoCo ``SimAction`` semantics."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoActionError


@dataclass(frozen=True)
class ActionLayout:
    active_agent: str
    active_robot_name: str
    joint_names: Tuple[str, ...]
    joint_ctrl_indices: Tuple[int, ...]
    joint_qpos_indices: Tuple[int, ...]
    gripper_ctrl_index: int
    gripper_name: str
    field_names: Tuple[str, ...]
    low: np.ndarray
    high: np.ndarray

    @property
    def action_dim(self) -> int:
        return len(self.field_names)


def _to_int_tuple(values: Any) -> Tuple[int, ...]:
    return tuple(int(v) for v in list(values))


def _robot_name_for_agent(env: Any, agent_name: str) -> str:
    if hasattr(env, "robot_name_map_inv"):
        return str(env.robot_name_map_inv[agent_name])
    robot = env.robots[agent_name]
    return str(getattr(robot, "name", agent_name))


def _joint_names_for_robot(env: Any, robot_name: str, robot: Any) -> Tuple[str, ...]:
    configs = getattr(env, "agent_configs", {})
    if robot_name in configs:
        return tuple(str(x) for x in configs[robot_name]["ik_joint_names"])
    if hasattr(robot, "ik_joint_names"):
        return tuple(str(x) for x in robot.ik_joint_names)
    return tuple("joint_%d" % i for i in range(len(robot.joint_idxs_in_qpos)))


def _gripper_name_for_robot(env: Any, robot_name: str, robot: Any) -> str:
    configs = getattr(env, "agent_configs", {})
    if robot_name in configs:
        return str(configs[robot_name]["grasp_actuator"])
    return str(getattr(robot, "grasp_actuator", "gripper"))


def build_action_layout(env: Any, active_agent: str) -> ActionLayout:
    if active_agent not in env.robots:
        raise RoCoActionError(
            "Unknown active agent.",
            code=ErrorCode.UNKNOWN_AGENT,
            details={"active_agent": active_agent, "known_agents": sorted(env.robots.keys())},
        )
    robot = env.robots[active_agent]
    robot_name = _robot_name_for_agent(env, active_agent)
    joint_names = _joint_names_for_robot(env, robot_name, robot)
    joint_ctrl_indices = _to_int_tuple(robot.joint_idxs_in_ctrl)
    joint_qpos_indices = _to_int_tuple(robot.joint_idxs_in_qpos)
    gripper_ctrl_index = int(robot.grasp_idx)
    gripper_name = _gripper_name_for_robot(env, robot_name, robot)

    if len(joint_ctrl_indices) != len(joint_qpos_indices):
        raise RoCoActionError(
            "Joint control and qpos index counts differ.",
            code=ErrorCode.INVALID_CONTROL_INDEX,
            details={
                "joint_ctrl_indices": joint_ctrl_indices,
                "joint_qpos_indices": joint_qpos_indices,
            },
        )
    if len(joint_names) != len(joint_ctrl_indices):
        raise RoCoActionError(
            "Joint names and control index counts differ.",
            code=ErrorCode.INVALID_CONTROL_INDEX,
            details={"joint_names": joint_names, "joint_ctrl_indices": joint_ctrl_indices},
        )

    ctrl_indices = list(joint_ctrl_indices) + [gripper_ctrl_index]
    ctrlrange = np.asarray(env.physics.model.actuator_ctrlrange, dtype=np.float32)[ctrl_indices]
    low = np.ascontiguousarray(ctrlrange[:, 0], dtype=np.float32)
    high = np.ascontiguousarray(ctrlrange[:, 1], dtype=np.float32)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise RoCoActionError("Action bounds must be finite.", code=ErrorCode.INVALID_CONTROL_INDEX)
    field_names = tuple(list(joint_names) + [gripper_name])

    return ActionLayout(
        active_agent=active_agent,
        active_robot_name=robot_name,
        joint_names=joint_names,
        joint_ctrl_indices=joint_ctrl_indices,
        joint_qpos_indices=joint_qpos_indices,
        gripper_ctrl_index=gripper_ctrl_index,
        gripper_name=gripper_name,
        field_names=field_names,
        low=low,
        high=high,
    )


class RoCoActionAdapter:
    def __init__(self, env: Any, active_agent: str, out_of_bounds: str = "reject") -> None:
        if out_of_bounds not in {"reject", "clip"}:
            raise ValueError("out_of_bounds must be 'reject' or 'clip'.")
        self.env = env
        self.active_agent = active_agent
        self.out_of_bounds = out_of_bounds
        self.layout = build_action_layout(env, active_agent)
        self.passive_agents = tuple(agent for agent in sorted(env.robots.keys()) if agent != active_agent)
        self._passive_joint_holds: Dict[str, np.ndarray] = {}
        self._passive_gripper_holds: Dict[str, np.float32] = {}
        self.refresh_holds()

    def refresh_holds(self) -> None:
        for agent in self.passive_agents:
            robot = self.env.robots[agent]
            qpos = np.asarray(self.env.physics.data.qpos, dtype=np.float32)
            ctrl = np.asarray(self.env.physics.data.ctrl, dtype=np.float32)
            self._passive_joint_holds[agent] = np.ascontiguousarray(
                qpos[list(robot.joint_idxs_in_qpos)], dtype=np.float32
            )
            self._passive_gripper_holds[agent] = np.float32(ctrl[int(robot.grasp_idx)])

    def current_active_hold_action(self) -> np.ndarray:
        robot = self.env.robots[self.active_agent]
        qpos = np.asarray(self.env.physics.data.qpos, dtype=np.float32)
        ctrl = np.asarray(self.env.physics.data.ctrl, dtype=np.float32)
        values = list(qpos[list(robot.joint_idxs_in_qpos)].astype(np.float32))
        values.append(np.float32(ctrl[int(robot.grasp_idx)]))
        action = np.ascontiguousarray(values, dtype=np.float32)
        return np.clip(action, self.layout.low, self.layout.high).astype(np.float32)

    def validate_action(self, action: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(action)
        if arr.dtype.hasobject:
            raise RoCoActionError("Object dtype action is not allowed.", code=ErrorCode.INVALID_ACTION_DTYPE)
        if tuple(arr.shape) != (self.layout.action_dim,):
            raise RoCoActionError(
                "Action shape mismatch.",
                code=ErrorCode.INVALID_ACTION_SHAPE,
                details={"expected": [self.layout.action_dim], "received": list(arr.shape)},
            )
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RoCoActionError("Action contains NaN or Inf.", code=ErrorCode.NONFINITE_ACTION)
        below = arr < self.layout.low
        above = arr > self.layout.high
        clipped = bool(np.any(below) or np.any(above))
        info = {"action_clipped": False}
        if clipped:
            details = {
                "low": self.layout.low.tolist(),
                "high": self.layout.high.tolist(),
                "received": arr.tolist(),
            }
            if self.out_of_bounds == "reject":
                raise RoCoActionError("Action is outside control bounds.", code=ErrorCode.ACTION_OUT_OF_BOUNDS, details=details)
            arr = np.clip(arr, self.layout.low, self.layout.high).astype(np.float32)
            info["action_clipped"] = True
        return arr, info

    def build_sim_action_kwargs(self, action: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        arr, info = self.validate_action(action)
        active_joint_values = arr[: len(self.layout.joint_ctrl_indices)]
        active_gripper_value = arr[-1:]

        ctrl_idxs: List[int] = list(self.layout.joint_ctrl_indices) + [self.layout.gripper_ctrl_index]
        ctrl_vals: List[np.float32] = [np.float32(x) for x in active_joint_values]
        ctrl_vals.append(np.float32(active_gripper_value[0]))
        qpos_idxs: List[int] = list(self.layout.joint_qpos_indices)
        qpos_target: List[np.float32] = [np.float32(x) for x in active_joint_values]

        for agent in self.passive_agents:
            robot = self.env.robots[agent]
            passive_joint_ctrl_indices = [int(x) for x in robot.joint_idxs_in_ctrl]
            passive_joint_qpos_indices = [int(x) for x in robot.joint_idxs_in_qpos]
            passive_joint_values = self._passive_joint_holds[agent]
            if len(passive_joint_ctrl_indices) != len(passive_joint_values):
                raise RoCoActionError("Passive hold shape mismatch.", code=ErrorCode.INVALID_CONTROL_INDEX)
            ctrl_idxs.extend(passive_joint_ctrl_indices)
            ctrl_vals.extend(np.float32(x) for x in passive_joint_values)
            ctrl_idxs.append(int(robot.grasp_idx))
            ctrl_vals.append(np.float32(self._passive_gripper_holds[agent]))
            qpos_idxs.extend(passive_joint_qpos_indices)
            qpos_target.extend(np.float32(x) for x in passive_joint_values)

        kwargs = {
            "ctrl_idxs": np.ascontiguousarray(ctrl_idxs, dtype=np.int32),
            "ctrl_vals": np.ascontiguousarray(ctrl_vals, dtype=np.float32),
            "qpos_idxs": np.ascontiguousarray(qpos_idxs, dtype=np.int32),
            "qpos_target": np.ascontiguousarray(qpos_target, dtype=np.float32),
            "eq_active_idxs": None,
            "eq_active_vals": None,
        }
        return kwargs, info

    def to_sim_action(self, action: Any) -> Tuple[Any, Dict[str, Any]]:
        kwargs, info = self.build_sim_action_kwargs(action)
        from rocobench.envs.base_env import SimAction

        return SimAction(**kwargs), info


def encode_sim_action(sim_action: Any) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {
        "ctrl_idxs": np.ascontiguousarray(sim_action.ctrl_idxs, dtype=np.int32),
        "ctrl_vals": np.ascontiguousarray(sim_action.ctrl_vals, dtype=np.float32),
        "qpos_idxs": np.ascontiguousarray(sim_action.qpos_idxs, dtype=np.int32),
        "qpos_target": np.ascontiguousarray(sim_action.qpos_target, dtype=np.float32),
    }
    if getattr(sim_action, "eq_active_idxs", None) is not None:
        payload["eq_active_idxs"] = np.ascontiguousarray(sim_action.eq_active_idxs, dtype=np.int32)
    else:
        payload["eq_active_idxs"] = np.ascontiguousarray([], dtype=np.int32)
    if getattr(sim_action, "eq_active_vals", None) is not None:
        payload["eq_active_vals"] = np.ascontiguousarray(sim_action.eq_active_vals, dtype=np.int32)
    else:
        payload["eq_active_vals"] = np.ascontiguousarray([], dtype=np.int32)
    return payload


def decode_sim_action(payload: Mapping[str, Any], model_nu: Optional[int] = None, model_neq: Optional[int] = None) -> Any:
    ctrl_idxs = np.ascontiguousarray(payload["ctrl_idxs"], dtype=np.int32)
    ctrl_vals = np.ascontiguousarray(payload["ctrl_vals"], dtype=np.float32)
    qpos_idxs = np.ascontiguousarray(payload["qpos_idxs"], dtype=np.int32)
    qpos_target = np.ascontiguousarray(payload["qpos_target"], dtype=np.float32)
    eq_active_idxs = np.ascontiguousarray(payload.get("eq_active_idxs", []), dtype=np.int32)
    eq_active_vals = np.ascontiguousarray(payload.get("eq_active_vals", []), dtype=np.int32)

    if len(ctrl_idxs) != len(ctrl_vals) or len(qpos_idxs) != len(qpos_target):
        raise RoCoActionError("Native SimAction payload length mismatch.", code=ErrorCode.INVALID_ACTION_SHAPE)
    if model_nu is not None and (np.any(ctrl_idxs < 0) or np.any(ctrl_idxs >= int(model_nu))):
        raise RoCoActionError("Native SimAction contains invalid control index.", code=ErrorCode.INVALID_CONTROL_INDEX)
    if model_neq is not None and len(eq_active_idxs) > 0:
        if np.any(eq_active_idxs < 0) or np.any(eq_active_idxs >= int(model_neq)):
            raise RoCoActionError("Native SimAction contains invalid equality index.", code=ErrorCode.INVALID_CONTROL_INDEX)

    from rocobench.envs.base_env import SimAction

    return SimAction(
        ctrl_idxs=ctrl_idxs,
        ctrl_vals=ctrl_vals,
        qpos_idxs=qpos_idxs,
        qpos_target=qpos_target,
        eq_active_idxs=eq_active_idxs if len(eq_active_idxs) > 0 else None,
        eq_active_vals=eq_active_vals if len(eq_active_vals) > 0 else None,
    )
