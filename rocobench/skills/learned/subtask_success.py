"""Task-agnostic success predicates for each subtask skill type.

Each checker is a callable that receives (env, obs, call_args) and returns
(success: bool, evidence: dict). They are stateful so they can implement
stability windows (require N consecutive successes before declaring done).

call_args is the dict from the active SkillCall.arguments.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class SubtaskSuccessChecker:
    """Base for all subtask success checkers."""

    def __init__(self, stable_checks: int = 2):
        self.stable_checks = int(stable_checks)
        self._consecutive = 0
        self.last_evidence: Dict[str, Any] = {}

    def reset(self) -> None:
        self._consecutive = 0
        self.last_evidence = {}

    def _once(self, env, obs, call_args: Dict[str, str]) -> bool:
        raise NotImplementedError

    def check(self, env, obs, call_args: Dict[str, str]) -> bool:
        try:
            ok = self._once(env, obs, call_args)
        except Exception as exc:
            self.last_evidence = {"error": str(exc)}
            self._consecutive = 0
            return False
        self.last_evidence["stable_count"] = self._consecutive + 1 if ok else 0
        if ok:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.stable_checks

    def progress_stage(self) -> str:
        return "stable" if self._consecutive >= self.stable_checks else "running"


# ---------------------------------------------------------------------------
# PICK success: robot contacts the object and lifts it above a threshold
# ---------------------------------------------------------------------------

class PickSuccessChecker(SubtaskSuccessChecker):
    """
    Declares success when the robot's end-effector is in contact with the
    target object AND the object is above a minimum lift height.

    Works for any task that uses MuJoCo contact-based state (base_env).
    """

    LIFT_THRESHOLD = 0.05  # metres above object's initial resting z

    def _once(self, env, obs, call_args: Dict[str, str]) -> bool:
        obj_name = call_args.get("object", "")
        obj_state = obs.objects.get(obj_name)
        if obj_state is None:
            self.last_evidence = {"error": f"object '{obj_name}' not in obs"}
            return False

        # Check if any robot contacts the object
        in_contact = False
        holding_agent = None
        for robot_name in env.agent_configs:
            agent_state = getattr(obs, robot_name, None)
            if agent_state is not None and obj_name in agent_state.contacts:
                in_contact = True
                holding_agent = robot_name
                break
        # Also accept weld-based contact (equality constraints fire before contact)
        if not in_contact:
            contacts = obs.objects.get(obj_name, None)
            if contacts is not None:
                for robot_name in env.agent_configs:
                    if robot_name in contacts.contacts:
                        in_contact = True
                        holding_agent = robot_name
                        break

        lift_height = float(obj_state.xpos[2])
        lifted = lift_height > self.LIFT_THRESHOLD

        self.last_evidence = {
            "object": obj_name,
            "in_contact": in_contact,
            "holding_agent": holding_agent,
            "object_z": lift_height,
            "lift_threshold": self.LIFT_THRESHOLD,
            "lifted": lifted,
        }
        return in_contact and lifted


# ---------------------------------------------------------------------------
# PLACE success: object is at target, robot is not holding it
# ---------------------------------------------------------------------------

class PlaceSuccessChecker(SubtaskSuccessChecker):
    """
    Declares success when:
    1. The held object is no longer in contact with any robot EE.
    2. The object is in contact with the target body or within target_radius.

    target in call_args can be a body name or (x,y,z) string.
    """

    RELEASE_DIST = 0.08  # max distance from target centre to count as placed

    def _once(self, env, obs, call_args: Dict[str, str]) -> bool:
        obj_name = call_args.get("object", "")
        target_name = call_args.get("target", "")
        obj_state = obs.objects.get(obj_name)
        if obj_state is None:
            self.last_evidence = {"error": f"object '{obj_name}' not in obs"}
            return False

        # Robot has released the object
        held_by = None
        for robot_name in env.agent_configs:
            agent_state = getattr(obs, robot_name, None)
            if agent_state is not None and obj_name in agent_state.contacts:
                held_by = robot_name
                break

        # Object is near/on the target
        near_target = False
        dist_to_target = None
        target_pos = env.get_target_pos(None, target_name)
        if target_pos is not None:
            dist_to_target = float(np.linalg.norm(obj_state.xpos - np.array(target_pos)))
            near_target = dist_to_target < self.RELEASE_DIST
        else:
            # Fall back to contact check
            target_state = obs.objects.get(target_name)
            if target_state is not None:
                near_target = target_name in obj_state.contacts or obj_name in target_state.contacts
            else:
                near_target = target_name in obj_state.contacts

        released = held_by is None
        self.last_evidence = {
            "object": obj_name,
            "target": target_name,
            "held_by": held_by,
            "released": released,
            "near_target": near_target,
            "dist_to_target": dist_to_target,
        }
        return released and near_target


# ---------------------------------------------------------------------------
# STACK_ON success: object rests on top of another with correct vertical order
# ---------------------------------------------------------------------------

class StackOnSuccessChecker(SubtaskSuccessChecker):
    """
    Declares success when the carried ingredient is placed on the target
    ingredient/board AND is above the target (positive z-offset).
    """

    MAX_LATERAL_DIST = 0.06  # metres — centring tolerance
    MIN_Z_ABOVE = 0.005       # object bottom must be above target top

    def _once(self, env, obs, call_args: Dict[str, str]) -> bool:
        obj_name = call_args.get("object", "")
        target_name = call_args.get("target", "")
        obj_state = obs.objects.get(obj_name)
        target_state = obs.objects.get(target_name)

        if obj_state is None:
            self.last_evidence = {"error": f"object '{obj_name}' not in obs"}
            return False
        if target_state is None:
            self.last_evidence = {"error": f"target '{target_name}' not in obs"}
            return False

        released = all(
            obj_name not in (getattr(obs, r, None) or type("", (), {"contacts": set()})()).contacts
            for r in env.agent_configs
        )
        lateral_dist = float(np.linalg.norm(obj_state.xpos[:2] - target_state.xpos[:2]))
        centred = lateral_dist < self.MAX_LATERAL_DIST
        z_above = float(obj_state.bottom_height - target_state.top_height)
        stacked = z_above >= self.MIN_Z_ABOVE

        self.last_evidence = {
            "object": obj_name,
            "target": target_name,
            "released": released,
            "lateral_dist": lateral_dist,
            "z_above": z_above,
            "centred": centred,
            "stacked": stacked,
        }
        return released and centred and stacked


# ---------------------------------------------------------------------------
# OPEN_CABINET success: door angle exceeds threshold
# ---------------------------------------------------------------------------

class OpenCabinetSuccessChecker(SubtaskSuccessChecker):
    """
    Declares success when the target door's hinge joint exceeds a minimum
    open angle (default 30°).
    """

    MIN_OPEN_ANGLE = np.radians(30)

    _DOOR_JOINT_MAP = {
        "left_door":  "leftdoorhinge",
        "right_door": "rightdoorhinge",
        "left_door_handle":  "leftdoorhinge",
        "right_door_handle": "rightdoorhinge",
    }

    def _once(self, env, obs, call_args: Dict[str, str]) -> bool:
        door_name = call_args.get("door", "")
        joint_name = self._DOOR_JOINT_MAP.get(door_name)
        if joint_name is None:
            self.last_evidence = {"error": f"unknown door '{door_name}'"}
            return False
        try:
            angle = float(env.physics.named.data.qpos[joint_name])
        except Exception as exc:
            self.last_evidence = {"error": str(exc)}
            return False
        open_enough = abs(angle) >= self.MIN_OPEN_ANGLE
        self.last_evidence = {
            "door": door_name,
            "joint": joint_name,
            "angle_deg": np.degrees(abs(angle)),
            "threshold_deg": np.degrees(self.MIN_OPEN_ANGLE),
            "open_enough": open_enough,
        }
        return open_enough


# ---------------------------------------------------------------------------
# PICK_AND_PLACE success: object at target, not held
# ---------------------------------------------------------------------------

class PickAndPlaceSuccessChecker(PlaceSuccessChecker):
    """Reuses PlaceSuccessChecker — the combined skill ends with placement."""
    pass


# ---------------------------------------------------------------------------
# Registry: skill_name → checker class
# ---------------------------------------------------------------------------

_CHECKER_MAP: Dict[str, type] = {
    "PICK": PickSuccessChecker,
    "PLACE": PlaceSuccessChecker,
    "PICK_AND_PLACE": PickAndPlaceSuccessChecker,
    "OPEN_CABINET": OpenCabinetSuccessChecker,
    "STACK_ON": StackOnSuccessChecker,
}


def build_success_checker(skill_name: str, stable_checks: int = 2) -> SubtaskSuccessChecker:
    """Return the right success checker for a given skill name."""
    cls = _CHECKER_MAP.get(skill_name.upper())
    if cls is None:
        raise ValueError(f"No success checker registered for skill '{skill_name}'. "
                         f"Available: {sorted(_CHECKER_MAP)}")
    return cls(stable_checks=stable_checks)
