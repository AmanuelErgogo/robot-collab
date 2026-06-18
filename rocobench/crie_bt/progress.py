"""Progress monitoring helpers for CRIE-BT manipulation skills."""

import math
from typing import Any, Mapping, Optional

from .status import ProgressStage
from .types import ProgressState, SkillCall


def _get_nested(value: Any, *keys) -> Any:
    current = value
    for key in keys:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _position(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("position", "pos", "xpos"):
            if key in value:
                return value[key]
    for key in ("position", "pos", "xpos"):
        if hasattr(value, key):
            return getattr(value, key)
    return value if isinstance(value, (list, tuple)) and len(value) >= 3 else None


def _distance(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))
    except Exception:
        return None


def get_object_position(obs: Any, object_name: str):
    return _position(_get_nested(obs, "objects", object_name) or _get_nested(obs, object_name))


def get_target_position(env: Any, target_name: str):
    if env is not None and hasattr(env, "get_target_position"):
        try:
            return env.get_target_position(target_name)
        except Exception:
            pass
    return _position(_get_nested(env, "bin_slot_xposes", target_name) or _get_nested(env, "targets", target_name))


def get_gripper_position(obs: Any, agent_name: str):
    return _position(
        _get_nested(obs, "agents", agent_name, "gripper_position")
        or _get_nested(obs, "agents", agent_name, "eef_pos")
        or _get_nested(obs, agent_name, "gripper_position")
    )


def is_object_grasped(obs: Any, agent_name: str, object_name: str) -> Optional[bool]:
    held = _get_nested(obs, "held", agent_name) or _get_nested(obs, "agents", agent_name, "held_object")
    if held is not None:
        return str(held) == str(object_name)
    contacts = _get_nested(obs, "agents", agent_name, "contacts")
    if contacts is not None:
        try:
            return object_name in contacts
        except Exception:
            return None
    return None


def is_object_at_target(obs: Any, object_name: str, target_name: str) -> Optional[bool]:
    packed = _get_nested(obs, "packed", object_name) or _get_nested(obs, "object_targets", object_name)
    if packed is not None:
        return str(packed) == str(target_name)
    occupancy = _get_nested(obs, "slot_occupancy", target_name)
    if occupancy is not None:
        return str(occupancy) == str(object_name)
    return None


class ProgressMonitor(object):
    def __init__(self, env: Any = None, no_progress_patience: int = 3, max_steps: int = 50) -> None:
        self.env = env
        self.no_progress_patience = int(no_progress_patience)
        self.max_steps = int(max_steps)
        self.skill_call = None
        self.elapsed_steps = 0
        self.last_score = 0.0
        self.stagnant_steps = 0
        self.was_grasped = False

    def reset(self, skill_call: SkillCall, observation: Any) -> None:
        self.skill_call = skill_call
        self.elapsed_steps = 0
        self.last_score = 0.0
        self.stagnant_steps = 0
        obj = skill_call.arguments.get("object", skill_call.arguments.get("target", ""))
        grasped = is_object_grasped(observation, skill_call.agent, obj)
        self.was_grasped = bool(grasped)

    def update(self, skill_call: SkillCall, observation: Any, action_info: Optional[dict] = None) -> ProgressState:
        self.elapsed_steps += 1
        action_info = dict(action_info or {})
        obj = skill_call.arguments.get("object", "")
        target = skill_call.arguments.get("container", skill_call.arguments.get("target", ""))
        obj_pos = get_object_position(observation, obj)
        target_pos = get_target_position(self.env or observation, target)
        gripper_pos = get_gripper_position(observation, skill_call.agent)
        grasped = is_object_grasped(observation, skill_call.agent, obj)
        at_target = is_object_at_target(observation, obj, target)
        gripper_object_dist = _distance(gripper_pos, obj_pos)
        object_target_dist = _distance(obj_pos, target_pos)

        evidence = {
            "object_position": obj_pos,
            "target_position": target_pos,
            "gripper_position": gripper_pos,
            "gripper_object_distance": gripper_object_dist,
            "object_target_distance": object_target_dist,
            "object_grasped": grasped,
            "object_at_target": at_target,
        }
        evidence.update(action_info)

        score = 0.0
        stage = ProgressStage.NOT_STARTED
        if gripper_object_dist is not None:
            score = max(score, max(0.0, 0.25 - min(gripper_object_dist, 1.0) * 0.2))
            stage = ProgressStage.APPROACHING_OBJECT
            if gripper_object_dist < 0.08:
                stage = ProgressStage.NEAR_OBJECT
                score = max(score, 0.35)
        if grasped:
            self.was_grasped = True
            stage = ProgressStage.OBJECT_GRASPED
            score = max(score, 0.55)
        if object_target_dist is not None:
            if grasped:
                stage = ProgressStage.TRANSPORTING
            score = max(score, max(0.0, 0.75 - min(object_target_dist, 1.0) * 0.3))
            if object_target_dist < 0.12:
                stage = ProgressStage.NEAR_TARGET
                score = max(score, 0.82)
        if at_target:
            stage = ProgressStage.STABLE_SUCCESS
            score = 1.0
        elif self.elapsed_steps >= self.max_steps:
            stage = ProgressStage.FAILED
        elif self.stagnant_steps >= self.no_progress_patience:
            stage = ProgressStage.STUCK

        if score <= self.last_score + 1e-4:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        self.last_score = max(self.last_score, score)

        if stage == ProgressStage.STUCK:
            score = self.last_score
        return ProgressState(
            stage=stage,
            score=score,
            elapsed_steps=self.elapsed_steps,
            stagnant_steps=self.stagnant_steps,
            postcondition_satisfied=bool(at_target),
            evidence=evidence,
        )
