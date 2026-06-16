"""Current-state summaries used for grounded planner feedback."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StateSummary:
    measured_facts: Tuple[str, ...]
    inferred_facts: Tuple[str, ...]
    digest: str
    quantized_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured_facts": list(self.measured_facts),
            "inferred_facts": list(self.inferred_facts),
            "digest": self.digest,
            "quantized_state": self.quantized_state,
        }


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _round_position(xpos: Any) -> Optional[Tuple[float, float, float]]:
    if xpos is None:
        return None
    try:
        values = list(xpos)
    except TypeError:
        return None
    if len(values) < 3:
        return None
    return tuple(round(float(values[index]), 3) for index in range(3))


def _object_state(obs: Any, object_name: str) -> Any:
    objects = getattr(obs, "objects", None)
    if isinstance(objects, dict):
        return objects.get(object_name)
    return None


def _env_items(env: Any, focus_object: Optional[str]) -> List[str]:
    names = list(getattr(env, "item_names", []) or [])
    if focus_object and focus_object not in names:
        names.insert(0, focus_object)
    if focus_object:
        names = [focus_object] + [name for name in names if name != focus_object]
    return names


def _get_held(env: Any, obs: Any, agent_name: str) -> Optional[str]:
    if hasattr(env, "get_agent_held_object"):
        return env.get_agent_held_object(obs, agent_name)
    robot_name = getattr(env, "robot_name_map_inv", {}).get(agent_name)
    robot_state = getattr(obs, robot_name, None) if robot_name is not None else None
    contacts = getattr(robot_state, "contacts", set()) if robot_state is not None else set()
    for item_name in getattr(env, "item_names", []):
        if item_name in contacts:
            return item_name
    return None


def _get_slot_occupancy(env: Any, obs: Any) -> Dict[str, Optional[str]]:
    if hasattr(env, "get_slot_occupancy"):
        return dict(env.get_slot_occupancy(obs))
    return {slot_name: None for slot_name in getattr(env, "bin_slot_xposes", {})}


def _packed_slot(env: Any, obs: Any, object_name: str) -> Optional[str]:
    if hasattr(env, "get_packed_slot_for_object"):
        return env.get_packed_slot_for_object(obs, object_name)
    return None


def build_state_summary(
    env: Any,
    obs: Any,
    agent_names: Optional[Sequence[str]] = None,
    focus_call: Any = None,
    inferred_facts: Optional[Iterable[str]] = None,
) -> StateSummary:
    """Build a deterministic summary from the supplied observation only."""
    if agent_names is None:
        agent_names = list(getattr(env, "robot_name_map", {}).values())
    focus_object = None
    focus_target = None
    if focus_call is not None:
        focus_object = focus_call.arguments.get("object")
        focus_target = focus_call.arguments.get("container")

    measured = []  # type: List[str]
    quantized = {"objects": {}, "holding": {}, "slots": {}}  # type: Dict[str, Any]

    for object_name in _env_items(env, focus_object):
        obj_state = _object_state(obs, object_name)
        packed = _packed_slot(env, obs, object_name)
        xpos = _round_position(getattr(obj_state, "xpos", None)) if obj_state is not None else None
        quantized["objects"][object_name] = {"packed_slot": packed, "xpos": xpos}
        if packed is not None:
            measured.append("{} is packed in {}.".format(object_name, packed))
        elif xpos is not None:
            measured.append("{} position is ({:.3f}, {:.3f}, {:.3f}).".format(object_name, xpos[0], xpos[1], xpos[2]))
            if focus_target and object_name == focus_object:
                measured.append("{} requested target is {}.".format(object_name, focus_target))
        elif object_name == focus_object:
            measured.append("{} is present but no position was reported.".format(object_name))

    for agent_name in agent_names:
        held = _get_held(env, obs, agent_name)
        quantized["holding"][agent_name] = held
        measured.append("{} holds {}.".format(agent_name, held if held is not None else "nothing"))

    occupancy = _get_slot_occupancy(env, obs)
    for slot_name in sorted(occupancy):
        occupant = occupancy.get(slot_name)
        quantized["slots"][slot_name] = occupant
        measured.append("{} is {}.".format(slot_name, "empty" if occupant is None else "occupied by {}".format(occupant)))

    if not measured and hasattr(env, "describe_obs"):
        measured.append("Scene description: {}".format(env.describe_obs(obs)))
    if not measured:
        measured.append("No structured measured facts were available from the current observation.")

    inferred = tuple(str(item) for item in (inferred_facts or ()))
    quantized_state = json.dumps(quantized, sort_keys=True, default=str)
    digest = _stable_digest({"measured": measured, "quantized": quantized})
    return StateSummary(
        measured_facts=tuple(measured),
        inferred_facts=inferred,
        digest=digest,
        quantized_state=quantized_state,
    )
