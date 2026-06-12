from typing import Dict, List, Optional, Sequence

from .models import SkillPlan, SkillValidationResult, ValidationIssue
from .pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT
from .registry import SkillRegistry


class SkillPlanValidator:
    """Base semantic validator interface for skill plans."""

    def validate(self, plan: SkillPlan, obs) -> SkillValidationResult:
        raise NotImplementedError


def _ordered_item_match(names: Sequence[str], candidates) -> Optional[str]:
    for name in names:
        if name in candidates:
            return name
    return None


class PackGrocerySkillPlanValidator(SkillPlanValidator):
    """Semantic validator for PackGroceryTask Phase 1 skills."""

    def __init__(self, env, registry: SkillRegistry, agent_names: Optional[Sequence[str]] = None) -> None:
        self.env = env
        self.registry = registry
        if agent_names is None:
            agent_names = list(getattr(env, "robot_name_map", {}).values())
        self.agent_names = list(agent_names)

    def validate(self, plan: SkillPlan, obs) -> SkillValidationResult:
        issues = []  # type: List[ValidationIssue]
        calls_by_agent = {}
        for call in plan.calls:
            if call.agent_name in calls_by_agent:
                issues.append(ValidationIssue(
                    code="DUPLICATE_AGENT",
                    message="{} has more than one skill call.".format(call.agent_name),
                    agent_name=call.agent_name,
                ))
            calls_by_agent[call.agent_name] = call

        for agent_name in self.agent_names:
            if agent_name not in calls_by_agent:
                issues.append(ValidationIssue(
                    code="MISSING_AGENT",
                    message="Missing skill call for {}.".format(agent_name),
                    agent_name=agent_name,
                ))

        for call in plan.calls:
            if call.agent_name not in self.agent_names:
                issues.append(ValidationIssue(
                    code="UNKNOWN_AGENT",
                    message="{} is not a configured agent.".format(call.agent_name),
                    agent_name=call.agent_name,
                ))
            shape = self.registry.validate_call_shape(call)
            issues.extend(shape.issues)

        if issues:
            return SkillValidationResult.invalid(issues)

        object_claims = {}  # type: Dict[str, str]
        target_claims = {}  # type: Dict[str, str]
        progress_calls = []
        slot_occupancy = self._get_slot_occupancy(obs)
        held_by_agent = {
            agent_name: self._get_agent_held_object(obs, agent_name)
            for agent_name in self.agent_names
        }

        for call in plan.calls:
            if call.skill_name == WAIT:
                if call.arguments:
                    issues.append(ValidationIssue(
                        code="UNKNOWN_ARGUMENT",
                        message="WAIT does not accept arguments.",
                        agent_name=call.agent_name,
                    ))
                continue

            if call.skill_name != PUT_OBJECT_IN_CONTAINER:
                issues.append(ValidationIssue(
                    code="UNKNOWN_SKILL",
                    message="{} cannot execute {}.".format(call.agent_name, call.skill_name),
                    agent_name=call.agent_name,
                ))
                continue

            obj_name = call.arguments.get("object")
            slot_name = call.arguments.get("container")
            item_names = list(getattr(self.env, "item_names", []))
            slot_names = list(getattr(self.env, "bin_slot_xposes", {}).keys())

            if obj_name not in item_names:
                issues.append(ValidationIssue(
                    code="UNKNOWN_OBJECT",
                    message="{} is not a valid grocery object.".format(obj_name),
                    agent_name=call.agent_name,
                ))
                continue
            if not hasattr(obs, "objects") or obj_name not in obs.objects:
                issues.append(ValidationIssue(
                    code="UNKNOWN_OBJECT",
                    message="{} is not present in the current observation.".format(obj_name),
                    agent_name=call.agent_name,
                ))
                continue
            if slot_name not in slot_names:
                issues.append(ValidationIssue(
                    code="UNKNOWN_TARGET",
                    message="{} is not a valid bin slot.".format(slot_name),
                    agent_name=call.agent_name,
                ))
                continue

            packed_slot = self._get_packed_slot_for_object(obs, obj_name)
            if packed_slot is not None:
                issues.append(ValidationIssue(
                    code="OBJECT_ALREADY_PACKED",
                    message="{} is already packed in {}.".format(obj_name, packed_slot),
                    agent_name=call.agent_name,
                ))

            occupant = slot_occupancy.get(slot_name)
            if occupant is not None:
                issues.append(ValidationIssue(
                    code="TARGET_OCCUPIED",
                    message="{} is already occupied by {}.".format(slot_name, occupant),
                    agent_name=call.agent_name,
                ))

            for other_agent, held_obj in held_by_agent.items():
                if other_agent != call.agent_name and held_obj == obj_name:
                    issues.append(ValidationIssue(
                        code="OBJECT_HELD_BY_OTHER_AGENT",
                        message="{} is already held by {}.".format(obj_name, other_agent),
                        agent_name=call.agent_name,
                    ))

            own_held = held_by_agent.get(call.agent_name)
            if own_held is not None and own_held != obj_name:
                issues.append(ValidationIssue(
                    code="AGENT_HOLDING_DIFFERENT_OBJECT",
                    message="{} is holding {}, not {}.".format(call.agent_name, own_held, obj_name),
                    agent_name=call.agent_name,
                ))

            if obj_name in object_claims:
                issues.append(ValidationIssue(
                    code="RESOURCE_CONFLICT",
                    message="{} and {} both target {}.".format(object_claims[obj_name], call.agent_name, obj_name),
                    agent_name=call.agent_name,
                ))
            else:
                object_claims[obj_name] = call.agent_name

            if slot_name in target_claims:
                issues.append(ValidationIssue(
                    code="RESOURCE_CONFLICT",
                    message="{} and {} both target {}.".format(target_claims[slot_name], call.agent_name, slot_name),
                    agent_name=call.agent_name,
                ))
            else:
                target_claims[slot_name] = call.agent_name

            progress_calls.append(call)

        if not progress_calls:
            issues.append(ValidationIssue(
                code="NO_PROGRESS",
                message="At least one agent must execute PUT_OBJECT_IN_CONTAINER.",
            ))

        if issues:
            return SkillValidationResult.invalid(issues)
        return SkillValidationResult.ok()

    def _get_agent_held_object(self, obs, agent_name: str) -> Optional[str]:
        if hasattr(self.env, "get_agent_held_object"):
            return self.env.get_agent_held_object(obs, agent_name)
        robot_name = getattr(self.env, "robot_name_map_inv", {}).get(agent_name)
        if robot_name is None:
            return None
        robot_state = getattr(obs, robot_name, None)
        contacts = getattr(robot_state, "contacts", set()) if robot_state is not None else set()
        return _ordered_item_match(getattr(self.env, "item_names", []), contacts)

    def _get_packed_slot_for_object(self, obs, object_name: str) -> Optional[str]:
        if hasattr(self.env, "get_packed_slot_for_object"):
            return self.env.get_packed_slot_for_object(obs, object_name)
        obj_state = obs.objects.get(object_name)
        if obj_state is None or "bin_inside" not in getattr(obj_state, "contacts", set()):
            return None
        occupancy = self._get_slot_occupancy(obs)
        for slot_name, occupant in occupancy.items():
            if occupant == object_name:
                return slot_name
        return None

    def _get_slot_occupancy(self, obs) -> Dict[str, Optional[str]]:
        if hasattr(self.env, "get_slot_occupancy"):
            return self.env.get_slot_occupancy(obs)
        return {slot_name: None for slot_name in getattr(self.env, "bin_slot_xposes", {})}
