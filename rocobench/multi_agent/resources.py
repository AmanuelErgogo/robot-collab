"""Typed resource claims for multi-agent skill calls."""

from typing import Any, Dict, List, Sequence, Tuple

from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT

from .models import SkillResourceClaim, WorkspaceTrajectoryHint
from .workspace import WorkspaceMapper


class ResourceClaimDeriver(object):
    def __init__(self, workspace_mapper=None) -> None:
        self.workspace_mapper = workspace_mapper or WorkspaceMapper()

    def derive_for_call(self, call: Any, env: Any = None, obs: Any = None) -> SkillResourceClaim:
        del obs
        exclusive = ["robot:{}".format(call.agent_name)]
        shared = []
        zones = []
        reason = ""
        if call.skill_name == WAIT:
            reason = "WAIT reserves only the robot hold resource."
            return SkillResourceClaim(
                agent_name=call.agent_name,
                skill_name=call.skill_name,
                exclusive=tuple(exclusive),
                shared=tuple(shared),
                workspace_trajectory_hint=WorkspaceTrajectoryHint(tuple(zones), reason),
            )
        if call.skill_name == PUT_OBJECT_IN_CONTAINER:
            obj = call.arguments.get("object", "")
            target = call.arguments.get("container", "")
            exclusive.extend(["object:{}".format(obj), "target:{}".format(target)])
            zones = self.workspace_mapper.zones_for_put(call.agent_name, obj, target, env=env)
            reason = "PUT_OBJECT_IN_CONTAINER corridor from agent/object to target."
        else:
            zones = ("unknown",)
            reason = "Unknown skill; default to uncertain workspace."
        return SkillResourceClaim(
            agent_name=call.agent_name,
            skill_name=call.skill_name,
            exclusive=tuple(exclusive),
            shared=tuple(shared),
            workspace_trajectory_hint=WorkspaceTrajectoryHint(tuple(zones), reason),
        )

    def derive_for_plan(self, plan: Any, env: Any = None, obs: Any = None) -> Dict[str, SkillResourceClaim]:
        claims = {}
        for call in plan.calls:
            claims[call.agent_name] = self.derive_for_call(call, env=env, obs=obs)
        return claims


def exclusive_resource_conflicts(claims: Sequence[SkillResourceClaim]) -> Tuple[str, ...]:
    owner_by_resource = {}
    conflicts = []
    for claim in claims:
        for resource in claim.exclusive:
            owner = owner_by_resource.get(resource)
            if owner is not None and owner != claim.agent_name:
                conflicts.append("{} claimed by {} and {}".format(resource, owner, claim.agent_name))
            owner_by_resource[resource] = claim.agent_name
    return tuple(conflicts)


def workspace_conflicts(claims: Sequence[SkillResourceClaim]) -> Tuple[str, ...]:
    conflicts = []
    for index, left in enumerate(claims):
        left_zones = set(left.workspace_trajectory_hint.zones)
        if "unknown" in left_zones or "handoff" in left_zones:
            conflicts.append("{} has uncertain workspace.".format(left.agent_name))
            continue
        for right in claims[index + 1 :]:
            right_zones = set(right.workspace_trajectory_hint.zones)
            if "unknown" in right_zones or "handoff" in right_zones:
                conflicts.append("{} has uncertain workspace.".format(right.agent_name))
                continue
            overlap = sorted(left_zones.intersection(right_zones))
            if overlap:
                conflicts.append(
                    "{} and {} overlap workspace zones {}.".format(left.agent_name, right.agent_name, ",".join(overlap))
                )
    return tuple(conflicts)
