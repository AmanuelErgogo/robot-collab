"""Versioned compatibility rules for concurrent skill admission."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER, WAIT


@dataclass(frozen=True)
class CompatibilityDecision:
    compatible: bool
    rule_id: str
    reason: str
    concurrent: str = "false"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": bool(self.compatible),
            "rule_id": self.rule_id,
            "reason": self.reason,
            "concurrent": self.concurrent,
        }


class CompatibilityMatrix(object):
    version = "phase7.pack.v1"

    def evaluate_pair(self, left: Any, right: Any, left_claim: Any, right_claim: Any) -> CompatibilityDecision:
        skills = tuple(sorted([left.skill_name, right.skill_name]))
        if left.skill_name == WAIT or right.skill_name == WAIT:
            return CompatibilityDecision(True, "phase7.wait.concurrent", "WAIT is compatible with one active skill.", "true")
        if skills == (PUT_OBJECT_IN_CONTAINER, PUT_OBJECT_IN_CONTAINER):
            left_object = left.arguments.get("object")
            right_object = right.arguments.get("object")
            left_target = left.arguments.get("container")
            right_target = right.arguments.get("container")
            if left_object == right_object:
                return CompatibilityDecision(False, "phase7.pack.put.same_object", "Both agents target the same object.", "false")
            if left_target == right_target:
                return CompatibilityDecision(False, "phase7.pack.put.same_target", "Both agents target the same target.", "false")
            left_zones = set(left_claim.workspace_trajectory_hint.zones)
            right_zones = set(right_claim.workspace_trajectory_hint.zones)
            if "unknown" in left_zones or "unknown" in right_zones:
                return CompatibilityDecision(False, "phase7.pack.put.unknown_workspace", "Workspace is uncertain.", "conditional")
            overlap = sorted(left_zones.intersection(right_zones))
            if overlap:
                return CompatibilityDecision(
                    False,
                    "phase7.pack.put.overlap_workspace",
                    "Workspace zones overlap: {}.".format(",".join(overlap)),
                    "conditional",
                )
            return CompatibilityDecision(True, "phase7.pack.put.disjoint", "Distinct resources and disjoint workspace zones.", "conditional")
        return CompatibilityDecision(False, "phase7.no_rule", "No permissive compatibility rule exists.", "false")

    def evaluate_group(self, calls: Sequence[Any], claims_by_agent: Mapping[str, Any]) -> CompatibilityDecision:
        last_decision = None
        for index, left in enumerate(calls):
            for right in calls[index + 1 :]:
                decision = self.evaluate_pair(left, right, claims_by_agent[left.agent_name], claims_by_agent[right.agent_name])
                if not decision.compatible:
                    return decision
                last_decision = decision
        if last_decision is not None:
            return last_decision
        return CompatibilityDecision(True, "phase7.group.compatible", "All call pairs are compatible.", "conditional")
