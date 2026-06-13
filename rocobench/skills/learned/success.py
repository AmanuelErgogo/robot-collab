"""Stable learned skill success predicates."""

from dataclasses import dataclass, field
from typing import Any, Dict

from .models import MonitorEvent, ProgressStage


@dataclass
class StableSkillSuccessChecker:
    stable_checks: int = 2
    consecutive: int = 0
    last_evidence: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        self.consecutive = 0
        self.last_evidence = {}

    def check(self, env, obs, agent_name, object_name, target_name):
        if not hasattr(env, "get_packed_slot_for_object") or not hasattr(env, "get_agent_held_object"):
            self.consecutive = 0
            self.last_evidence = {"measurable": False, "reason": "missing task predicate helpers"}
            return False

        packed_slot = env.get_packed_slot_for_object(obs, object_name)
        held = env.get_agent_held_object(obs, agent_name)
        occupancy = env.get_slot_occupancy(obs) if hasattr(env, "get_slot_occupancy") else {}
        target_occupant = occupancy.get(target_name)
        target_ok = packed_slot == target_name
        released = held is None
        no_conflict = target_occupant in (None, object_name)
        ok = bool(target_ok and released and no_conflict)
        self.last_evidence = {
            "object": object_name,
            "target": target_name,
            "packed_slot": packed_slot,
            "held_by_agent": held,
            "target_occupant": target_occupant,
            "target_ok": target_ok,
            "released": released,
            "no_conflict": no_conflict,
            "stable_count": self.consecutive + 1 if ok else 0,
        }
        if ok:
            self.consecutive += 1
        else:
            self.consecutive = 0
        return self.consecutive >= int(self.stable_checks)

    def progress_stage(self):
        if self.consecutive >= int(self.stable_checks):
            return ProgressStage.STABLE.value
        if self.last_evidence.get("released"):
            return ProgressStage.RELEASED.value
        return ProgressStage.NOT_STARTED.value

