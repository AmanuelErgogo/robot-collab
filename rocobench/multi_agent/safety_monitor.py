"""Central safety monitor for synchronized execution."""

from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .models import AgentActionFragment, SafetyEvent, SafetySeverity


class CentralSafetyMonitor(object):
    def __init__(self, max_policy_latency_ms: Optional[float] = None) -> None:
        self.max_policy_latency_ms = max_policy_latency_ms

    def check_actions(self, fragments: Mapping[str, AgentActionFragment], latencies_ms: Optional[Mapping[str, float]] = None) -> List[SafetyEvent]:
        events = []
        for agent, fragment in sorted(fragments.items()):
            arrays = fragment.arrays()
            if not np.all(np.isfinite(arrays["ctrl_vals"])) or not np.all(np.isfinite(arrays["qpos_target"])):
                events.append(SafetyEvent("NONFINITE_ACTION", SafetySeverity.STOP_ALL.value, "Nonfinite action fragment.", agent))
        if self.max_policy_latency_ms is not None:
            for agent, latency in sorted(dict(latencies_ms or {}).items()):
                if float(latency) > float(self.max_policy_latency_ms):
                    events.append(
                        SafetyEvent(
                            "POLICY_LATENCY_EXCEEDED",
                            SafetySeverity.STOP_ALL.value,
                            "Policy latency exceeded synchronized limit.",
                            agent,
                            {"latency_ms": float(latency), "limit_ms": float(self.max_policy_latency_ms)},
                        )
                    )
        return events

    def check_step_info(self, info: Mapping[str, Any]) -> List[SafetyEvent]:
        info = dict(info or {})
        events = []
        if info.get("collision") or info.get("contact_violation"):
            events.append(SafetyEvent("FORBIDDEN_CONTACT", SafetySeverity.STOP_ALL.value, "Forbidden contact reported.", evidence=info))
        if info.get("reservation_violation"):
            events.append(SafetyEvent("RESERVATION_VIOLATION", SafetySeverity.STOP_ALL.value, "Reservation violation reported.", evidence=info))
        if info.get("object_ownership_conflict"):
            events.append(SafetyEvent("OBJECT_OWNERSHIP_CONFLICT", SafetySeverity.STOP_ALL.value, "Object ownership conflict reported.", evidence=info))
        return events

    def should_stop_all(self, events: Sequence[SafetyEvent]) -> bool:
        return any(event.severity == SafetySeverity.STOP_ALL.value for event in events)
