"""Metrics for Phase 7 scheduling and execution."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MultiAgentMetrics:
    schedule_modes: Dict[str, int] = field(default_factory=dict)
    admissions: int = 0
    rejections: int = 0
    central_steps: int = 0
    stop_all_events: int = 0
    reservation_violations: int = 0
    makespan_steps: int = 0
    summed_agent_steps: int = 0

    def record_schedule(self, mode: str, accepted: bool) -> None:
        self.schedule_modes[mode] = self.schedule_modes.get(mode, 0) + 1
        if accepted:
            self.admissions += 1
        else:
            self.rejections += 1

    def to_dict(self) -> Dict[str, Any]:
        speedup = 0.0
        if self.makespan_steps > 0:
            speedup = float(self.summed_agent_steps) / float(self.makespan_steps)
        return {
            "schedule_modes": dict(self.schedule_modes),
            "admissions": int(self.admissions),
            "rejections": int(self.rejections),
            "central_steps": int(self.central_steps),
            "stop_all_events": int(self.stop_all_events),
            "reservation_violations": int(self.reservation_violations),
            "makespan_steps": int(self.makespan_steps),
            "summed_agent_steps": int(self.summed_agent_steps),
            "parallel_speedup": speedup,
        }
