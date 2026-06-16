"""Structured combined execution result for Phase 7."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from .metrics import MultiAgentMetrics
from .models import AgentExecutionOutcome, SafetyEvent, ScheduleDecision


@dataclass(frozen=True)
class CombinedExecutionResult:
    success: bool
    schedule: ScheduleDecision
    per_agent_outcomes: Mapping[str, AgentExecutionOutcome]
    central_status: str
    central_reason: str = ""
    safety_events: Tuple[SafetyEvent, ...] = ()
    fallback_to_sequential_recommended: bool = False
    learned_success: bool = False
    fallback_success: bool = False
    metrics: MultiAgentMetrics = field(default_factory=MultiAgentMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "schedule": self.schedule.to_dict(),
            "per_agent_outcomes": {
                agent: outcome.to_dict()
                for agent, outcome in sorted(self.per_agent_outcomes.items())
            },
            "central_status": self.central_status,
            "central_reason": self.central_reason,
            "safety_events": [event.to_dict() for event in self.safety_events],
            "fallback_to_sequential_recommended": bool(self.fallback_to_sequential_recommended),
            "learned_success": bool(self.learned_success),
            "fallback_success": bool(self.fallback_success),
            "metrics": self.metrics.to_dict(),
        }
