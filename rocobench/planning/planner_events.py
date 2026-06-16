"""Structured event log for Phase 6 planner runs."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class PlannerEventType(Enum):
    OBSERVED = "OBSERVED"
    PROMPTED = "PROMPTED"
    PLAN_PARSED = "PLAN_PARSED"
    PLAN_REJECTED = "PLAN_REJECTED"
    EXECUTOR_SELECTED = "EXECUTOR_SELECTED"
    STATE_SNAPSHOTTED = "STATE_SNAPSHOTTED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    STATE_RESTORED = "STATE_RESTORED"
    FEEDBACK_RENDERED = "FEEDBACK_RENDERED"
    REPLAN_REQUESTED = "REPLAN_REQUESTED"
    FALLBACK_STARTED = "FALLBACK_STARTED"
    SKILL_SUCCEEDED = "SKILL_SUCCEEDED"
    TASK_SUCCEEDED = "TASK_SUCCEEDED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TASK_ABORTED = "TASK_ABORTED"


@dataclass(frozen=True)
class PlannerEvent:
    event_type: str
    run_id: str
    index: int
    timestamp_utc: str
    plan_id: Optional[str] = None
    backend: Optional[str] = None
    state_digest: Optional[str] = None
    reason: Optional[str] = None
    budget: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "index": int(self.index),
            "timestamp_utc": self.timestamp_utc,
            "plan_id": self.plan_id,
            "backend": self.backend,
            "state_digest": self.state_digest,
            "reason": self.reason,
            "budget": dict(self.budget or {}),
            "metadata": dict(self.metadata),
        }


class PlannerEventLog(object):
    def __init__(self, run_id: str = "phase6") -> None:
        self.run_id = str(run_id)
        self.events = []  # type: List[PlannerEvent]

    def append(
        self,
        event_type: Any,
        plan_id: Optional[str] = None,
        backend: Optional[str] = None,
        state_digest: Optional[str] = None,
        reason: Optional[str] = None,
        budget: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlannerEvent:
        value = event_type.value if hasattr(event_type, "value") else str(event_type)
        event = PlannerEvent(
            event_type=value,
            run_id=self.run_id,
            index=len(self.events),
            timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            plan_id=plan_id,
            backend=backend,
            state_digest=state_digest,
            reason=reason,
            budget=dict(budget or {}),
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    def event_types(self) -> List[str]:
        return [event.event_type for event in self.events]

    def to_list(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def from_events(cls, run_id: str, events: Iterable[PlannerEvent]) -> "PlannerEventLog":
        log = cls(run_id)
        log.events = list(events)
        return log
