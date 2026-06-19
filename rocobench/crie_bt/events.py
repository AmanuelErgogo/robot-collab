"""Runtime event helpers for CRIE-BT."""

from typing import Iterable, List, Optional

from .types import BTDecision, RuntimeEvent, SkillCall


class RuntimeEventLog(object):
    """Small append-only event log used by controllers and tests."""

    def __init__(self) -> None:
        self._events = []  # type: List[RuntimeEvent]

    def append(
        self,
        event_type: str,
        message: str = "",
        step_id: Optional[str] = None,
        skill_call: Optional[SkillCall] = None,
        decision: Optional[BTDecision] = None,
        **payload
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_type=event_type,
            message=message,
            step_id=step_id,
            skill_call=skill_call,
            decision=decision,
            payload=payload,
        )
        self._events.append(event)
        return event

    def extend(self, events: Iterable[RuntimeEvent]) -> None:
        self._events.extend(list(events))

    def clear(self) -> None:
        self._events = []

    def to_list(self) -> List[RuntimeEvent]:
        return list(self._events)

    def to_dicts(self):
        return [event.to_dict() for event in self._events]
