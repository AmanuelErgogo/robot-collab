"""CRIE-BT typed data models.

These dataclasses are deliberately lightweight and JSON-serializable. They form
the common contract among planners, executors, behavior-tree controllers,
communication managers, and evaluation scripts.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .status import (
    BTStatus,
    ExecutionMode,
    FailureCode,
    ProgressStage,
    RuntimeDecision,
    coerce_enum,
    enum_value,
)


def _jsonable(value):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (BTStatus, ExecutionMode, FailureCode, ProgressStage, RuntimeDecision)):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class SkillCall:
    agent: str
    skill_name: str
    arguments: Mapping[str, str] = field(default_factory=dict)
    instruction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": str(self.agent),
            "skill_name": str(self.skill_name),
            "arguments": {str(k): str(v) for k, v in dict(self.arguments).items()},
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillCall":
        return cls(
            agent=str(data["agent"]),
            skill_name=str(data["skill_name"]),
            arguments={str(k): str(v) for k, v in dict(data.get("arguments", {})).items()},
            instruction=data.get("instruction"),
        )

    @property
    def key(self) -> str:
        args = ",".join("{}={}".format(k, v) for k, v in sorted(dict(self.arguments).items()))
        return "{}:{}:{}".format(self.agent, self.skill_name, args)


@dataclass(frozen=True)
class SubtaskCommand:
    subtask_id: str
    skill_call: SkillCall
    goal: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "skill_call": self.skill_call.to_dict(),
            "goal": self.goal,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubtaskCommand":
        return cls(
            subtask_id=str(data["subtask_id"]),
            skill_call=SkillCall.from_dict(data["skill_call"]),
            goal=str(data.get("goal", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RoleAssignment:
    assignments: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"assignments": {str(k): str(v) for k, v in dict(self.assignments).items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoleAssignment":
        if "assignments" in data:
            data = data["assignments"]
        return cls(assignments={str(k): str(v) for k, v in dict(data).items()})


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    skill_call: SkillCall
    role_assignment: Mapping[str, str] = field(default_factory=dict)
    explanation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "skill_call": self.skill_call.to_dict(),
            "role_assignment": {str(k): str(v) for k, v in dict(self.role_assignment).items()},
            "explanation": self.explanation,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanStep":
        return cls(
            step_id=str(data["step_id"]),
            skill_call=SkillCall.from_dict(data["skill_call"]),
            role_assignment={str(k): str(v) for k, v in dict(data.get("role_assignment", {})).items()},
            explanation=str(data.get("explanation", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CollaborativePlan:
    steps: List[PlanStep]
    plan_id: str = "plan_001"
    task_goal: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_goal": self.task_goal,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollaborativePlan":
        return cls(
            plan_id=str(data.get("plan_id", "plan_001")),
            task_goal=str(data.get("task_goal", "")),
            steps=[PlanStep.from_dict(item) for item in data.get("steps", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExecutionContext:
    mode: ExecutionMode = ExecutionMode.BT_MEDIATED
    task_name: str = ""
    episode_id: str = ""
    max_steps: int = 100
    max_retries: int = 1
    step_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": enum_value(self.mode),
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "max_steps": int(self.max_steps),
            "max_retries": int(self.max_retries),
            "step_index": int(self.step_index),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionContext":
        return cls(
            mode=coerce_enum(ExecutionMode, data.get("mode", ExecutionMode.BT_MEDIATED.value)),
            task_name=str(data.get("task_name", "")),
            episode_id=str(data.get("episode_id", "")),
            max_steps=int(data.get("max_steps", 100)),
            max_retries=int(data.get("max_retries", 1)),
            step_index=int(data.get("step_index", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ProgressState:
    stage: ProgressStage = ProgressStage.NOT_STARTED
    score: float = 0.0
    elapsed_steps: int = 0
    stagnant_steps: int = 0
    postcondition_satisfied: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": enum_value(self.stage),
            "score": float(self.score),
            "elapsed_steps": int(self.elapsed_steps),
            "stagnant_steps": int(self.stagnant_steps),
            "postcondition_satisfied": bool(self.postcondition_satisfied),
            "evidence": _jsonable(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgressState":
        return cls(
            stage=coerce_enum(ProgressStage, data.get("stage", ProgressStage.NOT_STARTED.value)),
            score=float(data.get("score", 0.0)),
            elapsed_steps=int(data.get("elapsed_steps", 0)),
            stagnant_steps=int(data.get("stagnant_steps", 0)),
            postcondition_satisfied=bool(data.get("postcondition_satisfied", False)),
            evidence=dict(data.get("evidence", {})),
        )


@dataclass(frozen=True)
class UncertaintyState:
    confidence: float = 1.0
    uncertainty: float = 0.0
    risk_level: str = "low"
    source: str = "none"
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "risk_level": self.risk_level,
            "source": self.source,
            "evidence": _jsonable(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UncertaintyState":
        return cls(
            confidence=float(data.get("confidence", 1.0)),
            uncertainty=float(data.get("uncertainty", 0.0)),
            risk_level=str(data.get("risk_level", "low")),
            source=str(data.get("source", "none")),
            evidence=dict(data.get("evidence", {})),
        )


@dataclass(frozen=True)
class FailureState:
    is_failure: bool = False
    failure_code: FailureCode = FailureCode.NONE
    severity: str = "none"
    message: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_failure": bool(self.is_failure),
            "failure_code": enum_value(self.failure_code),
            "severity": self.severity,
            "message": self.message,
            "evidence": _jsonable(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureState":
        return cls(
            is_failure=bool(data.get("is_failure", False)),
            failure_code=coerce_enum(FailureCode, data.get("failure_code", FailureCode.NONE.value), FailureCode.UNKNOWN),
            severity=str(data.get("severity", "none")),
            message=str(data.get("message", "")),
            evidence=dict(data.get("evidence", {})),
        )


@dataclass(frozen=True)
class ExecutionFeedback:
    skill_call: SkillCall
    status: BTStatus = BTStatus.RUNNING
    progress: ProgressState = field(default_factory=ProgressState)
    uncertainty: UncertaintyState = field(default_factory=UncertaintyState)
    failure: FailureState = field(default_factory=FailureState)
    message: str = ""
    raw_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_call": self.skill_call.to_dict(),
            "status": enum_value(self.status),
            "progress": self.progress.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "failure": self.failure.to_dict(),
            "message": self.message,
            "raw_info": _jsonable(self.raw_info),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionFeedback":
        return cls(
            skill_call=SkillCall.from_dict(data["skill_call"]),
            status=coerce_enum(BTStatus, data.get("status", BTStatus.RUNNING.value)),
            progress=ProgressState.from_dict(data.get("progress", {})),
            uncertainty=UncertaintyState.from_dict(data.get("uncertainty", {})),
            failure=FailureState.from_dict(data.get("failure", {})),
            message=str(data.get("message", "")),
            raw_info=dict(data.get("raw_info", {})),
        )


@dataclass(frozen=True)
class BTDecision:
    decision: RuntimeDecision = RuntimeDecision.CONTINUE
    reason: str = ""
    retry_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": enum_value(self.decision),
            "reason": self.reason,
            "retry_count": int(self.retry_count),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BTDecision":
        return cls(
            decision=coerce_enum(RuntimeDecision, data.get("decision", RuntimeDecision.CONTINUE.value)),
            reason=str(data.get("reason", "")),
            retry_count=int(data.get("retry_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RuntimeEvent:
    event_type: str
    message: str = ""
    step_id: Optional[str] = None
    skill_call: Optional[SkillCall] = None
    decision: Optional[BTDecision] = None
    timestamp_s: float = field(default_factory=time.time)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "step_id": self.step_id,
            "skill_call": self.skill_call.to_dict() if self.skill_call is not None else None,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "timestamp_s": float(self.timestamp_s),
            "payload": _jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEvent":
        skill_data = data.get("skill_call")
        decision_data = data.get("decision")
        return cls(
            event_type=str(data["event_type"]),
            message=str(data.get("message", "")),
            step_id=data.get("step_id"),
            skill_call=SkillCall.from_dict(skill_data) if skill_data else None,
            decision=BTDecision.from_dict(decision_data) if decision_data else None,
            timestamp_s=float(data.get("timestamp_s", time.time())),
            payload=dict(data.get("payload", {})),
        )


@dataclass(frozen=True)
class ControllerResult:
    status: BTStatus = BTStatus.RUNNING
    decision: BTDecision = field(default_factory=BTDecision)
    feedback: Optional[ExecutionFeedback] = None
    events: List[RuntimeEvent] = field(default_factory=list)
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": enum_value(self.status),
            "decision": self.decision.to_dict(),
            "feedback": self.feedback.to_dict() if self.feedback is not None else None,
            "events": [event.to_dict() for event in self.events],
            "completed_subtasks": int(self.completed_subtasks),
            "failed_subtasks": int(self.failed_subtasks),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControllerResult":
        feedback_data = data.get("feedback")
        return cls(
            status=coerce_enum(BTStatus, data.get("status", BTStatus.RUNNING.value)),
            decision=BTDecision.from_dict(data.get("decision", {})),
            feedback=ExecutionFeedback.from_dict(feedback_data) if feedback_data else None,
            events=[RuntimeEvent.from_dict(item) for item in data.get("events", [])],
            completed_subtasks=int(data.get("completed_subtasks", 0)),
            failed_subtasks=int(data.get("failed_subtasks", 0)),
            message=str(data.get("message", "")),
        )
