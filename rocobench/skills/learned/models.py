"""Models for learned skill execution."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


class LearnedSkillExecutionStatus(Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    LOADING_POLICY = "LOADING_POLICY"
    RESETTING_POLICY = "RESETTING_POLICY"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    INTERRUPTED = "INTERRUPTED"
    FALLBACK_REQUESTED = "FALLBACK_REQUESTED"
    CLOSED = "CLOSED"


class ProgressStage(Enum):
    NOT_STARTED = "not_started"
    APPROACHING = "approaching"
    GRASPED = "grasped"
    TRANSPORTING = "transporting"
    OVER_TARGET = "over_target"
    RELEASED = "released"
    STABLE = "stable"


@dataclass(frozen=True)
class LearnedPolicySpec:
    policy_id: str
    skill_name: str
    agent_name: str
    embodiment_id: str
    task_id: str
    checkpoint: str
    checkpoint_revision: str
    policy_type: str
    schema_hash: str
    action_representation: str
    cameras: Tuple[str, ...]
    max_steps: int
    execution_horizon: int
    success_monitor: str
    failure_monitors: Tuple[str, ...]
    enabled: bool = True

    def __post_init__(self):
        object.__setattr__(self, "cameras", tuple(str(x) for x in self.cameras))
        object.__setattr__(self, "failure_monitors", tuple(str(x) for x in self.failure_monitors))
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")

    def to_dict(self):
        return {
            "policy_id": self.policy_id,
            "skill_name": self.skill_name,
            "agent_name": self.agent_name,
            "embodiment_id": self.embodiment_id,
            "task_id": self.task_id,
            "checkpoint": self.checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "policy_type": self.policy_type,
            "schema_hash": self.schema_hash,
            "action_representation": self.action_representation,
            "cameras": list(self.cameras),
            "max_steps": int(self.max_steps),
            "execution_horizon": int(self.execution_horizon),
            "success_monitor": self.success_monitor,
            "failure_monitors": list(self.failure_monitors),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            policy_id=str(data["policy_id"]),
            skill_name=str(data["skill_name"]),
            agent_name=str(data["agent_name"]),
            embodiment_id=str(data["embodiment_id"]),
            task_id=str(data["task_id"]),
            checkpoint=str(data["checkpoint"]),
            checkpoint_revision=str(data["checkpoint_revision"]),
            policy_type=str(data["policy_type"]),
            schema_hash=str(data["schema_hash"]),
            action_representation=str(data["action_representation"]),
            cameras=tuple(str(x) for x in data.get("cameras", ())),
            max_steps=int(data["max_steps"]),
            execution_horizon=int(data["execution_horizon"]),
            success_monitor=str(data["success_monitor"]),
            failure_monitors=tuple(str(x) for x in data.get("failure_monitors", ())),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class MonitorEvent:
    code: str
    severity: str
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    recommended_recovery: Optional[str] = None

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": dict(self.evidence),
            "recommended_recovery": self.recommended_recovery,
        }


@dataclass(frozen=True)
class CancellationToken:
    _cancelled: bool = False

    @property
    def cancelled(self):
        return bool(self._cancelled)

    def throw_if_cancelled(self):
        if self.cancelled:
            from .errors import LearnedExecutionError

            raise LearnedExecutionError("MANUAL_INTERRUPT", "Execution was cancelled.")


class MutableCancellationToken(object):
    def __init__(self):
        self._cancelled = False

    @property
    def cancelled(self):
        return bool(self._cancelled)

    def cancel(self):
        self._cancelled = True

    def throw_if_cancelled(self):
        if self._cancelled:
            from .errors import LearnedExecutionError

            raise LearnedExecutionError("MANUAL_INTERRUPT", "Execution was cancelled.")


@dataclass
class StateTransitionLog:
    transitions: list = field(default_factory=list)

    def append(self, state, evidence=None):
        item = {
            "index": len(self.transitions),
            "state": state.value if hasattr(state, "value") else str(state),
            "evidence": dict(evidence or {}),
        }
        self.transitions.append(item)
        return item

    def to_jsonl(self):
        return "\n".join(json.dumps(item, sort_keys=True) for item in self.transitions) + ("\n" if self.transitions else "")


@dataclass(frozen=True)
class LearnedSkillResultMetadata:
    policy_id: str
    policy_revision: str
    learned_backend: str
    failure_code: Optional[str]
    progress_stage: str
    monitor_events: Tuple[MonitorEvent, ...]
    fallback_recommended: bool
    fallback_used: bool
    fallback_success: bool
    learned_success: bool
    artifact_path: Optional[str]
    state_transitions: Tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "learned_backend": self.learned_backend,
            "failure_code": self.failure_code,
            "progress_stage": self.progress_stage,
            "monitor_events": [event.to_dict() for event in self.monitor_events],
            "fallback_recommended": bool(self.fallback_recommended),
            "fallback_used": bool(self.fallback_used),
            "fallback_success": bool(self.fallback_success),
            "learned_success": bool(self.learned_success),
            "artifact_path": self.artifact_path,
            "state_transitions": [dict(x) for x in self.state_transitions],
            "evidence": dict(self.evidence),
        }

