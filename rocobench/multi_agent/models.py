"""Shared models for Phase 7 multi-agent scheduling and execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class ScheduleMode(Enum):
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    REJECT = "reject"


class SafetySeverity(Enum):
    INFO = "INFO"
    WARN = "WARN"
    STOP_ONE = "STOP_ONE"
    STOP_ALL = "STOP_ALL"


@dataclass(frozen=True)
class WorkspaceTrajectoryHint:
    zones: Tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"zones": list(self.zones), "reason": self.reason}


@dataclass(frozen=True)
class SkillResourceClaim:
    agent_name: str
    skill_name: str
    exclusive: Tuple[str, ...] = ()
    shared: Tuple[str, ...] = ()
    workspace_trajectory_hint: WorkspaceTrajectoryHint = WorkspaceTrajectoryHint(())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "skill_name": self.skill_name,
            "exclusive": list(self.exclusive),
            "shared": list(self.shared),
            "workspace_trajectory_hint": self.workspace_trajectory_hint.to_dict(),
        }


@dataclass(frozen=True)
class ScheduleGroup:
    index: int
    calls: Tuple[Any, ...]
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "mode": self.mode,
            "calls": [call.to_dict() if hasattr(call, "to_dict") else str(call) for call in self.calls],
        }


@dataclass(frozen=True)
class ScheduleDecision:
    mode: str
    ordered_groups: Tuple[ScheduleGroup, ...]
    resource_claims: Mapping[str, SkillResourceClaim]
    rule_id: str
    reason: str
    conflicts: Tuple[str, ...] = ()
    fallback_to_sequential: bool = False

    @property
    def accepted(self) -> bool:
        return self.mode != ScheduleMode.REJECT.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "ordered_groups": [group.to_dict() for group in self.ordered_groups],
            "resource_claims": {agent: claim.to_dict() for agent, claim in sorted(self.resource_claims.items())},
            "rule_id": self.rule_id,
            "reason": self.reason,
            "conflicts": list(self.conflicts),
            "fallback_to_sequential": bool(self.fallback_to_sequential),
        }


@dataclass(frozen=True)
class AgentActionFragment:
    agent_name: str
    ctrl_idxs: Any
    ctrl_vals: Any
    qpos_idxs: Any
    qpos_target: Any
    eq_active_idxs: Any = None
    eq_active_vals: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def arrays(self) -> Dict[str, np.ndarray]:
        payload = {
            "ctrl_idxs": np.ascontiguousarray(self.ctrl_idxs, dtype=np.int32),
            "ctrl_vals": np.ascontiguousarray(self.ctrl_vals, dtype=np.float32),
            "qpos_idxs": np.ascontiguousarray(self.qpos_idxs, dtype=np.int32),
            "qpos_target": np.ascontiguousarray(self.qpos_target, dtype=np.float32),
        }
        if self.eq_active_idxs is None:
            payload["eq_active_idxs"] = np.ascontiguousarray([], dtype=np.int32)
        else:
            payload["eq_active_idxs"] = np.ascontiguousarray(self.eq_active_idxs, dtype=np.int32)
        if self.eq_active_vals is None:
            payload["eq_active_vals"] = np.ascontiguousarray([], dtype=np.int32)
        else:
            payload["eq_active_vals"] = np.ascontiguousarray(self.eq_active_vals, dtype=np.int32)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        arrays = self.arrays()
        return {
            "agent_name": self.agent_name,
            "ctrl_idxs": arrays["ctrl_idxs"].tolist(),
            "ctrl_vals": arrays["ctrl_vals"].tolist(),
            "qpos_idxs": arrays["qpos_idxs"].tolist(),
            "qpos_target": arrays["qpos_target"].tolist(),
            "eq_active_idxs": arrays["eq_active_idxs"].tolist(),
            "eq_active_vals": arrays["eq_active_vals"].tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class JointStepResult:
    observation: Any
    reward: float
    done: bool
    info: Mapping[str, Any]
    step_index: int
    sim_action: Any
    merged_agent_order: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": float(self.reward),
            "done": bool(self.done),
            "info": dict(self.info or {}),
            "step_index": int(self.step_index),
            "merged_agent_order": list(self.merged_agent_order),
        }


@dataclass(frozen=True)
class SafetyEvent:
    code: str
    severity: str
    message: str
    agent_name: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "agent_name": self.agent_name,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class AgentExecutionOutcome:
    agent_name: str
    success: bool
    status: str
    reason: str = ""
    backend: Optional[str] = None
    num_steps: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": bool(self.success),
            "status": self.status,
            "reason": self.reason,
            "backend": self.backend,
            "num_steps": int(self.num_steps),
            "metadata": dict(self.metadata),
        }
