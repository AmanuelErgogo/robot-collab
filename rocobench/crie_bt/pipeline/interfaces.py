"""Swappable interfaces and typed data models for the CRIE-BT / SARM pipeline.

This module implements the interface contract described in
``docs/crie_next_stage_plan/04_interface_api_spec.md``.  The purpose of these
interfaces is to make robot-robot simulation (Step 1), human-in-simulation
(Step 2), and real-world execution (Step 3) swappable without changing the
controller logic.

Fairness rule (see ``docs/crie_next_stage_plan/03_architecture_spec.md`` section C):
non-oracle planners must never see ``ObservationBundle.oracle_state``.  Use
:func:`planner_safe_observation` before every planner prompt.  Only the coded
simulator monitor and the evaluator may read ``oracle_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

TeamType = Literal["RR", "HR"]
CoordinationMode = Literal["Cent", "Dialog"]
SkillBackend = Literal["RRT", "LearnedSkill"]
MonitorBackend = Literal["VLM-self", "CodedSim", "SARM"]
EnvironmentType = Literal["sim", "real"]
ControllerFamily = Literal["VLM", "CRIE-BT"]


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    STAGE_DONE = "stage_done"
    TASK_DONE = "task_done"
    STUCK = "stuck"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ObservationBundle:
    """Perception products shared across sim and real environments.

    ``oracle_state`` is strictly monitor/evaluation-only.  It is never passed to
    non-oracle planner prompts; :func:`planner_safe_observation` strips it.
    """

    rgb: Dict[str, Any] = field(default_factory=dict)
    depth: Optional[Dict[str, Any]] = None
    proprioception: Dict[str, Any] = field(default_factory=dict)
    public_percepts: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    # Strictly monitor/evaluation only. Never pass this to non-oracle planner prompts.
    oracle_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rgb": _summarize_images(self.rgb),
            "depth": _summarize_images(self.depth) if self.depth is not None else None,
            "proprioception": _jsonable(self.proprioception),
            "public_percepts": _jsonable(self.public_percepts),
            "timestamp": float(self.timestamp),
            "has_oracle_state": self.oracle_state is not None,
        }


@dataclass
class SkillCall:
    agent_id: str
    skill_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    stage_id: Optional[str] = None
    timeout_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "skill_name": str(self.skill_name),
            "args": _jsonable(self.args),
            "stage_id": self.stage_id,
            "timeout_s": self.timeout_s,
        }


@dataclass
class ExecutionFeedback:
    agent_id: str
    skill_call: Optional[SkillCall]
    status: Literal["running", "success", "failure", "timeout"]
    done: Optional[bool] = None
    message: str = ""
    raw_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("success", "failure", "timeout")

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "skill_call": self.skill_call.to_dict() if self.skill_call is not None else None,
            "status": self.status,
            "done": self.done,
            "message": self.message,
            "raw_info": _jsonable(self.raw_info),
        }


@dataclass
class MonitorDecision:
    stage_id: str
    status: StageStatus
    progress_score: float
    should_replan: bool = False
    should_retry: bool = False
    is_stage_done: bool = False
    is_task_done: bool = False
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    monitor_backend: MonitorBackend = "CodedSim"
    privileged: bool = False

    def __post_init__(self) -> None:
        # progress_score is a normalized progress signal; clamp defensively so a
        # noisy learned monitor can never emit an out-of-range score.
        self.progress_score = _clamp01(self.progress_score)
        if not isinstance(self.status, StageStatus):
            self.status = StageStatus(str(self.status))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": str(self.stage_id),
            "status": self.status.value,
            "progress_score": float(self.progress_score),
            "should_replan": bool(self.should_replan),
            "should_retry": bool(self.should_retry),
            "is_stage_done": bool(self.is_stage_done),
            "is_task_done": bool(self.is_task_done),
            "message": self.message,
            "evidence": _jsonable(self.evidence),
            "monitor_backend": self.monitor_backend,
            "privileged": bool(self.privileged),
        }


@dataclass
class HumanInstruction:
    text: str
    target_human_id: str = "human_1"
    expected_stage_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "target_human_id": self.target_human_id,
            "expected_stage_id": self.expected_stage_id,
        }


@dataclass
class HumanResponse:
    text: str
    action: Literal["accept", "reject", "counter_propose", "done", "unknown"] = "unknown"
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "action": self.action, "timestamp": float(self.timestamp)}


@dataclass
class PlanStep:
    step_id: str
    agent_id: str
    skill_call: Optional[SkillCall] = None
    human_instruction: Optional[HumanInstruction] = None
    preconditions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)

    @property
    def is_human_step(self) -> bool:
        return self.human_instruction is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "skill_call": self.skill_call.to_dict() if self.skill_call is not None else None,
            "human_instruction": self.human_instruction.to_dict() if self.human_instruction is not None else None,
            "preconditions": list(self.preconditions),
            "success_conditions": list(self.success_conditions),
        }


@dataclass
class PlanUpdate:
    steps: List[PlanStep]
    dialogue_message: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "dialogue_message": self.dialogue_message,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Fairness enforcement
# ---------------------------------------------------------------------------


def planner_safe_observation(obs: ObservationBundle) -> ObservationBundle:
    """Return a copy of ``obs`` with privileged simulator state removed.

    Only the coded simulator monitor and the evaluator may consume
    ``oracle_state``.  Every non-oracle planner prompt must go through this
    gate so the baseline can never accidentally read privileged simulator
    predicates, object poses, or oracle progress scores.
    """
    return replace(obs, oracle_state=None)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class EnvironmentAdapter:
    """Produces :class:`ObservationBundle` and applies robot/human actions."""

    environment: EnvironmentType = "sim"

    def reset(self, task_id: str, seed: int) -> ObservationBundle:
        raise NotImplementedError

    def observe(self) -> ObservationBundle:
        raise NotImplementedError

    def step_robot(self, agent_id: str, low_level_action: Any) -> ObservationBundle:
        raise NotImplementedError

    def apply_human_action(self, human_id: str, action: Any) -> ObservationBundle:
        raise NotImplementedError

    def get_task_done(self) -> Optional[bool]:
        raise NotImplementedError

    def get_oracle_state(self) -> Dict[str, Any]:
        """Privileged simulator state.  Empty/limited in the real world."""
        return {}


class PlannerInterface:
    """Produces a :class:`PlanUpdate` (skill calls, human instructions, dialogue)."""

    def reset_episode(self, task_id: str, goal: str, capabilities: Dict[str, Any]) -> None:
        pass

    def propose_next(
        self,
        observation: ObservationBundle,
        goal: str,
        history: List[Dict[str, Any]],
        dialogue: List[Dict[str, str]],
        feedback: Optional[ExecutionFeedback] = None,
        monitor_decision: Optional[MonitorDecision] = None,
    ) -> PlanUpdate:
        raise NotImplementedError


class SkillExecutorInterface:
    backend_name: SkillBackend = "RRT"

    def reset(self, env: EnvironmentAdapter) -> None:
        pass

    def start(self, skill_call: SkillCall, observation: ObservationBundle) -> None:
        raise NotImplementedError

    def step(self, observation: ObservationBundle) -> ExecutionFeedback:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class ProgressMonitorInterface:
    backend_name: MonitorBackend = "CodedSim"
    privileged: bool = True

    def reset_stage(
        self,
        stage_id: str,
        skill_call: Optional[SkillCall],
        observation: ObservationBundle,
    ) -> None:
        pass

    def update(
        self,
        observation: ObservationBundle,
        feedback: Optional[ExecutionFeedback],
        history: List[Dict[str, Any]],
    ) -> MonitorDecision:
        raise NotImplementedError


class CommunicationInterface:
    def send_to_human(self, instruction: HumanInstruction) -> None:
        raise NotImplementedError

    def read_human_response(self, timeout_s: Optional[float] = None) -> HumanResponse:
        raise NotImplementedError

    def send_robot_dialogue(self, speaker_id: str, text: str) -> None:
        raise NotImplementedError


class CollaborationController:
    def run_episode(
        self,
        condition_name: str,
        env: EnvironmentAdapter,
        task_id: str,
        goal: str,
        max_steps: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Small serialization helpers
# ---------------------------------------------------------------------------


def _clamp01(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score:  # NaN
        return 0.0
    return max(0.0, min(1.0, score))


def _summarize_images(images: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace raw image arrays with compact shape descriptors for logging."""
    if not images:
        return {}
    summary: Dict[str, Any] = {}
    for key, value in images.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            summary[str(key)] = {"shape": list(shape)}
        else:
            summary[str(key)] = _jsonable(value)
    return summary


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy always available in this repo
        np = None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "CollaborationController",
    "CommunicationInterface",
    "CoordinationMode",
    "ControllerFamily",
    "EnvironmentAdapter",
    "EnvironmentType",
    "ExecutionFeedback",
    "HumanInstruction",
    "HumanResponse",
    "MonitorBackend",
    "MonitorDecision",
    "ObservationBundle",
    "PlanStep",
    "PlanUpdate",
    "PlannerInterface",
    "ProgressMonitorInterface",
    "SkillBackend",
    "SkillCall",
    "SkillExecutorInterface",
    "StageStatus",
    "TeamType",
    "planner_safe_observation",
]
