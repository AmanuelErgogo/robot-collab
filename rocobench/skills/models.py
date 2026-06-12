import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple


_SKILL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def normalize_skill_name(name: str) -> str:
    """Return the canonical uppercase snake-case form used by skill specs."""
    normalized = re.sub(r"[\s\-]+", "_", str(name).strip()).upper()
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


@dataclass(frozen=True)
class SkillSpec:
    """Static declaration for one planner-visible skill."""

    name: str
    description: str
    required_arguments: Tuple[str, ...]
    optional_arguments: Tuple[str, ...] = ()
    supported_agents: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    resource_arguments: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        canonical = normalize_skill_name(self.name)
        if not _SKILL_NAME_RE.match(canonical):
            raise ValueError("Skill name must be uppercase snake case")
        object.__setattr__(self, "name", canonical)

        required = tuple(self.required_arguments)
        optional = tuple(self.optional_arguments)
        if len(set(required)) != len(required):
            raise ValueError("Duplicate required skill arguments")
        if len(set(optional)) != len(optional):
            raise ValueError("Duplicate optional skill arguments")
        overlap = set(required).intersection(optional)
        if overlap:
            raise ValueError("Arguments cannot be both required and optional: {}".format(sorted(overlap)))

        aliases = tuple(normalize_skill_name(alias) for alias in self.aliases)
        if len(set(aliases)) != len(aliases):
            raise ValueError("Duplicate skill aliases")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "required_arguments", required)
        object.__setattr__(self, "optional_arguments", optional)
        object.__setattr__(self, "supported_agents", tuple(self.supported_agents))
        object.__setattr__(self, "resource_arguments", tuple(self.resource_arguments))

    @property
    def argument_names(self) -> Tuple[str, ...]:
        """Return required and optional argument names in deterministic prompt order."""
        return self.required_arguments + self.optional_arguments


@dataclass(frozen=True)
class SkillCall:
    """One agent's canonical skill invocation."""

    agent_name: str
    skill_name: str
    arguments: Mapping[str, str]
    raw_action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_name", normalize_skill_name(self.skill_name))
        copied = {str(key): str(value) for key, value in self.arguments.items()}
        object.__setattr__(self, "arguments", MappingProxyType(copied))

    def __str__(self) -> str:
        args = ", ".join("{}={}".format(key, value) for key, value in self.arguments.items())
        return "{}({})".format(self.skill_name, args)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""
        return {
            "agent_name": self.agent_name,
            "skill_name": self.skill_name,
            "arguments": dict(self.arguments),
            "raw_action": self.raw_action,
        }


@dataclass
class PreparedSkillExecution:
    """Backend-specific executable representation attached after preparation."""

    backend_name: str
    source_plan_id: str
    compiled_plans: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe metadata without serializing backend path objects."""
        return {
            "backend_name": self.backend_name,
            "source_plan_id": self.source_plan_id,
            "compiled_plan_count": len(self.compiled_plans),
            "metadata": dict(self.metadata),
        }


@dataclass
class SkillPlan:
    """Coordinated set of one skill call per configured agent."""

    calls: List[SkillCall]
    parsed_proposal: str
    plan_id: str = ""
    prepared_execution: Optional[PreparedSkillExecution] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.calls = list(self.calls)
        if not self.plan_id:
            self.plan_id = self._compute_plan_id()

    def _compute_plan_id(self) -> str:
        payload = [call.to_dict() for call in self.calls]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def get_action_desp(self) -> str:
        """Return skill-level plan text suitable for history and logs."""
        return "\n".join(
            "NAME {} ACTION {}".format(call.agent_name, str(call)) for call in self.calls
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable plan data, excluding runtime preparation."""
        return {
            "plan_id": self.plan_id,
            "parsed_proposal": self.parsed_proposal,
            "calls": [call.to_dict() for call in self.calls],
        }


@dataclass(frozen=True)
class ValidationIssue:
    """Stable validation issue suitable for LLM feedback."""

    code: str
    message: str
    agent_name: Optional[str] = None
    retryable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "agent_name": self.agent_name,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:
        prefix = "[{}]".format(self.code)
        if self.agent_name:
            return "{} {}: {}".format(prefix, self.agent_name, self.message)
        return "{} {}".format(prefix, self.message)


@dataclass(frozen=True)
class SkillValidationResult:
    """Validation aggregate returned by registries and semantic validators."""

    valid: bool
    issues: Tuple[ValidationIssue, ...] = ()

    @classmethod
    def ok(cls) -> "SkillValidationResult":
        return cls(valid=True, issues=())

    @classmethod
    def invalid(cls, issues: List[ValidationIssue]) -> "SkillValidationResult":
        return cls(valid=False, issues=tuple(issues))

    def to_feedback(self) -> str:
        return "\n".join(str(issue) for issue in self.issues)


class SkillExecutionStatus(Enum):
    """Structured status values for skill execution backends."""

    SUCCESS = "success"
    INVALID_PLAN = "invalid_plan"
    NOT_PREPARED = "not_prepared"
    MOTION_PLANNING_FAILED = "motion_planning_failed"
    EXECUTION_FAILED = "execution_failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"


@dataclass
class SkillExecutionResult:
    """Result returned by a skill executor."""

    success: bool
    status: SkillExecutionStatus
    reason: str
    num_sim_steps: int
    reward: float
    done: bool
    info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "reason": self.reason,
            "num_sim_steps": self.num_sim_steps,
            "reward": self.reward,
            "done": self.done,
            "info": dict(self.info),
            "metadata": dict(self.metadata),
        }
