"""Explicit learned-skill fallback decisions."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class FallbackConfig:
    mode: str = "return_request"
    backend: str = "rrt"
    allowed_reasons: Tuple[str, ...] = ("POLICY_INFERENCE_FAILURE", "NO_PROGRESS")
    max_fallbacks: int = 1

    def __post_init__(self):
        if self.mode not in ("disabled", "return_request", "automatic"):
            raise ValueError("fallback mode must be disabled, return_request, or automatic")
        object.__setattr__(self, "allowed_reasons", tuple(str(x) for x in self.allowed_reasons))
        if self.max_fallbacks < 0:
            raise ValueError("max_fallbacks cannot be negative")

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        return cls(
            mode=str(data.get("mode", "return_request")),
            backend=str(data.get("backend", "rrt")),
            allowed_reasons=tuple(str(x) for x in data.get("allowed_reasons", ("POLICY_INFERENCE_FAILURE", "NO_PROGRESS"))),
            max_fallbacks=int(data.get("max_fallbacks", 1)),
        )

    def to_dict(self):
        return {
            "mode": self.mode,
            "backend": self.backend,
            "allowed_reasons": list(self.allowed_reasons),
            "max_fallbacks": int(self.max_fallbacks),
        }


@dataclass(frozen=True)
class FallbackDecision:
    recommended: bool
    used: bool = False
    success: bool = False
    backend: str = "rrt"
    reason: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "recommended": bool(self.recommended),
            "used": bool(self.used),
            "success": bool(self.success),
            "backend": self.backend,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


class FallbackController(object):
    def __init__(self, config=None, fallback_executor=None):
        self.config = config or FallbackConfig()
        self.fallback_executor = fallback_executor
        self.used_count = 0

    def decide(self, failure_code, plan=None, obs=None, artifact_dir=None):
        if self.config.mode == "disabled" or failure_code not in self.config.allowed_reasons:
            return FallbackDecision(False, backend=self.config.backend, reason=failure_code)
        if self.config.mode == "return_request":
            return FallbackDecision(True, backend=self.config.backend, reason=failure_code)
        if self.used_count >= self.config.max_fallbacks or self.fallback_executor is None:
            return FallbackDecision(True, backend=self.config.backend, reason=failure_code, evidence={"automatic_blocked": True})
        self.used_count += 1
        result = self.fallback_executor.execute(plan, obs, artifact_dir=artifact_dir)
        return FallbackDecision(
            recommended=True,
            used=True,
            success=bool(result.success),
            backend=self.config.backend,
            reason=failure_code,
            evidence={"fallback_result": result.to_dict()},
        )

