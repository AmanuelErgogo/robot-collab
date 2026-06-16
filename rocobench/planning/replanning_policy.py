"""Deterministic recovery policy for failed Phase 6 skill executions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class RecoveryAction(Enum):
    RETRY_LEARNED = "RETRY_LEARNED"
    REPLAN_FROM_CURRENT_STATE = "REPLAN_FROM_CURRENT_STATE"
    ROLLBACK_AND_REPLAN = "ROLLBACK_AND_REPLAN"
    USE_RRT_FALLBACK = "USE_RRT_FALLBACK"
    ABORT_TASK = "ABORT_TASK"


DEFAULT_FAILURE_ACTIONS = {
    "MISSED_GRASP": RecoveryAction.RETRY_LEARNED.value,
    "SLIPPAGE": RecoveryAction.REPLAN_FROM_CURRENT_STATE.value,
    "TARGET_OCCUPIED": RecoveryAction.REPLAN_FROM_CURRENT_STATE.value,
    "OBJECT_LOST": RecoveryAction.ROLLBACK_AND_REPLAN.value,
    "POLICY_INFERENCE_FAILURE": RecoveryAction.USE_RRT_FALLBACK.value,
    "BRIDGE_FAILURE": RecoveryAction.USE_RRT_FALLBACK.value,
    "ACTION_OUT_OF_BOUNDS": RecoveryAction.USE_RRT_FALLBACK.value,
    "NONFINITE_ACTION": RecoveryAction.USE_RRT_FALLBACK.value,
    "NO_PROGRESS": RecoveryAction.USE_RRT_FALLBACK.value,
    "TIMEOUT": RecoveryAction.USE_RRT_FALLBACK.value,
    "MOTION_PLANNING_FAILED": RecoveryAction.REPLAN_FROM_CURRENT_STATE.value,
    "INVALID_PLAN": RecoveryAction.REPLAN_FROM_CURRENT_STATE.value,
}

DEFAULT_ROLLBACK_RULES = {
    "POLICY_INFERENCE_FAILURE": True,
    "BRIDGE_FAILURE": True,
    "ACTION_OUT_OF_BOUNDS": True,
    "NONFINITE_ACTION": True,
    "OBJECT_LOST": True,
    "TIMEOUT": True,
    "NO_PROGRESS": True,
    "MOTION_PLANNING_FAILED": False,
    "MISSED_GRASP": False,
    "SLIPPAGE": False,
    "TARGET_OCCUPIED": False,
}


@dataclass(frozen=True)
class ReplanningPolicy:
    max_plan_rounds: int = 4
    max_retries_per_skill: int = 1
    max_learned_failures: int = 3
    allow_same_plan_retry: bool = True
    fallback_rules: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FAILURE_ACTIONS))
    rollback_rules: Mapping[str, bool] = field(default_factory=lambda: dict(DEFAULT_ROLLBACK_RULES))
    max_repeated_failures: int = 2
    max_fallbacks: int = 1

    def __post_init__(self) -> None:
        if self.max_plan_rounds <= 0:
            raise ValueError("max_plan_rounds must be positive")
        if self.max_retries_per_skill < 0:
            raise ValueError("max_retries_per_skill cannot be negative")
        if self.max_learned_failures < 0:
            raise ValueError("max_learned_failures cannot be negative")
        if self.max_repeated_failures < 0:
            raise ValueError("max_repeated_failures cannot be negative")
        if self.max_fallbacks < 0:
            raise ValueError("max_fallbacks cannot be negative")
        normalized = {}
        for key, value in dict(self.fallback_rules).items():
            action = value.value if hasattr(value, "value") else str(value)
            if action not in [item.value for item in RecoveryAction]:
                raise ValueError("Unknown recovery action for {}: {}".format(key, value))
            normalized[str(key).upper()] = action
        object.__setattr__(self, "fallback_rules", normalized)
        object.__setattr__(self, "rollback_rules", {str(k).upper(): bool(v) for k, v in dict(self.rollback_rules).items()})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplanningPolicy":
        data = dict(data or {})
        fallback_rules = dict(DEFAULT_FAILURE_ACTIONS)
        fallback_rules.update({str(k).upper(): str(v) for k, v in dict(data.get("fallback_rules", {})).items()})
        rollback_rules = dict(DEFAULT_ROLLBACK_RULES)
        rollback_rules.update({str(k).upper(): bool(v) for k, v in dict(data.get("rollback_rules", {})).items()})
        return cls(
            max_plan_rounds=int(data.get("max_plan_rounds", 4)),
            max_retries_per_skill=int(data.get("max_retries_per_skill", 1)),
            max_learned_failures=int(data.get("max_learned_failures", 3)),
            allow_same_plan_retry=bool(data.get("allow_same_plan_retry", True)),
            fallback_rules=fallback_rules,
            rollback_rules=rollback_rules,
            max_repeated_failures=int(data.get("max_repeated_failures", 2)),
            max_fallbacks=int(data.get("max_fallbacks", 1)),
        )

    def action_for(self, failure_code: str) -> RecoveryAction:
        code = str(failure_code or "").upper()
        action = self.fallback_rules.get(code, RecoveryAction.ABORT_TASK.value)
        return RecoveryAction(action)

    def rollback_required(self, failure_code: str, action: RecoveryAction) -> bool:
        code = str(failure_code or "").upper()
        if action == RecoveryAction.ROLLBACK_AND_REPLAN:
            return True
        return bool(self.rollback_rules.get(code, False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_plan_rounds": int(self.max_plan_rounds),
            "max_retries_per_skill": int(self.max_retries_per_skill),
            "max_learned_failures": int(self.max_learned_failures),
            "allow_same_plan_retry": bool(self.allow_same_plan_retry),
            "fallback_rules": dict(self.fallback_rules),
            "rollback_rules": dict(self.rollback_rules),
            "max_repeated_failures": int(self.max_repeated_failures),
            "max_fallbacks": int(self.max_fallbacks),
        }
