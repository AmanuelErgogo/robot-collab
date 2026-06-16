"""Budget accounting for bounded Phase 6 replanning."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .replanning_policy import ReplanningPolicy


@dataclass
class RetryBudget:
    policy: ReplanningPolicy
    plan_rounds_used: int = 0
    learned_failures: int = 0
    fallbacks_used: int = 0
    invalid_plans: int = 0
    retries_by_key: Dict[str, int] = field(default_factory=dict)

    def can_start_plan_round(self) -> bool:
        return self.plan_rounds_used < self.policy.max_plan_rounds

    def start_plan_round(self) -> None:
        if not self.can_start_plan_round():
            raise RuntimeError("Plan round budget exhausted")
        self.plan_rounds_used += 1

    def remaining_plan_rounds(self) -> int:
        return max(0, self.policy.max_plan_rounds - self.plan_rounds_used)

    def record_invalid_plan(self) -> None:
        self.invalid_plans += 1

    def can_retry(self, retry_key: str) -> bool:
        if not self.policy.allow_same_plan_retry:
            return False
        return self.retries_by_key.get(str(retry_key), 0) < self.policy.max_retries_per_skill

    def consume_retry(self, retry_key: str) -> bool:
        key = str(retry_key)
        if not self.can_retry(key):
            return False
        self.retries_by_key[key] = self.retries_by_key.get(key, 0) + 1
        return True

    def remaining_retries(self, retry_key: str) -> int:
        used = self.retries_by_key.get(str(retry_key), 0)
        return max(0, self.policy.max_retries_per_skill - used)

    def can_record_learned_failure(self) -> bool:
        return self.learned_failures < self.policy.max_learned_failures

    def consume_learned_failure(self) -> bool:
        if not self.can_record_learned_failure():
            return False
        self.learned_failures += 1
        return True

    def remaining_learned_failures(self) -> int:
        return max(0, self.policy.max_learned_failures - self.learned_failures)

    def can_use_fallback(self) -> bool:
        return self.fallbacks_used < self.policy.max_fallbacks

    def consume_fallback(self) -> bool:
        if not self.can_use_fallback():
            return False
        self.fallbacks_used += 1
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_rounds_used": int(self.plan_rounds_used),
            "max_plan_rounds": int(self.policy.max_plan_rounds),
            "remaining_plan_rounds": int(self.remaining_plan_rounds()),
            "learned_failures": int(self.learned_failures),
            "max_learned_failures": int(self.policy.max_learned_failures),
            "remaining_learned_failures": int(self.remaining_learned_failures()),
            "fallbacks_used": int(self.fallbacks_used),
            "max_fallbacks": int(self.policy.max_fallbacks),
            "invalid_plans": int(self.invalid_plans),
            "retries_by_key": dict(self.retries_by_key),
        }

    @classmethod
    def from_dict(cls, policy: ReplanningPolicy, data: Mapping[str, Any]) -> "RetryBudget":
        data = dict(data or {})
        return cls(
            policy=policy,
            plan_rounds_used=int(data.get("plan_rounds_used", 0)),
            learned_failures=int(data.get("learned_failures", 0)),
            fallbacks_used=int(data.get("fallbacks_used", 0)),
            invalid_plans=int(data.get("invalid_plans", 0)),
            retries_by_key={str(k): int(v) for k, v in dict(data.get("retries_by_key", {})).items()},
        )
