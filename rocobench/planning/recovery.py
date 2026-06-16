"""Failure fingerprinting and deterministic recovery decisions."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rocobench.skills.models import SkillCall, SkillExecutionResult

from .executor_router import BACKEND_RRT, BACKEND_UNSUPPORTED, ExecutorDecision
from .replanning_policy import RecoveryAction, ReplanningPolicy
from .retry_budget import RetryBudget
from .state_summary import StateSummary


def failure_code_from_result(result: SkillExecutionResult) -> str:
    metadata = dict(result.metadata or {})
    code = metadata.get("failure_code")
    if code:
        return str(code).upper()
    if result.status is not None:
        return str(result.status.value if hasattr(result.status, "value") else result.status).upper()
    return "UNKNOWN"


def retry_key_for_call(call: SkillCall) -> str:
    obj = call.arguments.get("object", "")
    target = call.arguments.get("container", "")
    return "{}:{}:{}:{}".format(call.agent_name, call.skill_name, obj, target)


def fingerprint_for_failure(call: SkillCall, failure_code: str, state_summary: StateSummary) -> str:
    payload = {
        "agent": call.agent_name,
        "skill": call.skill_name,
        "object": call.arguments.get("object", ""),
        "target": call.arguments.get("container", ""),
        "failure_code": str(failure_code or "").upper(),
        "state": state_summary.quantized_state,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class FailureLoopDetector(object):
    def __init__(self, threshold: int = 2) -> None:
        self.threshold = int(threshold)
        self.counts = {}  # type: Dict[str, int]

    def record(self, call: SkillCall, failure_code: str, state_summary: StateSummary) -> Dict[str, Any]:
        fingerprint = fingerprint_for_failure(call, failure_code, state_summary)
        count = self.counts.get(fingerprint, 0) + 1
        self.counts[fingerprint] = count
        return {
            "fingerprint": fingerprint,
            "count": count,
            "loop_detected": bool(self.threshold >= 0 and count > self.threshold),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "counts": dict(self.counts)}


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    rollback: bool
    reason: str
    failure_code: str
    fingerprint: Optional[str] = None
    fingerprint_count: int = 0
    fallback_backend: str = BACKEND_UNSUPPORTED
    budget_exhausted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "rollback": bool(self.rollback),
            "reason": self.reason,
            "failure_code": self.failure_code,
            "fingerprint": self.fingerprint,
            "fingerprint_count": int(self.fingerprint_count),
            "fallback_backend": self.fallback_backend,
            "budget_exhausted": bool(self.budget_exhausted),
        }


class RecoveryEngine(object):
    def __init__(
        self,
        policy: ReplanningPolicy,
        budget: RetryBudget,
        loop_detector: Optional[FailureLoopDetector] = None,
    ) -> None:
        self.policy = policy
        self.budget = budget
        self.loop_detector = loop_detector or FailureLoopDetector(policy.max_repeated_failures)

    def decide(
        self,
        result: SkillExecutionResult,
        call: SkillCall,
        state_summary: StateSummary,
        executor_decision: ExecutorDecision,
        backend: str,
    ) -> RecoveryDecision:
        failure_code = failure_code_from_result(result)
        loop = self.loop_detector.record(call, failure_code, state_summary)
        action = self.policy.action_for(failure_code)
        reason = "Mapped {} to {}.".format(failure_code, action.value)
        budget_exhausted = False

        if backend == "learned":
            if not self.budget.consume_learned_failure():
                budget_exhausted = True
                action = RecoveryAction.USE_RRT_FALLBACK if executor_decision.fallback_backend != BACKEND_UNSUPPORTED else RecoveryAction.ABORT_TASK
                reason = "Learned failure budget exhausted."

        if loop["loop_detected"]:
            if executor_decision.fallback_backend != BACKEND_UNSUPPORTED:
                action = RecoveryAction.USE_RRT_FALLBACK
                reason = "Repeated failure fingerprint exceeded threshold."
            else:
                action = RecoveryAction.ABORT_TASK
                budget_exhausted = True
                reason = "Repeated failure fingerprint exceeded threshold and no fallback is configured."

        if action == RecoveryAction.RETRY_LEARNED:
            key = retry_key_for_call(call)
            if not self.budget.consume_retry(key):
                budget_exhausted = True
                if executor_decision.fallback_backend != BACKEND_UNSUPPORTED:
                    action = RecoveryAction.USE_RRT_FALLBACK
                    reason = "Retry budget exhausted; using configured fallback."
                else:
                    action = RecoveryAction.REPLAN_FROM_CURRENT_STATE
                    reason = "Retry budget exhausted; requesting replan."

        if action == RecoveryAction.USE_RRT_FALLBACK:
            if executor_decision.fallback_backend in (BACKEND_UNSUPPORTED, ""):
                action = RecoveryAction.ABORT_TASK
                budget_exhausted = True
                reason = "Fallback requested but no fallback backend is configured."
            elif executor_decision.fallback_backend != BACKEND_RRT:
                action = RecoveryAction.ABORT_TASK
                budget_exhausted = True
                reason = "Unsupported fallback backend: {}.".format(executor_decision.fallback_backend)
            elif not self.budget.consume_fallback():
                action = RecoveryAction.ABORT_TASK
                budget_exhausted = True
                reason = "Fallback budget exhausted."

        rollback = self.policy.rollback_required(failure_code, action)
        return RecoveryDecision(
            action=action,
            rollback=rollback,
            reason=reason,
            failure_code=failure_code,
            fingerprint=loop["fingerprint"],
            fingerprint_count=int(loop["count"]),
            fallback_backend=executor_decision.fallback_backend,
            budget_exhausted=budget_exhausted,
        )
