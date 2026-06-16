"""Conservative deterministic scheduler for Phase 7."""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from rocobench.skills.pack_grocery import WAIT

from .compatibility import CompatibilityMatrix
from .models import ScheduleDecision, ScheduleGroup, ScheduleMode
from .resources import ResourceClaimDeriver, exclusive_resource_conflicts, workspace_conflicts


@dataclass(frozen=True)
class SchedulerConfig:
    agent_order: Tuple[str, ...] = ("Alice", "Bob")
    enable_concurrency: bool = False
    max_concurrent_agents: int = 2
    reject_conflicts: bool = True


class MultiAgentScheduler(object):
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        claim_deriver: Optional[ResourceClaimDeriver] = None,
        compatibility: Optional[CompatibilityMatrix] = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.claim_deriver = claim_deriver or ResourceClaimDeriver()
        self.compatibility = compatibility or CompatibilityMatrix()

    def schedule(self, plan: Any, env: Any = None, obs: Any = None) -> ScheduleDecision:
        claims = self.claim_deriver.derive_for_plan(plan, env=env, obs=obs)
        active = [call for call in plan.calls if call.skill_name != WAIT]
        ordered = sorted(active, key=lambda call: self._agent_index(call.agent_name))
        active_claims = [claims[call.agent_name] for call in ordered]

        resource_conflicts = list(exclusive_resource_conflicts(active_claims))
        workspace = list(workspace_conflicts(active_claims))
        pair_decision = self.compatibility.evaluate_group(ordered, claims) if len(ordered) > 1 else None
        conflicts = resource_conflicts + workspace
        if pair_decision is not None and not pair_decision.compatible:
            conflicts.append(pair_decision.reason)

        if conflicts and self.config.reject_conflicts:
            return ScheduleDecision(
                mode=ScheduleMode.REJECT.value,
                ordered_groups=(),
                resource_claims=claims,
                rule_id=pair_decision.rule_id if pair_decision is not None else "phase7.conflict",
                reason="Conflicting resource/workspace claims.",
                conflicts=tuple(conflicts),
            )

        if not ordered:
            return ScheduleDecision(
                mode=ScheduleMode.REJECT.value,
                ordered_groups=(),
                resource_claims=claims,
                rule_id="phase7.no_active_skill",
                reason="At least one active skill is required.",
                conflicts=("all agents WAIT",),
            )

        if self.config.enable_concurrency and len(ordered) > 1 and len(ordered) <= self.config.max_concurrent_agents and not conflicts:
            group = ScheduleGroup(0, tuple(ordered), ScheduleMode.CONCURRENT.value)
            return ScheduleDecision(
                mode=ScheduleMode.CONCURRENT.value,
                ordered_groups=(group,),
                resource_claims=claims,
                rule_id=pair_decision.rule_id if pair_decision is not None else "phase7.single",
                reason=pair_decision.reason if pair_decision is not None else "Single active call.",
                conflicts=(),
            )

        groups = tuple(
            ScheduleGroup(index, (call,), ScheduleMode.SEQUENTIAL.value)
            for index, call in enumerate(ordered)
        )
        return ScheduleDecision(
            mode=ScheduleMode.SEQUENTIAL.value,
            ordered_groups=groups,
            resource_claims=claims,
            rule_id="phase7.sequential.default",
            reason="Concurrency disabled or uncertain; using deterministic sequential order.",
            conflicts=tuple(conflicts),
            fallback_to_sequential=bool(conflicts),
        )

    def _agent_index(self, agent_name: str) -> int:
        try:
            return self.config.agent_order.index(agent_name)
        except ValueError:
            return len(self.config.agent_order)
