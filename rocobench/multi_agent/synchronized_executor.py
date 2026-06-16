"""Sequential and synchronized execution helpers for Phase 7."""

import time
from typing import Any, Dict, Mapping, Optional, Sequence

from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus, SkillPlan
from rocobench.skills.pack_grocery import WAIT

from .cancellation import MultiAgentCancellationToken
from .combined_result import CombinedExecutionResult
from .joint_stepper import CentralJointStepper, JointActionMergeError
from .metrics import MultiAgentMetrics
from .models import AgentExecutionOutcome, SafetyEvent, SafetySeverity, ScheduleDecision, ScheduleMode
from .safety_monitor import CentralSafetyMonitor


class SequentialMultiAgentExecutor(object):
    """Execute scheduled active skill calls one at a time."""

    def __init__(self, agent_order: Sequence[str], executors: Mapping[str, Any]) -> None:
        self.agent_order = tuple(agent_order)
        self.executors = dict(executors)

    def execute(self, schedule: ScheduleDecision, original_plan: SkillPlan, obs: Any, artifact_dir: Optional[str] = None) -> CombinedExecutionResult:
        metrics = MultiAgentMetrics()
        metrics.record_schedule(schedule.mode, schedule.accepted)
        outcomes = {}
        if schedule.mode == ScheduleMode.REJECT.value:
            return CombinedExecutionResult(False, schedule, outcomes, "REJECTED", schedule.reason, metrics=metrics)

        current_obs = obs
        for group in schedule.ordered_groups:
            for call in group.calls:
                executor = self.executors.get(call.agent_name)
                if executor is None:
                    outcomes[call.agent_name] = AgentExecutionOutcome(call.agent_name, False, "NO_EXECUTOR", "No executor configured.")
                    return CombinedExecutionResult(False, schedule, outcomes, "FAILED", "No executor configured.", metrics=metrics)
                if hasattr(executor, "reset_agent"):
                    executor.reset_agent(call.agent_name)
                single_plan = self._single_active_plan(original_plan, call)
                result = executor.execute(single_plan, current_obs, artifact_dir=artifact_dir)
                outcomes[call.agent_name] = AgentExecutionOutcome(
                    agent_name=call.agent_name,
                    success=bool(result.success),
                    status=result.status.value if hasattr(result.status, "value") else str(result.status),
                    reason=result.reason,
                    backend=getattr(executor, "backend_name", None),
                    num_steps=result.num_sim_steps,
                    metadata=result.metadata,
                )
                metrics.summed_agent_steps += int(result.num_sim_steps)
                metrics.makespan_steps += int(result.num_sim_steps)
                if not result.success:
                    return CombinedExecutionResult(False, schedule, outcomes, "FAILED", result.reason, metrics=metrics)
                current_obs = getattr(result, "observation", current_obs)
        return CombinedExecutionResult(True, schedule, outcomes, "SUCCEEDED", metrics=metrics)

    def _single_active_plan(self, original_plan: SkillPlan, active_call: Any) -> SkillPlan:
        calls = []
        for call in original_plan.calls:
            if call.agent_name == active_call.agent_name:
                calls.append(active_call)
            else:
                calls.append(type(call)(call.agent_name, WAIT, {}, "WAIT()"))
        return SkillPlan(calls, original_plan.parsed_proposal)


class SynchronizedExecutor(object):
    """Run per-agent action providers through one central joint stepper."""

    def __init__(
        self,
        env: Any,
        agent_order: Sequence[str],
        action_providers: Mapping[str, Any],
        safety_monitor: Optional[CentralSafetyMonitor] = None,
        cancellation_token: Optional[MultiAgentCancellationToken] = None,
        sim_action_factory: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.agent_order = tuple(agent_order)
        self.action_providers = dict(action_providers)
        self.safety_monitor = safety_monitor or CentralSafetyMonitor()
        self.cancellation_token = cancellation_token or MultiAgentCancellationToken()
        self.joint_stepper = CentralJointStepper(env, agent_order, sim_action_factory=sim_action_factory)

    def run(self, schedule: ScheduleDecision, initial_obs: Any, max_joint_steps: int = 1) -> CombinedExecutionResult:
        metrics = MultiAgentMetrics()
        metrics.record_schedule(schedule.mode, schedule.accepted)
        outcomes = {
            agent: AgentExecutionOutcome(agent, False, "PENDING")
            for agent in self.agent_order
            if agent in self.action_providers
        }
        events = []
        if schedule.mode == ScheduleMode.REJECT.value:
            return CombinedExecutionResult(False, schedule, outcomes, "REJECTED", schedule.reason, metrics=metrics)
        if schedule.mode != ScheduleMode.CONCURRENT.value:
            return CombinedExecutionResult(False, schedule, outcomes, "NOT_CONCURRENT", "Synchronized executor requires concurrent schedule.", metrics=metrics)

        obs = initial_obs
        for provider in self.action_providers.values():
            if hasattr(provider, "reset"):
                provider.reset()
        try:
            for _ in range(int(max_joint_steps)):
                self.cancellation_token.throw_if_cancelled()
                fragments = {}
                latencies = {}
                for agent in self.agent_order:
                    provider = self.action_providers.get(agent)
                    if provider is None:
                        continue
                    start = time.perf_counter()
                    fragments[agent] = provider.next_action(obs, agent)
                    latencies[agent] = (time.perf_counter() - start) * 1000.0
                step_events = self.safety_monitor.check_actions(fragments, latencies)
                events.extend(step_events)
                if self.safety_monitor.should_stop_all(step_events):
                    self.cancellation_token.cancel("central safety stop")
                    metrics.stop_all_events += 1
                    break
                try:
                    step_result = self.joint_stepper.step(fragments)
                except JointActionMergeError as exc:
                    events.append(SafetyEvent("ACTION_MERGE_FAILED", SafetySeverity.STOP_ALL.value, str(exc)))
                    self.cancellation_token.cancel(str(exc))
                    metrics.stop_all_events += 1
                    break
                metrics.central_steps += 1
                metrics.makespan_steps += 1
                metrics.summed_agent_steps += len(fragments)
                info_events = self.safety_monitor.check_step_info(step_result.info)
                events.extend(info_events)
                if self.safety_monitor.should_stop_all(info_events):
                    self.cancellation_token.cancel("central safety stop")
                    metrics.stop_all_events += 1
                    break
                obs = step_result.observation
                if step_result.done:
                    break
        except RuntimeError as exc:
            events.append(SafetyEvent("CANCELLED", SafetySeverity.STOP_ALL.value, str(exc)))
            metrics.stop_all_events += 1

        success = not self.cancellation_token.cancelled and not self.safety_monitor.should_stop_all(events)
        final_status = "SUCCEEDED" if success else "STOP_ALL"
        outcomes = {
            agent: AgentExecutionOutcome(agent, success, final_status, num_steps=metrics.central_steps)
            for agent in self.action_providers
        }
        return CombinedExecutionResult(
            success,
            schedule,
            outcomes,
            final_status,
            self.cancellation_token.reason,
            safety_events=tuple(events),
            fallback_to_sequential_recommended=not success,
            metrics=metrics,
        )
