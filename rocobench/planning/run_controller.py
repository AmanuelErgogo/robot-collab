"""Sequential Phase 6 planner controller.

The controller wires existing parser/validator/compiler/executor interfaces into
bounded, failure-aware planning. It is intentionally dependency-injected so unit
tests can run with canned planner responses and fake executors.
"""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rocobench.skills.compiler import SkillCompilationError, SkillCompiler
from rocobench.skills.models import SkillExecutionResult, SkillExecutionStatus, SkillPlan, SkillValidationResult, ValidationIssue
from rocobench.skills.validation import SkillPlanValidator

from .executor_router import BACKEND_LEARNED, BACKEND_RRT, BACKEND_UNSUPPORTED, ExecutorDecision, SkillExecutorRouter
from .feedback_renderer import GroundedFeedbackRenderer
from .planner_events import PlannerEventLog, PlannerEventType
from .recovery import RecoveryEngine, failure_code_from_result
from .replanning_policy import RecoveryAction, ReplanningPolicy
from .retry_budget import RetryBudget
from .state_summary import StateSummary, build_state_summary

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class PlannerMetrics:
    task_success: bool = False
    learned_attempts: int = 0
    learned_successes: int = 0
    rrt_attempts: int = 0
    rrt_successes: int = 0
    fallback_attempts: int = 0
    fallback_successes: int = 0
    replans: int = 0
    invalid_plans: int = 0
    rollbacks: int = 0
    learned_failures: int = 0
    failure_distribution: Dict[str, int] = field(default_factory=dict)
    total_sim_steps: int = 0

    def record_attempt(self, backend: str, result: SkillExecutionResult, fallback: bool = False) -> None:
        if backend == BACKEND_LEARNED:
            self.learned_attempts += 1
            if result.success:
                self.learned_successes += 1
            else:
                self.learned_failures += 1
        if backend == BACKEND_RRT:
            self.rrt_attempts += 1
            if result.success:
                self.rrt_successes += 1
        if fallback:
            self.fallback_attempts += 1
            if result.success:
                self.fallback_successes += 1
        if not result.success:
            code = failure_code_from_result(result)
            self.failure_distribution[code] = self.failure_distribution.get(code, 0) + 1
        self.total_sim_steps += int(result.num_sim_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_success": bool(self.task_success),
            "learned": {
                "attempts": int(self.learned_attempts),
                "successes": int(self.learned_successes),
                "failures": int(self.learned_failures),
            },
            "rrt": {
                "attempts": int(self.rrt_attempts),
                "successes": int(self.rrt_successes),
            },
            "fallback": {
                "attempts": int(self.fallback_attempts),
                "successes": int(self.fallback_successes),
            },
            "replans": int(self.replans),
            "invalid_plans": int(self.invalid_plans),
            "rollbacks": int(self.rollbacks),
            "failure_distribution": dict(self.failure_distribution),
            "total_sim_steps": int(self.total_sim_steps),
        }


@dataclass
class PlannerRunResult:
    success: bool
    reason: str
    final_obs: Any
    events: PlannerEventLog
    metrics: PlannerMetrics
    feedback_history: List[str]
    budget: RetryBudget
    final_state_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "reason": self.reason,
            "events": self.events.to_list(),
            "metrics": self.metrics.to_dict(),
            "feedback_history": list(self.feedback_history),
            "budget": self.budget.to_dict(),
            "final_state_digest": self.final_state_digest,
        }


class CannedPlanner(object):
    """Deterministic planner response source for tests and dry runs."""

    def __init__(self, responses: Iterable[str]) -> None:
        self.responses = list(responses)
        self.prompts = []  # type: List[str]
        self.contexts = []  # type: List[Dict[str, Any]]
        self.index = 0

    def next_response(self, prompt: str, context: Optional[Mapping[str, Any]] = None) -> str:
        self.prompts.append(prompt)
        self.contexts.append(dict(context or {}))
        if self.index >= len(self.responses):
            return self.responses[-1] if self.responses else ""
        response = self.responses[self.index]
        self.index += 1
        return response


class SimulatorStateController(object):
    """Own snapshot, restore, and digest details for the caller."""

    def snapshot(self, env: Any) -> Any:
        if hasattr(env, "save_intermediate_state"):
            return env.save_intermediate_state()
        if hasattr(env, "snapshot"):
            return env.snapshot()
        return copy.deepcopy(getattr(env, "__dict__", {}))

    def restore(self, env: Any, snapshot: Any) -> None:
        if hasattr(env, "load_saved_state"):
            env.load_saved_state(snapshot)
            return
        if hasattr(env, "restore"):
            env.restore(snapshot)
            return
        if isinstance(snapshot, dict):
            env.__dict__.clear()
            env.__dict__.update(copy.deepcopy(snapshot))
            return
        raise RuntimeError("Environment does not expose a supported restore method")

    def digest(self, env: Any, obs: Any = None, summary: Optional[StateSummary] = None) -> str:
        if hasattr(env, "get_state_digest"):
            digest = env.get_state_digest()
            if isinstance(digest, dict):
                return str(digest.get("digest", digest))
            return str(digest)
        if summary is not None:
            return summary.digest
        if obs is not None:
            return build_state_summary(env, obs).digest
        return ""


class Phase6PromptRenderer(object):
    def __init__(self, router: SkillExecutorRouter, registry: Any = None, agent_names: Optional[Sequence[str]] = None) -> None:
        self.router = router
        self.registry = registry
        self.agent_names = list(agent_names or ())

    def render(
        self,
        state_summary: StateSummary,
        previous_feedback: Optional[str],
        budget: RetryBudget,
    ) -> str:
        capabilities = self.router.prompt_capabilities(self.agent_names, self.registry)
        agent_lines = "\n".join("NAME {} ACTION <SKILL_CALL>".format(name) for name in self.agent_names)
        lines = [
            "[Current State Summary]",
        ]
        for fact in state_summary.measured_facts:
            lines.append("- {}".format(fact))
        lines.extend([
            "",
            "[Per-Agent Skills and Capabilities]",
            capabilities,
            "",
            "[Previous Structured Outcome]",
            previous_feedback or "none",
            "",
            "[Budget and Fallback Constraints]",
            json.dumps(budget.to_dict(), sort_keys=True),
            "- Learned capabilities are available, not guaranteed.",
            "- RRT fallback is explicit when configured.",
            "- Retries, replans, and fallbacks are bounded.",
            "",
            "[Typed Grammar]",
            "EXECUTE",
            agent_lines,
            "",
            "[Rules]",
            "- Output exactly one NAME ... ACTION line per configured agent.",
            "- Use only typed high-level skill calls from the available capabilities.",
            "- Do not output low-level paths, waypoints, controller names, checkpoint paths, PICK, PLACE, MOVE, or raw actions.",
        ])
        return "\n".join(lines)


class SequentialPlanningController(object):
    """Sequential planner integration with bounded recovery."""

    def __init__(
        self,
        env: Any,
        agent_names: Sequence[str],
        parser: Any,
        validator: SkillPlanValidator,
        router: SkillExecutorRouter,
        policy: Optional[ReplanningPolicy],
        executors: Mapping[str, Any],
        rrt_compiler: Optional[SkillCompiler] = None,
        prompt_renderer: Optional[Phase6PromptRenderer] = None,
        feedback_renderer: Optional[GroundedFeedbackRenderer] = None,
        state_controller: Optional[SimulatorStateController] = None,
        registry: Any = None,
    ) -> None:
        self.env = env
        self.agent_names = list(agent_names)
        self.parser = parser
        self.validator = validator
        self.router = router
        self.policy = policy or ReplanningPolicy()
        self.budget = RetryBudget(self.policy)
        self.executors = dict(executors)
        self.rrt_compiler = rrt_compiler
        self.feedback_renderer = feedback_renderer or GroundedFeedbackRenderer()
        self.state_controller = state_controller or SimulatorStateController()
        self.prompt_renderer = prompt_renderer or Phase6PromptRenderer(router, registry=registry, agent_names=agent_names)
        self.recovery_engine = RecoveryEngine(self.policy, self.budget)
        self.metrics = PlannerMetrics()
        self._task_success_fn = None  # type: Optional[Callable[[Any, SkillExecutionResult], bool]]

    def run(
        self,
        planner: Any,
        initial_obs: Any = None,
        run_id: str = "phase6",
        task_success_fn: Optional[Callable[[Any, SkillExecutionResult], bool]] = None,
        artifact_dir: Optional[str] = None,
    ) -> PlannerRunResult:
        event_log = PlannerEventLog(run_id)
        self._task_success_fn = task_success_fn
        feedback_history = []  # type: List[str]
        previous_feedback = None
        obs = initial_obs if initial_obs is not None else self._observe()

        while self.budget.can_start_plan_round():
            self.budget.start_plan_round()
            state_summary = build_state_summary(self.env, obs, self.agent_names)
            event_log.append(
                PlannerEventType.OBSERVED,
                state_digest=state_summary.digest,
                budget=self.budget.to_dict(),
            )
            prompt = self.prompt_renderer.render(state_summary, previous_feedback, self.budget)
            event_log.append(PlannerEventType.PROMPTED, state_digest=state_summary.digest, budget=self.budget.to_dict())
            response = self._planner_response(planner, prompt, {"budget": self.budget.to_dict()})

            parse_ok, parse_message, plans = self.parser.parse(obs, response)
            if not parse_ok or not plans:
                self.budget.record_invalid_plan()
                self.metrics.invalid_plans += 1
                feedback = self.feedback_renderer.render_parse_failure(parse_message, state_summary, self.budget)
                feedback_history.append(feedback)
                previous_feedback = feedback
                event_log.append(
                    PlannerEventType.PLAN_REJECTED,
                    reason=parse_message,
                    state_digest=state_summary.digest,
                    budget=self.budget.to_dict(),
                    metadata={"response": response},
                )
                continue

            plan = plans[0]
            event_log.append(PlannerEventType.PLAN_PARSED, plan_id=plan.plan_id, state_digest=state_summary.digest, budget=self.budget.to_dict())
            validation = self.validator.validate(plan, obs)
            if not validation.valid:
                self.budget.record_invalid_plan()
                self.metrics.invalid_plans += 1
                feedback = self.feedback_renderer.render_validation_failure(validation, state_summary, self.budget)
                feedback_history.append(feedback)
                previous_feedback = feedback
                event_log.append(
                    PlannerEventType.PLAN_REJECTED,
                    plan_id=plan.plan_id,
                    reason=validation.to_feedback(),
                    state_digest=state_summary.digest,
                    budget=self.budget.to_dict(),
                )
                continue

            active_call = self.router.select_active_call(plan)
            if active_call is None:
                validation = SkillValidationResult.invalid([
                    ValidationIssue(
                        code="ACTIVE_CALL_COUNT",
                        message="Phase 6 sequential mode requires exactly one active non-WAIT skill.",
                        retryable=True,
                    )
                ])
                self.budget.record_invalid_plan()
                self.metrics.invalid_plans += 1
                feedback = self.feedback_renderer.render_validation_failure(validation, state_summary, self.budget)
                feedback_history.append(feedback)
                previous_feedback = feedback
                event_log.append(
                    PlannerEventType.PLAN_REJECTED,
                    plan_id=plan.plan_id,
                    reason=validation.to_feedback(),
                    state_digest=state_summary.digest,
                    budget=self.budget.to_dict(),
                )
                continue

            decision = self.router.resolve(active_call, {"budget": self.budget.to_dict()})
            event_log.append(
                PlannerEventType.EXECUTOR_SELECTED,
                plan_id=plan.plan_id,
                backend=decision.backend,
                state_digest=state_summary.digest,
                reason=decision.reason,
                budget=self.budget.to_dict(),
                metadata=decision.to_dict(include_internal=True),
            )
            if not decision.supported:
                self.budget.record_invalid_plan()
                self.metrics.invalid_plans += 1
                validation = SkillValidationResult.invalid([
                    ValidationIssue("UNSUPPORTED_BACKEND", decision.reason, agent_name=active_call.agent_name)
                ])
                feedback = self.feedback_renderer.render_validation_failure(validation, state_summary, self.budget)
                feedback_history.append(feedback)
                previous_feedback = feedback
                event_log.append(
                    PlannerEventType.PLAN_REJECTED,
                    plan_id=plan.plan_id,
                    reason=decision.reason,
                    state_digest=state_summary.digest,
                    budget=self.budget.to_dict(),
                )
                continue

            outcome = self._execute_with_recovery(
                plan,
                active_call,
                decision,
                obs,
                event_log,
                feedback_history,
                artifact_dir=artifact_dir,
            )
            obs = outcome["obs"]
            previous_feedback = outcome.get("feedback")
            if outcome["status"] == "task_success":
                self.metrics.task_success = True
                digest = self.state_controller.digest(self.env, obs, build_state_summary(self.env, obs, self.agent_names))
                event_log.append(PlannerEventType.TASK_SUCCEEDED, plan_id=plan.plan_id, state_digest=digest, budget=self.budget.to_dict())
                return PlannerRunResult(True, "Task succeeded.", obs, event_log, self.metrics, feedback_history, self.budget, digest)
            if outcome["status"] == "abort":
                digest = self.state_controller.digest(self.env, obs, build_state_summary(self.env, obs, self.agent_names))
                event_log.append(PlannerEventType.TASK_ABORTED, plan_id=plan.plan_id, state_digest=digest, reason=outcome.get("reason"), budget=self.budget.to_dict())
                return PlannerRunResult(False, outcome.get("reason", "Task aborted."), obs, event_log, self.metrics, feedback_history, self.budget, digest)
            self.metrics.replans += 1
            event_log.append(PlannerEventType.REPLAN_REQUESTED, plan_id=plan.plan_id, reason=outcome.get("reason"), budget=self.budget.to_dict())

        final_summary = build_state_summary(self.env, obs, self.agent_names)
        event_log.append(
            PlannerEventType.BUDGET_EXHAUSTED,
            state_digest=final_summary.digest,
            reason="Plan round budget exhausted.",
            budget=self.budget.to_dict(),
        )
        return PlannerRunResult(False, "Plan round budget exhausted.", obs, event_log, self.metrics, feedback_history, self.budget, final_summary.digest)

    def _execute_with_recovery(
        self,
        plan: SkillPlan,
        active_call: Any,
        decision: ExecutorDecision,
        obs: Any,
        event_log: PlannerEventLog,
        feedback_history: List[str],
        artifact_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        backend = decision.backend
        fallback_attempt = False
        current_obs = obs

        while True:
            if backend not in self.executors:
                return {"status": "abort", "reason": "No executor configured for backend {}.".format(backend), "obs": current_obs}
            snapshot = self.state_controller.snapshot(self.env)
            before_summary = build_state_summary(self.env, current_obs, self.agent_names, active_call)
            before_digest = self.state_controller.digest(self.env, current_obs, before_summary)
            event_log.append(
                PlannerEventType.STATE_SNAPSHOTTED,
                plan_id=plan.plan_id,
                backend=backend,
                state_digest=before_digest,
                budget=self.budget.to_dict(),
            )
            prepared_ok, prepared_reason = self._prepare_for_backend(plan, backend, current_obs)
            if not prepared_ok:
                result = SkillExecutionResult(
                    success=False,
                    status=SkillExecutionStatus.INVALID_PLAN,
                    reason=prepared_reason,
                    num_sim_steps=0,
                    reward=0.0,
                    done=False,
                    info={},
                    metadata={"failure_code": "INVALID_PLAN"},
                )
            else:
                event_log.append(
                    PlannerEventType.EXECUTION_STARTED,
                    plan_id=plan.plan_id,
                    backend=backend,
                    state_digest=before_digest,
                    budget=self.budget.to_dict(),
                )
                skill_artifact_dir = self._artifact_path(artifact_dir, plan.plan_id, backend, fallback_attempt)
                result = self.executors[backend].execute(plan, current_obs, artifact_dir=skill_artifact_dir)

            self.metrics.record_attempt(backend, result, fallback=fallback_attempt)
            if result.success:
                current_obs = self._observe(default=current_obs)
                digest = self.state_controller.digest(self.env, current_obs, build_state_summary(self.env, current_obs, self.agent_names, active_call))
                event_log.append(
                    PlannerEventType.SKILL_SUCCEEDED,
                    plan_id=plan.plan_id,
                    backend=backend,
                    state_digest=digest,
                    budget=self.budget.to_dict(),
                    metadata={"result": result.to_dict(), "fallback": fallback_attempt},
                )
                if self._task_succeeded(current_obs, result):
                    return {"status": "task_success", "obs": current_obs}
                return {
                    "status": "replan",
                    "obs": current_obs,
                    "reason": "Skill succeeded but task is not complete.",
                    "feedback": "Previous skill succeeded; continue from the current state.",
                }

            failed_obs = self._observe(default=current_obs)
            failed_summary = build_state_summary(self.env, failed_obs, self.agent_names, active_call)
            recovery = self.recovery_engine.decide(result, active_call, failed_summary, decision, backend)
            event_log.append(
                PlannerEventType.EXECUTION_FAILED,
                plan_id=plan.plan_id,
                backend=backend,
                state_digest=failed_summary.digest,
                reason=result.reason,
                budget=self.budget.to_dict(),
                metadata={"result": result.to_dict(), "recovery": recovery.to_dict()},
            )

            feedback_obs = failed_obs
            feedback_summary = failed_summary
            if recovery.rollback:
                self.state_controller.restore(self.env, snapshot)
                feedback_obs = self._observe(default=current_obs)
                feedback_summary = build_state_summary(self.env, feedback_obs, self.agent_names, active_call)
                after_digest = self.state_controller.digest(self.env, feedback_obs, feedback_summary)
                self.metrics.rollbacks += 1
                event_log.append(
                    PlannerEventType.STATE_RESTORED,
                    plan_id=plan.plan_id,
                    backend=backend,
                    state_digest=after_digest,
                    reason="rollback_digest_equal={}".format(before_digest == after_digest),
                    budget=self.budget.to_dict(),
                    metadata={"before_digest": before_digest, "after_digest": after_digest, "equal": before_digest == after_digest},
                )

            feedback = self.feedback_renderer.render_failure(
                active_call,
                result.status.value if hasattr(result.status, "value") else str(result.status),
                recovery.failure_code,
                result.metadata.get("progress_stage") if isinstance(result.metadata, dict) else None,
                feedback_summary,
                self.budget,
                recovery.action,
                reason=result.reason,
            )
            feedback_history.append(feedback)
            event_log.append(
                PlannerEventType.FEEDBACK_RENDERED,
                plan_id=plan.plan_id,
                backend=backend,
                state_digest=feedback_summary.digest,
                reason=recovery.reason,
                budget=self.budget.to_dict(),
            )

            if recovery.action == RecoveryAction.RETRY_LEARNED:
                current_obs = feedback_obs
                backend = BACKEND_LEARNED
                continue
            if recovery.action == RecoveryAction.USE_RRT_FALLBACK:
                backend = BACKEND_RRT
                fallback_attempt = True
                current_obs = feedback_obs
                event_log.append(
                    PlannerEventType.FALLBACK_STARTED,
                    plan_id=plan.plan_id,
                    backend=backend,
                    state_digest=feedback_summary.digest,
                    reason=recovery.reason,
                    budget=self.budget.to_dict(),
                    metadata=recovery.to_dict(),
                )
                continue
            if recovery.action in (RecoveryAction.REPLAN_FROM_CURRENT_STATE, RecoveryAction.ROLLBACK_AND_REPLAN):
                return {"status": "replan", "obs": feedback_obs, "feedback": feedback, "reason": recovery.reason}
            return {"status": "abort", "obs": feedback_obs, "feedback": feedback, "reason": recovery.reason}

    def _prepare_for_backend(self, plan: SkillPlan, backend: str, obs: Any) -> Tuple[bool, str]:
        if backend == BACKEND_RRT:
            if self.rrt_compiler is None:
                return False, "RRT backend requires an RRT skill compiler."
            try:
                plan.prepared_execution = self.rrt_compiler.compile(plan, obs)
            except SkillCompilationError as exc:
                return False, "[{}] {}".format(exc.code, exc.message)
            return True, ""
        if backend == BACKEND_LEARNED:
            plan.prepared_execution = None
            return True, ""
        if backend == BACKEND_UNSUPPORTED:
            return False, "Unsupported backend."
        return False, "Unknown backend {}.".format(backend)

    def _planner_response(self, planner: Any, prompt: str, context: Mapping[str, Any]) -> str:
        if hasattr(planner, "next_response"):
            return planner.next_response(prompt, context)
        return planner(prompt, context)

    def _observe(self, default: Any = None) -> Any:
        if hasattr(self.env, "get_obs"):
            return self.env.get_obs()
        return default

    def _task_succeeded(self, obs: Any, result: SkillExecutionResult) -> bool:
        if self._task_success_fn is not None:
            return bool(self._task_success_fn(obs, result))
        if bool(result.info.get("is_success")):
            return True
        if result.done or result.reward > 0:
            return True
        if hasattr(self.env, "is_task_success"):
            return bool(self.env.is_task_success(obs))
        return False

    def _artifact_path(self, artifact_dir: Optional[str], plan_id: str, backend: str, fallback: bool) -> Optional[str]:
        if artifact_dir is None:
            return None
        suffix = "fallback_{}".format(backend) if fallback else backend
        path = os.path.join(artifact_dir, "{}_{}".format(plan_id, suffix))
        os.makedirs(path, exist_ok=True)
        return path


def load_phase6_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            return json.load(f)
        return yaml.safe_load(f)
