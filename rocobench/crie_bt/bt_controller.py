"""Behavior-tree-mediated runtime controller."""

from typing import Any, Dict, List, Optional

from .bt_nodes import ExecuteSkillNode
from .events import RuntimeEventLog
from .failure import FailureDetector
from .progress import ProgressMonitor
from .status import BTStatus, FailureCode, RuntimeDecision
from .types import BTDecision, CollaborativePlan, ControllerResult, ExecutionContext, ExecutionFeedback, RuntimeEvent, SkillCall
from .uncertainty import UncertaintyEstimator


RECOVERABLE_FAILURES = set([
    FailureCode.MISSED_GRASP,
    FailureCode.NO_PROGRESS,
    FailureCode.SLIPPAGE,
])

REPLAN_FAILURES = set([
    FailureCode.POSTCONDITION_FAILED,
    FailureCode.TARGET_OCCUPIED,
    FailureCode.WRONG_OBJECT,
    FailureCode.WRONG_TARGET,
    FailureCode.SAFETY_CONFLICT,
])


class BehaviorTreeController(object):
    def __init__(
        self,
        executor: Any,
        progress_monitor: Optional[ProgressMonitor] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        failure_detector: Optional[FailureDetector] = None,
        max_retries: int = 1,
        timeout_decision: RuntimeDecision = RuntimeDecision.REQUEST_REPLAN,
    ) -> None:
        self.executor = executor
        self.progress_monitor = progress_monitor or ProgressMonitor()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator("heuristic")
        self.failure_detector = failure_detector or FailureDetector()
        self.max_retries = int(max_retries)
        self.timeout_decision = timeout_decision
        self.plan = None  # type: Optional[CollaborativePlan]
        self.env = None
        self.context = ExecutionContext(max_retries=max_retries)
        self.current_index = 0
        self.completed_subtasks = 0
        self.failed_subtasks = 0
        self.retry_counts = {}  # type: Dict[str, int]
        self.events = RuntimeEventLog()
        self.current_node = None  # type: Optional[ExecuteSkillNode]
        self.last_feedback = None  # type: Optional[ExecutionFeedback]
        self.done = False

    def reset(self, plan: CollaborativePlan, env, observation: Any) -> None:
        self.plan = plan
        self.env = env
        self.current_index = 0
        self.completed_subtasks = 0
        self.failed_subtasks = 0
        self.retry_counts = {}
        self.events.clear()
        self.done = False
        self.current_node = None
        self.last_feedback = None
        self.context = ExecutionContext(mode=self.context.mode, task_name=plan.task_goal, max_retries=self.max_retries)
        self.executor.reset(env, self.context)
        self.uncertainty_estimator.reset()
        self.events.append("PLAN_STARTED", "Behavior tree started plan {}.".format(plan.plan_id), plan_id=plan.plan_id)
        del observation

    def tick(self, observation: Any) -> ControllerResult:
        if self.plan is None:
            raise RuntimeError("BehaviorTreeController.reset must be called before tick.")
        if self.done:
            return ControllerResult(status=BTStatus.SUCCESS, completed_subtasks=self.completed_subtasks, events=self.get_events())
        if self.current_index >= len(self.plan.steps):
            self.done = True
            event = self.events.append("TASK_COMPLETE", "All subtasks completed.")
            return ControllerResult(status=BTStatus.SUCCESS, events=[event], completed_subtasks=self.completed_subtasks)

        step = self.plan.steps[self.current_index]
        if self.current_node is None:
            self.current_node = ExecuteSkillNode(
                self.executor,
                self.progress_monitor,
                self.uncertainty_estimator,
                self.failure_detector,
                step.skill_call,
                self.context,
            )

        result = self.current_node.tick(observation)
        self.events.extend(result.events)
        self.last_feedback = result.feedback
        feedback = result.feedback

        if feedback is not None and (
            feedback.status == BTStatus.SUCCESS
            or feedback.progress.postcondition_satisfied
            or feedback.progress.stage.value == "STABLE_SUCCESS"
        ) and not feedback.failure.is_failure:
            self.completed_subtasks += 1
            self.current_index += 1
            self.current_node = None
            event = self.events.append("SUBTASK_COMPLETED", "Completed subtask {}.".format(step.step_id), step_id=step.step_id, skill_call=step.skill_call)
            if self.current_index >= len(self.plan.steps):
                self.done = True
                done_event = self.events.append("TASK_COMPLETE", "All subtasks completed.")
                return ControllerResult(status=BTStatus.SUCCESS, feedback=feedback, events=[event, done_event], completed_subtasks=self.completed_subtasks)
            decision = BTDecision(RuntimeDecision.CONTINUE, "Subtask completed; continuing.", retry_count=0)
            return ControllerResult(status=BTStatus.RUNNING, decision=decision, feedback=feedback, events=[event], completed_subtasks=self.completed_subtasks)

        if feedback is not None and feedback.failure.is_failure:
            return self._handle_failure(step.step_id, feedback)

        if feedback is not None and feedback.uncertainty.risk_level in ("medium", "high"):
            decision = BTDecision(RuntimeDecision.EXPLAIN, "Uncertainty observed while progress continues.", retry_count=self.retry_counts.get(step.step_id, 0))
            event = self.events.append("EXPLAIN", "Progress continues with uncertainty.", step_id=step.step_id, skill_call=step.skill_call, decision=decision)
            return ControllerResult(status=BTStatus.RUNNING, decision=decision, feedback=feedback, events=[event], completed_subtasks=self.completed_subtasks)

        decision = BTDecision(RuntimeDecision.CONTINUE, "Skill running.", retry_count=self.retry_counts.get(step.step_id, 0))
        return ControllerResult(status=BTStatus.RUNNING, decision=decision, feedback=feedback, events=[], completed_subtasks=self.completed_subtasks)

    def _handle_failure(self, step_id: str, feedback: ExecutionFeedback) -> ControllerResult:
        code = feedback.failure.failure_code
        retry_count = self.retry_counts.get(step_id, 0)
        if code == FailureCode.LOW_CONFIDENCE:
            decision = BTDecision(RuntimeDecision.EXPLAIN, feedback.failure.message, retry_count=retry_count)
            event = self.events.append("LOW_CONFIDENCE", feedback.failure.message, step_id=step_id, skill_call=feedback.skill_call, decision=decision)
            return ControllerResult(BTStatus.RUNNING, decision, feedback, [event], self.completed_subtasks, self.failed_subtasks)
        if code in RECOVERABLE_FAILURES and retry_count < self.max_retries:
            retry_count += 1
            self.retry_counts[step_id] = retry_count
            self.executor.stop()
            if self.current_node is not None:
                self.current_node.reset_started()
            decision = BTDecision(RuntimeDecision.LOCAL_RETRY, feedback.failure.message, retry_count=retry_count)
            event = self.events.append("LOCAL_RETRY", feedback.failure.message, step_id=step_id, skill_call=feedback.skill_call, decision=decision)
            return ControllerResult(BTStatus.RUNNING, decision, feedback, [event], self.completed_subtasks, self.failed_subtasks)

        # RRT-aware local recovery: when an RRT timeout occurred on a simultaneous
        # multi-agent action, the executor embeds a pre-computed serialized action in
        # raw_info["recovery_response"].  Retry once using that action before escalating
        # to the LLM planner.
        if code == FailureCode.TIMEOUT and retry_count < self.max_retries:
            recovery_response = dict(feedback.raw_info or {}).get("recovery_response")
            if recovery_response and self.current_node is not None:
                retry_count += 1
                self.retry_counts[step_id] = retry_count
                self.executor.stop()
                original = self.current_node.skill_call
                recovery_skill_call = SkillCall(
                    agent=original.agent,
                    skill_name=original.skill_name,
                    arguments=dict(original.arguments, response=recovery_response),
                    instruction="RRT-aware serialized recovery (retry {}).".format(retry_count),
                )
                self.current_node.skill_call = recovery_skill_call
                self.current_node.reset_started()
                msg = "RRT timeout on simultaneous actions — retrying with serialized execution."
                decision = BTDecision(RuntimeDecision.LOCAL_RETRY, msg, retry_count=retry_count)
                event = self.events.append("LOCAL_RETRY", msg, step_id=step_id, skill_call=feedback.skill_call, decision=decision)
                return ControllerResult(BTStatus.RUNNING, decision, feedback, [event], self.completed_subtasks, self.failed_subtasks)

        self.failed_subtasks += 1
        if code == FailureCode.TIMEOUT:
            decision_value = self.timeout_decision
        elif code in REPLAN_FAILURES or code in RECOVERABLE_FAILURES:
            decision_value = RuntimeDecision.REQUEST_REPLAN
        else:
            decision_value = RuntimeDecision.REQUEST_REPLAN
        decision = BTDecision(decision_value, feedback.failure.message, retry_count=retry_count, metadata={"failure_code": code.value})
        event_type = "REQUEST_REPLAN" if decision_value == RuntimeDecision.REQUEST_REPLAN else decision_value.value
        event = self.events.append(event_type, feedback.failure.message, step_id=step_id, skill_call=feedback.skill_call, decision=decision)
        return ControllerResult(BTStatus.FAILURE, decision, feedback, [event], self.completed_subtasks, self.failed_subtasks)

    def is_done(self) -> bool:
        return bool(self.done)

    def get_events(self) -> List[RuntimeEvent]:
        return self.events.to_list()
