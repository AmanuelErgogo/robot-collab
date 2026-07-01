"""Architecture-level CRIE-BT controllers for ablation studies."""

from collections import Counter
from typing import Any, Dict, List, Optional

from .bt_controller import BehaviorTreeController
from .communication import CommunicationManager
from .executor import BaseSkillExecutor
from .failure import FailureDetector
from .planner import BasePlanner
from .progress import ProgressMonitor
from .status import BTStatus, ExecutionMode, RuntimeDecision
from .types import CollaborativePlan, ExecutionContext, ExecutionFeedback, RuntimeEvent
from .uncertainty import UncertaintyEstimator


class BaseArchitectureController(object):
    mode = None

    def __init__(
        self,
        planner: BasePlanner,
        executor: BaseSkillExecutor,
        uncertainty_mode: str = "heuristic",
        max_retries: int = 1,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.uncertainty_mode = uncertainty_mode
        self.max_retries = int(max_retries)
        self.communication = CommunicationManager()

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _empty_log(self, mode: ExecutionMode, task_goal: str) -> Dict[str, Any]:
        return {
            "mode": mode.value,
            "task": task_goal,
            "success": False,
            "steps": 0,
            "completed_subtasks": 0,
            "failed_subtasks": 0,
            "planner_calls": 0,
            "replans": 0,
            "local_retries": 0,
            "failure_counts": {},
            "events": [],
            "subtask_results": [],
            "explanations": [],
        }

    def _record_event(self, log: Dict[str, Any], event: RuntimeEvent) -> None:
        log["events"].append(event.to_dict())
        messages = self.communication.handle_event(event)
        log["explanations"].extend(messages)

    def _record_failure(self, log: Dict[str, Any], feedback: Optional[ExecutionFeedback]) -> None:
        if feedback is None or not feedback.failure.is_failure:
            return
        key = feedback.failure.failure_code.value
        log["failure_counts"][key] = int(log["failure_counts"].get(key, 0)) + 1

    def _get_observation(self, env, feedback: Optional[ExecutionFeedback] = None, fallback: Any = None) -> Any:
        if feedback is not None:
            raw_info = dict(feedback.raw_info or {})
            if "observation" in raw_info:
                return raw_info["observation"]
        if hasattr(env, "get_obs"):
            return env.get_obs()
        if hasattr(env, "get_observation"):
            return env.get_observation()
        return {} if fallback is None else fallback

    def _task_done(self, env, observation: Any) -> bool:
        """Return True when the environment signals the full task is complete.

        Falls back to True when the env has no get_reward_done() method so that
        controllers behave correctly with simple test stubs (where "plan done"
        is equivalent to "task done").
        """
        if not hasattr(env, "get_reward_done"):
            return True
        if observation is None:
            return False
        try:
            return bool(env.get_reward_done(observation)[1])
        except Exception:
            return True

    def _replan(
        self,
        log: Dict[str, Any],
        task_goal: str,
        observation: Any,
        feedback: Optional[ExecutionFeedback] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Optional[CollaborativePlan]:
        """Call the planner and return the new plan, or None on planner error."""
        log["planner_calls"] += 1
        try:
            return self.planner.generate_plan(task_goal, observation, feedback=feedback, context=context)
        except Exception as exc:
            log["failure_counts"]["PLANNER_ERROR"] = int(log["failure_counts"].get("PLANNER_ERROR", 0)) + 1
            log["explanations"].append("Planner failed: {}".format(exc))
            return None


class OpenLoopController(BaseArchitectureController):
    mode = ExecutionMode.OPEN_LOOP

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=0)
        observation = env.reset() if hasattr(env, "reset") else {}
        self.planner.reset_episode()
        try:
            plan = self.planner.generate_plan(task_goal, observation, context=context)
        except Exception as exc:
            log["planner_calls"] = 1
            log["failure_counts"]["PLANNER_ERROR"] = 1
            log["explanations"].append("Planner failed to generate plan: {}".format(exc))
            return log
        log["planner_calls"] = 1
        self.executor.reset(env, context)
        for step in plan.steps:
            self.executor.start_skill(step.skill_call, observation)
            feedback = None
            while log["steps"] < max_steps:
                feedback = self.executor.step(observation)
                log["steps"] += 1
                observation = self._get_observation(env, feedback, observation)
                if feedback.status != BTStatus.RUNNING:
                    break
            self.executor.stop()
            self._record_failure(log, feedback)
            log["subtask_results"].append(feedback.to_dict() if feedback is not None else {"step_id": step.step_id})
            if feedback is not None and feedback.status == BTStatus.SUCCESS and not feedback.failure.is_failure:
                log["completed_subtasks"] += 1
            else:
                log["failed_subtasks"] += 1
                break
        log["success"] = log["completed_subtasks"] == len(plan.steps)
        return log


class DirectFeedbackController(BaseArchitectureController):
    """Multi-step reactive controller.

    Each call to the planner generates exactly one action (one EXECUTE block via
    LegacyPromptPlanner).  After a successful action the controller checks whether
    the task is done via env.get_reward_done(); if not it asks the planner for the
    next action, passing the updated observation so the LLM can use round history.
    On failure the planner is called immediately for a corrective action.
    """

    mode = ExecutionMode.DIRECT_FEEDBACK

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=self.max_retries)
        observation = env.reset() if hasattr(env, "reset") else {}

        # Clear any state left over from a previous episode.
        self.planner.reset_episode()

        plan = self._replan(log, task_goal, observation, context=context)
        if plan is None:
            return log  # planner error on first call
        self.executor.reset(env, context)
        step_index = 0
        consecutive_failures = 0

        while log["steps"] < max_steps:
            if step_index >= len(plan.steps):
                # All steps in the current plan executed successfully.
                if self._task_done(env, observation):
                    log["success"] = True
                    return log
                # Task not complete yet: notify the planner the last action
                # succeeded and ask for the next action.
                self.planner.notify_result(True)
                log["replans"] += 1
                plan = self._replan(log, task_goal, observation, context=context)
                if plan is None:
                    return log
                step_index = 0
                continue

            step = plan.steps[step_index]
            self.executor.start_skill(step.skill_call, observation)
            feedback = None
            while log["steps"] < max_steps:
                feedback = self.executor.step(observation)
                log["steps"] += 1
                observation = self._get_observation(env, feedback, observation)
                if feedback.status != BTStatus.RUNNING:
                    break
            self.executor.stop()
            log["subtask_results"].append(feedback.to_dict() if feedback is not None else {"step_id": step.step_id})
            self._record_failure(log, feedback)

            if feedback is not None and feedback.status == BTStatus.SUCCESS and not feedback.failure.is_failure:
                log["completed_subtasks"] += 1
                step_index += 1
                consecutive_failures = 0
                continue

            # Step failed: count consecutive failures and enforce retry cap before
            # asking the planner to generate a new plan.
            log["failed_subtasks"] += 1
            consecutive_failures += 1
            if consecutive_failures > self.max_retries:
                log["explanations"].append(
                    "Executor failed {} consecutive time(s) — max_retries={} exceeded. "
                    "Terminating episode.".format(consecutive_failures, self.max_retries)
                )
                return log
            log["replans"] += 1
            self.planner.notify_result(False)
            plan = self._replan(log, task_goal, observation, feedback=feedback, context=context)
            if plan is None:
                return log
            step_index = 0

        return log


class BTMediatedController(BaseArchitectureController):
    """Multi-step BT-mediated controller.

    Each plan produced by the planner contains one action step.  When the BT
    reports SUCCESS (all steps done), the controller checks whether the task is
    complete.  If not, it requests the next action from the planner (notifying it
    that the last action succeeded so the LLM history is updated) and resets the
    BT with the new plan.  Failure handling follows the BT's decision: local retry,
    replan, or hard stop.
    """

    mode = ExecutionMode.BT_MEDIATED

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=self.max_retries)
        observation = env.reset() if hasattr(env, "reset") else {}

        self.planner.reset_episode()

        plan = self._replan(log, task_goal, observation, context=context)
        if plan is None:
            return log

        bt = BehaviorTreeController(
            self.executor,
            progress_monitor=ProgressMonitor(env=env, max_steps=max_steps),
            uncertainty_estimator=UncertaintyEstimator(self.uncertainty_mode),
            failure_detector=FailureDetector(max_steps=max_steps),
            max_retries=self.max_retries,
        )
        bt.reset(plan, env, observation)
        seen_events = 0

        while log["steps"] < max_steps:
            result = bt.tick(observation)
            log["steps"] += 1
            for event in result.events:
                self._record_event(log, event)
            all_events = bt.get_events()
            for event in all_events[seen_events:]:
                if event.to_dict() not in log["events"]:
                    self._record_event(log, event)
            seen_events = len(all_events)
            self._record_failure(log, result.feedback)
            if result.decision.decision == RuntimeDecision.LOCAL_RETRY:
                log["local_retries"] += 1
            if result.feedback is not None:
                log["subtask_results"].append(result.feedback.to_dict())
            observation = self._get_observation(env, result.feedback, observation)

            if result.status == BTStatus.SUCCESS:
                log["completed_subtasks"] = result.completed_subtasks
                # All plan steps succeeded.  Check whether the task itself is done.
                if self._task_done(env, observation):
                    log["success"] = True
                    return log
                # Task not complete: notify the planner that the last action
                # succeeded and ask for the next action.
                self.planner.notify_result(True)
                log["replans"] += 1
                plan = self._replan(log, task_goal, observation, context=context)
                if plan is None:
                    return log
                bt.reset(plan, env, observation)
                seen_events = 0
                continue

            if result.status == BTStatus.FAILURE and result.decision.decision == RuntimeDecision.REQUEST_REPLAN:
                log["failed_subtasks"] = result.failed_subtasks
                log["replans"] += 1
                self.planner.notify_result(False)
                plan = self._replan(log, task_goal, observation, feedback=result.feedback, context=context)
                if plan is None:
                    return log
                bt.reset(plan, env, observation)
                seen_events = 0
            elif result.status == BTStatus.FAILURE:
                log["failed_subtasks"] = result.failed_subtasks
                return log

        return log


def build_controller(mode: str, planner: BasePlanner, executor: BaseSkillExecutor, uncertainty_mode: str = "heuristic", max_retries: int = 1):
    if mode == ExecutionMode.OPEN_LOOP.value:
        return OpenLoopController(planner, executor, uncertainty_mode=uncertainty_mode, max_retries=max_retries)
    if mode == ExecutionMode.DIRECT_FEEDBACK.value:
        return DirectFeedbackController(planner, executor, uncertainty_mode=uncertainty_mode, max_retries=max_retries)
    if mode == ExecutionMode.BT_MEDIATED.value:
        return BTMediatedController(planner, executor, uncertainty_mode=uncertainty_mode, max_retries=max_retries)
    raise ValueError("Unknown CRIE-BT mode: {}".format(mode))
