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


class OpenLoopController(BaseArchitectureController):
    mode = ExecutionMode.OPEN_LOOP

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=0)
        observation = env.reset() if hasattr(env, "reset") else {}
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
    mode = ExecutionMode.DIRECT_FEEDBACK

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=0)
        observation = env.reset() if hasattr(env, "reset") else {}
        feedback = None  # type: Optional[ExecutionFeedback]
        plan = self.planner.generate_plan(task_goal, observation, context=context)
        log["planner_calls"] = 1
        self.executor.reset(env, context)
        step_index = 0
        while log["steps"] < max_steps:
            if step_index >= len(plan.steps):
                log["success"] = True
                return log
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
                continue
            log["failed_subtasks"] += 1
            log["replans"] += 1
            log["planner_calls"] += 1
            plan = self.planner.generate_plan(task_goal, observation, feedback=feedback, context=context)
            step_index = 0
        return log


class BTMediatedController(BaseArchitectureController):
    mode = ExecutionMode.BT_MEDIATED

    def run_episode(self, env, task_goal: str, max_steps: int) -> Dict[str, Any]:
        log = self._empty_log(self.mode, task_goal)
        context = ExecutionContext(mode=self.mode, task_name=task_goal, max_steps=max_steps, max_retries=self.max_retries)
        observation = env.reset() if hasattr(env, "reset") else {}
        plan = self.planner.generate_plan(task_goal, observation, context=context)
        log["planner_calls"] = 1
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
                log["success"] = True
                return log
            if result.status == BTStatus.FAILURE and result.decision.decision == RuntimeDecision.REQUEST_REPLAN:
                log["failed_subtasks"] = result.failed_subtasks
                log["replans"] += 1
                log["planner_calls"] += 1
                plan = self.planner.generate_plan(task_goal, observation, feedback=result.feedback, context=context)
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
