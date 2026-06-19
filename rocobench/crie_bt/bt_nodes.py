"""Lightweight behavior-tree nodes for CRIE-BT."""

from typing import Any, Callable, List, Optional

from .failure import FailureDetector
from .progress import ProgressMonitor
from .status import BTStatus
from .types import BTDecision, ControllerResult, ExecutionContext, RuntimeEvent, SkillCall
from .uncertainty import UncertaintyEstimator


class BTNode(object):
    def tick(self, observation: Any) -> ControllerResult:
        raise NotImplementedError


class SequenceNode(BTNode):
    def __init__(self, children: List[BTNode]) -> None:
        self.children = list(children)

    def tick(self, observation: Any) -> ControllerResult:
        events = []
        for child in self.children:
            result = child.tick(observation)
            events.extend(result.events)
            if result.status != BTStatus.SUCCESS:
                return ControllerResult(status=result.status, decision=result.decision, feedback=result.feedback, events=events, message=result.message)
        return ControllerResult(status=BTStatus.SUCCESS, events=events)


class FallbackNode(BTNode):
    def __init__(self, children: List[BTNode]) -> None:
        self.children = list(children)

    def tick(self, observation: Any) -> ControllerResult:
        events = []
        last = None
        for child in self.children:
            result = child.tick(observation)
            events.extend(result.events)
            last = result
            if result.status != BTStatus.FAILURE:
                return ControllerResult(status=result.status, decision=result.decision, feedback=result.feedback, events=events, message=result.message)
        return last or ControllerResult(status=BTStatus.FAILURE, events=events)


class ConditionNode(BTNode):
    def __init__(self, predicate: Callable[[Any], bool], message: str = "") -> None:
        self.predicate = predicate
        self.message = message

    def tick(self, observation: Any) -> ControllerResult:
        if self.predicate(observation):
            return ControllerResult(status=BTStatus.SUCCESS, message=self.message)
        return ControllerResult(status=BTStatus.FAILURE, message=self.message)


class ActionNode(BTNode):
    def __init__(self, action: Callable[[Any], ControllerResult]) -> None:
        self.action = action

    def tick(self, observation: Any) -> ControllerResult:
        return self.action(observation)


class ExecuteSkillNode(BTNode):
    """Core execution node: executor step plus progress/uncertainty/failure."""

    def __init__(
        self,
        executor: Any,
        progress_monitor: ProgressMonitor,
        uncertainty_estimator: UncertaintyEstimator,
        failure_detector: FailureDetector,
        skill_call: SkillCall,
        context: ExecutionContext,
    ) -> None:
        self.executor = executor
        self.progress_monitor = progress_monitor
        self.uncertainty_estimator = uncertainty_estimator
        self.failure_detector = failure_detector
        self.skill_call = skill_call
        self.context = context
        self.started = False

    def tick(self, observation: Any) -> ControllerResult:
        if not self.started:
            self.executor.start_skill(self.skill_call, observation)
            self.progress_monitor.reset(self.skill_call, observation)
            self.started = True
            start_event = RuntimeEvent("SUBTASK_STARTED", "Started {}.".format(self.skill_call.skill_name), skill_call=self.skill_call)
        else:
            start_event = None
        feedback = self.executor.step(observation)
        progress = self.progress_monitor.update(self.skill_call, observation, dict(feedback.raw_info or {}))
        uncertainty = self.uncertainty_estimator.estimate(self.skill_call, observation, feedback)
        failure = self.failure_detector.detect(self.skill_call, progress, uncertainty, observation, feedback)
        feedback = type(feedback)(
            skill_call=feedback.skill_call,
            status=feedback.status,
            progress=progress if progress.score >= feedback.progress.score else feedback.progress,
            uncertainty=uncertainty if uncertainty.uncertainty >= feedback.uncertainty.uncertainty else feedback.uncertainty,
            failure=failure if failure.is_failure else feedback.failure,
            message=feedback.message,
            raw_info=feedback.raw_info,
        )
        events = [event for event in (start_event,) if event is not None]
        events.append(RuntimeEvent("EXECUTOR_FEEDBACK", feedback.message, skill_call=self.skill_call, payload=feedback.to_dict()))
        status = feedback.status
        if feedback.failure.is_failure:
            status = BTStatus.FAILURE
        return ControllerResult(status=status, feedback=feedback, events=events, message=feedback.message)

    def reset_started(self) -> None:
        self.started = False


class RetryNode(ActionNode):
    pass


class ExplainNode(ActionNode):
    pass


class RequestHumanInputNode(ActionNode):
    pass


class RequestReplanNode(ActionNode):
    pass
