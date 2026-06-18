import json

from rocobench.crie_bt.status import BTStatus, ExecutionMode, FailureCode, RuntimeDecision
from rocobench.crie_bt.types import (
    BTDecision,
    CollaborativePlan,
    ControllerResult,
    ExecutionContext,
    ExecutionFeedback,
    FailureState,
    PlanStep,
    SkillCall,
)


def test_types_round_trip_json_serialization():
    skill = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin"}, "Place apple.")
    plan = CollaborativePlan([PlanStep("s1", skill, {"Alice": "robot_executor"}, "demo")], "p1", "pack")
    encoded = json.loads(json.dumps(plan.to_dict()))
    assert CollaborativePlan.from_dict(encoded).steps[0].skill_call.arguments["object"] == "apple"

    feedback = ExecutionFeedback(
        skill,
        status=BTStatus.FAILURE,
        failure=FailureState(True, FailureCode.MISSED_GRASP, "warning", "missed", {}),
    )
    result = ControllerResult(
        status=BTStatus.FAILURE,
        decision=BTDecision(RuntimeDecision.LOCAL_RETRY, "retry", 1),
        feedback=feedback,
    )
    decoded = ControllerResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert decoded.decision.decision == RuntimeDecision.LOCAL_RETRY
    assert decoded.feedback.failure.failure_code == FailureCode.MISSED_GRASP


def test_execution_context_serialization_uses_mode_value():
    context = ExecutionContext(mode=ExecutionMode.OPEN_LOOP, task_name="pack")
    data = context.to_dict()
    assert data["mode"] == "open_loop"
    assert ExecutionContext.from_dict(data).mode == ExecutionMode.OPEN_LOOP
