from rocobench.crie_bt.planner import LLMPlannerAdapter, ScriptedPlanner
from rocobench.crie_bt.status import FailureCode
from rocobench.crie_bt.types import ExecutionFeedback, FailureState, SkillCall


def test_scripted_planner_generates_pack_plan():
    plan = ScriptedPlanner().generate_plan("pack object=apple into bin_front_left", {})
    call = plan.steps[0].skill_call
    assert call.skill_name == "PUT_OBJECT_IN_CONTAINER"
    assert call.arguments["object"] == "apple"
    assert call.arguments["container"] == "bin_front_left"


def test_scripted_planner_uses_feedback_for_target_occupied():
    feedback = ExecutionFeedback(
        SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin_front_left"}),
        failure=FailureState(True, FailureCode.TARGET_OCCUPIED, "error", "occupied", {}),
    )
    plan = ScriptedPlanner().generate_plan("pack object=apple into bin_front_left", {}, feedback=feedback)
    assert plan.steps[0].skill_call.arguments["container"] == "bin_front_right"


def test_llm_adapter_placeholder_is_explicit():
    try:
        LLMPlannerAdapter().generate_plan("pack", {})
    except NotImplementedError as exc:
        assert "requires an injected" in str(exc)
    else:
        raise AssertionError("LLMPlannerAdapter should require injection")
