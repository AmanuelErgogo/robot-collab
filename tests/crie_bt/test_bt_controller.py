from rocobench.crie_bt.bt_controller import BehaviorTreeController
from rocobench.crie_bt.executor import ScriptedSkillExecutor
from rocobench.crie_bt.planner import ScriptedPlanner
from rocobench.crie_bt.status import BTStatus, RuntimeDecision


def test_bt_mediated_retries_locally_after_missed_grasp():
    plan = ScriptedPlanner().generate_plan("pack object=apple into bin_front_left", {})
    bt = BehaviorTreeController(ScriptedSkillExecutor("missed_grasp", fail_attempts=1), max_retries=1)
    bt.reset(plan, None, {})
    first = bt.tick({})
    assert first.decision.decision == RuntimeDecision.LOCAL_RETRY
    second = bt.tick({})
    assert second.status == BTStatus.SUCCESS


def test_bt_mediated_escalates_after_repeated_failure():
    plan = ScriptedPlanner().generate_plan("pack object=apple into bin_front_left", {})
    bt = BehaviorTreeController(ScriptedSkillExecutor("missed_grasp", fail_attempts=5), max_retries=1)
    bt.reset(plan, None, {})
    first = bt.tick({})
    assert first.decision.decision == RuntimeDecision.LOCAL_RETRY
    second = bt.tick({})
    assert second.status == BTStatus.FAILURE
    assert second.decision.decision == RuntimeDecision.REQUEST_REPLAN


def test_low_confidence_explains_without_failure():
    plan = ScriptedPlanner().generate_plan("pack object=apple into bin_front_left", {})
    bt = BehaviorTreeController(ScriptedSkillExecutor("none", low_confidence=True), max_retries=1)
    bt.reset(plan, None, {})
    result = bt.tick({})
    assert result.status == BTStatus.RUNNING
    assert result.decision.decision == RuntimeDecision.EXPLAIN
