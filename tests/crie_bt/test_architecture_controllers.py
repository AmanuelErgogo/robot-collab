from rocobench.crie_bt.controllers import BTMediatedController, DirectFeedbackController, OpenLoopController, VLMSARMMonitorPlannerController
from rocobench.crie_bt.executor import ScriptedSkillExecutor
from rocobench.crie_bt.planner import ScriptedPlanner


class Env:
    def reset(self):
        return {}


def test_open_loop_runs_successful_scripted_plan():
    log = OpenLoopController(ScriptedPlanner(), ScriptedSkillExecutor()).run_episode(Env(), "pack object=apple into bin", 5)
    assert log["success"]
    assert log["planner_calls"] == 1
    assert log["replans"] == 0


def test_direct_feedback_replans_after_scripted_failure():
    log = DirectFeedbackController(
        ScriptedPlanner(),
        ScriptedSkillExecutor("missed_grasp", fail_attempts=1),
    ).run_episode(Env(), "pack object=apple into bin", 5)
    assert log["success"]
    assert log["planner_calls"] == 2
    assert log["replans"] == 1


def test_bt_mediated_retries_before_replanning():
    log = BTMediatedController(
        ScriptedPlanner(),
        ScriptedSkillExecutor("missed_grasp", fail_attempts=1),
        max_retries=1,
    ).run_episode(Env(), "pack object=apple into bin", 5)
    assert log["success"]
    assert log["local_retries"] == 1
    assert log["replans"] == 0


def test_vlm_sarm_monitor_planner_replans_from_sim_failure_signal():
    log = VLMSARMMonitorPlannerController(
        ScriptedPlanner(),
        ScriptedSkillExecutor("missed_grasp", fail_attempts=1),
    ).run_episode(Env(), "pack object=apple into bin", 5)
    assert log["success"]
    assert log["planner_calls"] == 2
    assert log["replans"] == 1
    assert log["local_retries"] == 0
    assert any(event["event_type"] == "VLM_SARM_MONITOR" for event in log["events"])
