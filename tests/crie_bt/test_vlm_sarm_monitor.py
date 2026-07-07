from rocobench.crie_bt.status import BTStatus, FailureCode, ProgressStage
from rocobench.crie_bt.types import ExecutionFeedback, FailureState, ProgressState, SkillCall, UncertaintyState
from rocobench.crie_bt.vlm_sarm_monitor import SimulatorSignalVLMSARMMonitor


class DoneEnv:
    def __init__(self, done):
        self.done = done

    def get_reward_done(self, obs):
        return 1.0 if self.done else 0.0, self.done


def _feedback(status=BTStatus.SUCCESS, failure=None):
    return ExecutionFeedback(
        skill_call=SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin"}),
        status=status,
        progress=ProgressState(stage=ProgressStage.STABLE_SUCCESS),
        uncertainty=UncertaintyState(confidence=1.0, uncertainty=0.0, risk_level="low"),
        failure=failure or FailureState(False, FailureCode.NONE),
        message="done",
    )


def test_simulator_monitor_reports_done_from_env():
    decision = SimulatorSignalVLMSARMMonitor().evaluate(DoneEnv(True), {}, feedback=_feedback())
    assert decision.is_done
    assert decision.status == "DONE"
    assert not decision.should_replan


def test_simulator_monitor_reports_failed_from_executor_feedback():
    failure = FailureState(True, FailureCode.MISSED_GRASP, "error", "missed")
    decision = SimulatorSignalVLMSARMMonitor().evaluate(
        DoneEnv(False),
        {},
        feedback=_feedback(status=BTStatus.FAILURE, failure=failure),
    )
    assert decision.is_failed
    assert decision.status == "FAILED"
    assert decision.should_replan
    assert decision.evidence["failure_code"] == "MISSED_GRASP"
