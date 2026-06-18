from rocobench.crie_bt.status import BTStatus, FailureCode, ProgressStage
from rocobench.crie_bt.types import ExecutionFeedback, FailureState, ProgressState, SkillCall
from rocobench.crie_bt.uncertainty import UncertaintyEstimator


def test_uncertainty_none_is_neutral():
    skill = SkillCall("Alice", "WAIT")
    state = UncertaintyEstimator("none").estimate(skill, {})
    assert state.confidence == 1.0
    assert state.uncertainty == 0.0


def test_uncertainty_heuristic_increases_on_failure():
    skill = SkillCall("Alice", "WAIT")
    feedback = ExecutionFeedback(
        skill,
        status=BTStatus.FAILURE,
        progress=ProgressState(stage=ProgressStage.STUCK, stagnant_steps=3),
        failure=FailureState(True, FailureCode.NO_PROGRESS, "warning", "stuck", {}),
    )
    state = UncertaintyEstimator("heuristic").estimate(skill, {}, feedback)
    assert state.risk_level == "high"


def test_uncertainty_policy_metadata_uses_confidence():
    skill = SkillCall("Alice", "WAIT")
    feedback = ExecutionFeedback(skill, raw_info={"confidence": 0.25})
    state = UncertaintyEstimator("policy_metadata").estimate(skill, {}, feedback)
    assert state.confidence == 0.25
    assert state.risk_level == "high"
