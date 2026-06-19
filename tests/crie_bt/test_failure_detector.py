from rocobench.crie_bt.failure import FailureDetector
from rocobench.crie_bt.status import FailureCode, ProgressStage
from rocobench.crie_bt.types import ProgressState, SkillCall, UncertaintyState


def test_failure_detector_reports_no_progress():
    skill = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin"})
    failure = FailureDetector(no_progress_patience=2).detect(
        skill,
        ProgressState(stage=ProgressStage.STUCK, stagnant_steps=2),
        UncertaintyState(),
        {},
    )
    assert failure.is_failure
    assert failure.failure_code == FailureCode.NO_PROGRESS


def test_failure_detector_reports_low_confidence():
    skill = SkillCall("Alice", "WAIT")
    failure = FailureDetector(low_confidence_threshold=0.7).detect(
        skill,
        ProgressState(stage=ProgressStage.TRANSPORTING, score=0.5),
        UncertaintyState(confidence=0.2, uncertainty=0.8, risk_level="high"),
        {},
    )
    assert failure.failure_code == FailureCode.LOW_CONFIDENCE
