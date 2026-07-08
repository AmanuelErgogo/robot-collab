"""Fairness gate and monitor tests (plan tests 5, 8)."""

import pytest

from rocobench.crie_bt.pipeline import (
    CodedSimProgressMonitor,
    ExecutionFeedback,
    MonitorDecision,
    ObservationBundle,
    SARMProgressMonitor,
    SkillCall,
    StageStatus,
    planner_safe_observation,
)


def test_planner_safe_observation_strips_oracle_state():
    obs = ObservationBundle(
        rgb={"cam": "img"},
        public_percepts={"scene": "two bottles"},
        oracle_state={"object_pose": [0, 1, 2], "task_done": False},
    )
    safe = planner_safe_observation(obs)
    assert safe.oracle_state is None
    # Non-privileged channels survive.
    assert safe.public_percepts == {"scene": "two bottles"}
    assert safe.rgb == {"cam": "img"}
    # The original bundle is not mutated (monitor/evaluator still need it).
    assert obs.oracle_state is not None


def test_monitor_decision_clamps_progress_score():
    assert MonitorDecision("s", StageStatus.IN_PROGRESS, 5.0).progress_score == 1.0
    assert MonitorDecision("s", StageStatus.IN_PROGRESS, -3.0).progress_score == 0.0
    assert MonitorDecision("s", StageStatus.IN_PROGRESS, float("nan")).progress_score == 0.0
    assert MonitorDecision("s", StageStatus.IN_PROGRESS, 0.42).progress_score == 0.42


def _feedback(status, stage_id="grasp"):
    return ExecutionFeedback(
        agent_id="robot_1",
        skill_call=SkillCall("robot_1", "pick_place", {}, stage_id=stage_id),
        status=status,
        raw_info={"stage_id": stage_id},
    )


def test_coded_monitor_progress_score_between_0_and_1():
    monitor = CodedSimProgressMonitor()
    for status in ("running", "success", "failure", "timeout"):
        obs = ObservationBundle(oracle_state={"stage_id": "grasp", "stage_progress": 0.3})
        monitor.reset_stage("grasp", None, obs)
        decision = monitor.update(obs, _feedback(status), history=[])
        assert 0.0 <= decision.progress_score <= 1.0
        assert decision.monitor_backend == "CodedSim"
        assert decision.privileged is True


def test_coded_monitor_reports_failure_and_task_done():
    monitor = CodedSimProgressMonitor()
    obs = ObservationBundle(oracle_state={"stage_id": "grasp", "task_done": False})
    monitor.reset_stage("grasp", None, obs)
    assert monitor.update(obs, _feedback("failure"), []).status == StageStatus.FAILED

    done_obs = ObservationBundle(oracle_state={"stage_id": "grasp", "task_done": True})
    assert monitor.update(done_obs, _feedback("success"), []).is_task_done is True


def _feedback_pc(status, postcondition, stage_id="grasp"):
    return ExecutionFeedback(
        agent_id="robot_1",
        skill_call=SkillCall("robot_1", "pick_place", {}, stage_id=stage_id),
        status=status,
        raw_info={"stage_id": stage_id, "postcondition_satisfied": postcondition},
    )


def test_coded_monitor_honours_postcondition_signal():
    # Merged from the legacy FailureDetector: a completed motion whose postcondition
    # is unmet must NOT be counted as stage-done (this is what the plan's privileged
    # coded monitor is supposed to check).
    monitor = CodedSimProgressMonitor()
    obs = ObservationBundle(oracle_state={"stage_id": "grasp", "task_done": False})

    monitor.reset_stage("grasp", None, obs)
    unmet = monitor.update(obs, _feedback_pc("success", postcondition=False), [])
    assert unmet.status != StageStatus.STAGE_DONE
    assert unmet.is_stage_done is False

    monitor.reset_stage("grasp", None, obs)
    met = monitor.update(obs, _feedback_pc("success", postcondition=True), [])
    assert met.status == StageStatus.STAGE_DONE
    assert met.is_stage_done is True


def test_coded_monitor_success_is_stage_done_when_postcondition_absent():
    # Synthetic backend has no postcondition key -> a success still counts as done
    # (backward compatible; the synthetic executor only reports success on real advance).
    monitor = CodedSimProgressMonitor()
    obs = ObservationBundle(oracle_state={"stage_id": "grasp", "task_done": False})
    monitor.reset_stage("grasp", None, obs)
    decision = monitor.update(obs, _feedback("success"), [])
    assert decision.status == StageStatus.STAGE_DONE


def test_sarm_monitor_requires_scorer_but_scores_in_range():
    with pytest.raises(NotImplementedError):
        SARMProgressMonitor().update(ObservationBundle(), None, [])

    monitor = SARMProgressMonitor(scorer=lambda stage_id, obs: 2.0)  # deliberately out of range
    monitor.reset_stage("grasp_bottle", None, ObservationBundle())
    decision = monitor.update(ObservationBundle(), _feedback("running"), [])
    assert 0.0 <= decision.progress_score <= 1.0
    assert decision.monitor_backend == "SARM"
    assert decision.privileged is False
