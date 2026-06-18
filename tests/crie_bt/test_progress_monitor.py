from rocobench.crie_bt.progress import ProgressMonitor
from rocobench.crie_bt.status import ProgressStage
from rocobench.crie_bt.types import SkillCall


def test_progress_success_marks_subtask_complete():
    obs = {
        "objects": {"apple": {"position": [0.8, 0.0, 0.1]}},
        "targets": {"bin": {"position": [0.8, 0.0, 0.1]}},
        "agents": {"Alice": {"gripper_position": [0.8, 0.0, 0.2]}},
        "packed": {"apple": "bin"},
    }
    skill = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "apple", "container": "bin"})
    monitor = ProgressMonitor(env=obs)
    monitor.reset(skill, obs)
    progress = monitor.update(skill, obs)
    assert progress.stage == ProgressStage.STABLE_SUCCESS
    assert progress.postcondition_satisfied


def test_progress_monitor_handles_missing_fields_gracefully():
    skill = SkillCall("Alice", "PUT_OBJECT_IN_CONTAINER", {"object": "missing", "container": "bin"})
    monitor = ProgressMonitor(no_progress_patience=1)
    monitor.reset(skill, {})
    progress = monitor.update(skill, {})
    assert "object_position" in progress.evidence
