import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoEpisodeError
from integrations.lerobot_roco.roco_runtime.episode import EpisodeStateMachine, EpisodeStatus


def test_episode_state_machine_legal_flow():
    sm = EpisodeStateMachine()
    sm.mark_ready()
    episode_id = sm.reset()
    sm.validate_step(episode_id, 0)
    assert sm.advance_step(False) == 1
    sm.validate_step(episode_id, 1)
    sm.advance_step(True)
    assert sm.state == EpisodeStatus.EPISODE_TERMINATED
    sm.close_episode()
    assert sm.state == EpisodeStatus.READY


def test_step_before_reset_is_rejected():
    sm = EpisodeStateMachine()
    sm.mark_ready()
    with pytest.raises(RoCoEpisodeError) as exc:
        sm.validate_step("missing", 0)
    assert exc.value.code == ErrorCode.EPISODE_NOT_ACTIVE


def test_wrong_episode_and_step_are_rejected():
    sm = EpisodeStateMachine()
    sm.mark_ready()
    episode_id = sm.reset()
    with pytest.raises(RoCoEpisodeError) as exc:
        sm.validate_step("wrong", 0)
    assert exc.value.code == ErrorCode.EPISODE_ID_MISMATCH
    with pytest.raises(RoCoEpisodeError) as exc:
        sm.validate_step(episode_id, 1)
    assert exc.value.code == ErrorCode.STEP_INDEX_MISMATCH
