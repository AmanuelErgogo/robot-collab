import numpy as np
import pytest

from integrations.lerobot_roco.evaluation.action_queue import ACTActionQueue, ActionQueueError


def test_action_queue_executes_only_horizon_and_resets():
    queue = ACTActionQueue(chunk_size=4, execution_horizon=2)
    chunk = np.arange(8, dtype=np.float32).reshape(4, 2)

    chunk_id = queue.load_chunk(chunk)

    assert chunk_id == 0
    assert len(queue) == 2
    first = queue.pop()
    second = queue.pop()
    assert first.chunk_id == 0
    assert first.chunk_offset == 0
    assert second.chunk_offset == 1
    assert len(queue) == 0

    queue.load_chunk(chunk + 10)
    assert queue.pop().chunk_id == 1
    queue.reset()
    assert len(queue) == 0
    assert queue.next_chunk_id == 0


def test_action_queue_rejects_invalid_horizon_and_chunk_shape():
    with pytest.raises(ActionQueueError):
        ACTActionQueue(chunk_size=4, execution_horizon=5)

    queue = ACTActionQueue(chunk_size=4, execution_horizon=1)
    with pytest.raises(ActionQueueError):
        queue.load_chunk(np.zeros((3, 2), dtype=np.float32))

