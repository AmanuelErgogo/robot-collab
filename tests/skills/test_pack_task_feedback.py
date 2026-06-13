import pytest

pytest.importorskip("transforms3d")

from rocobench.envs.task_pack import PackGroceryTask


class FakePlan:
    def __init__(self, action_strs):
        self.action_strs = dict(action_strs)


def test_pack_task_feedback_allows_passive_wait():
    plan = FakePlan(
        {
            "Alice": "PICK apple PLACE bin_front_left",
            "Bob": "WAIT",
        }
    )

    assert PackGroceryTask.get_task_feedback(None, plan, {}) == ""


def test_pack_task_feedback_rejects_unknown_action():
    plan = FakePlan({"Alice": "DANCE", "Bob": "WAIT"})

    feedback = PackGroceryTask.get_task_feedback(None, plan, {})

    assert "can only PICK, PLACE, or WAIT" in feedback
