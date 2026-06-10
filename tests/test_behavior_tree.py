import unittest

import numpy as np

from rocobench.behavior_tree import (
    BehaviorContext,
    BehaviorStatus,
    BehaviorTreePlan,
    MotionPrimitive,
)


def _pose(x, y, z):
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _primitive():
    return MotionPrimitive(
        agent_names=["Alice", "Bob"],
        ee_waypoints={
            "Alice": [_pose(0.0, 0.0, 0.2)],
            "Bob": [_pose(0.1, 0.0, 0.2)],
        },
        tograsp={"Alice": None, "Bob": None},
        inhand={"Alice": None, "Bob": None},
        ee_targets={"Alice": _pose(0.0, 0.1, 0.2), "Bob": _pose(0.1, 0.1, 0.2)},
        parsed_proposal="EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION WAIT",
        action_strs={"Alice": "WAIT", "Bob": "WAIT"},
        return_home={"Alice": False, "Bob": False},
    )


class _FakeExecutor:
    def __init__(self):
        self.action_buffer = ["planned-action"]
        self.rrt_plan_results = ["rrt"]
        self.plan_exhausted = False

    def plan(self, env):
        return True, ""


class BehaviorTreeTests(unittest.TestCase):
    def test_plan_wraps_motion_primitives_in_sequence(self):
        primitive_a = _primitive()
        primitive_b = _primitive()
        plan = BehaviorTreePlan.from_primitives([primitive_a, primitive_b], name="pack_lunch")

        self.assertEqual(plan.num_motion_primitives, 2)
        self.assertEqual(plan.get_action_desp(), primitive_a.get_action_desp())
        self.assertEqual(len(list(plan.iter_motion_primitives())), 2)

    def test_motion_primitive_node_activates_executor_then_succeeds(self):
        primitive = _primitive()
        plan = BehaviorTreePlan.from_primitives([primitive])
        created_executors = []

        def executor_factory(_primitive):
            executor = _FakeExecutor()
            created_executors.append(executor)
            return executor

        context = BehaviorContext(
            blackboard={
                "env": object(),
                "primitive_validator": lambda _primitive: (True, ""),
                "executor_factory": executor_factory,
                "planned_leaf_actions": [],
                "planned_leaf_rrt": [],
            }
        )

        first_status = plan.root.tick(context)
        self.assertEqual(first_status, BehaviorStatus.RUNNING)
        self.assertEqual(len(created_executors), 1)
        self.assertIs(context.blackboard["active_executor"], created_executors[0])

        created_executors[0].plan_exhausted = True
        second_status = plan.root.tick(context)
        self.assertEqual(second_status, BehaviorStatus.SUCCESS)

    def test_motion_primitive_node_propagates_validation_failure(self):
        primitive = _primitive()
        plan = BehaviorTreePlan.from_primitives([primitive])
        context = BehaviorContext(
            blackboard={
                "env": object(),
                "primitive_validator": lambda _primitive: (False, "validation failed"),
                "executor_factory": lambda _primitive: _FakeExecutor(),
                "planned_leaf_actions": [],
                "planned_leaf_rrt": [],
            }
        )

        status = plan.root.tick(context)
        self.assertEqual(status, BehaviorStatus.FAILURE)
        self.assertEqual(context.blackboard["last_failure"], "validation failed")


if __name__ == "__main__":
    unittest.main()
