#!/usr/bin/env python
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prompting.skill_feedback import SkillFeedbackManager
from prompting.skill_parser import SkillResponseParser
from rocobench.skills.compiler import RRTSkillCompiler
from rocobench.skills.models import SkillExecutionStatus
from rocobench.skills.pack_grocery import build_pack_grocery_skill_registry
from rocobench.skills.validation import PackGrocerySkillPlanValidator


CANNED_RESPONSE = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
"""


class ObjectState:
    def __init__(self, name):
        self.name = name
        self.contacts = set()
        self.xpos = (0.0, 0.0, 0.0)


class RobotState:
    def __init__(self, contacts=None):
        self.contacts = set(contacts or [])


class FakeObs:
    def __init__(self):
        self.objects = {
            "apple": ObjectState("apple"),
            "banana": ObjectState("banana"),
            "milk": ObjectState("milk"),
        }
        self.ur5e_robotiq = RobotState()
        self.panda = RobotState()


class FakeEnv:
    item_names = ["apple", "banana", "milk"]
    bin_slot_xposes = {
        "bin_front_left": (0.0, 0.0, 0.0),
        "bin_front_right": (1.0, 0.0, 0.0),
    }
    robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
    robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}

    def get_agent_held_object(self, obs, agent_name):
        robot_state = getattr(obs, self.robot_name_map_inv[agent_name])
        for item_name in self.item_names:
            if item_name in robot_state.contacts:
                return item_name
        return None

    def get_slot_occupancy(self, obs):
        return {slot_name: None for slot_name in self.bin_slot_xposes}

    def get_packed_slot_for_object(self, obs, object_name):
        return None


class FakeLegacyParser:
    def parse(self, obs, response):
        return True, "parsed", ["compiled-path-plan"]


class FakeGeometryFeedback:
    def give_feedback(self, plan):
        return True, "[Environment Feedback]: fake geometric checks passed."


def run_fake_smoke():
    env = FakeEnv()
    obs = FakeObs()
    agent_names = ["Alice", "Bob"]
    registry = build_pack_grocery_skill_registry(agent_names)
    parser = SkillResponseParser(registry, agent_names)
    success, message, plans = parser.parse(obs, CANNED_RESPONSE)
    if not success:
        print("parse failed:", message)
        return 1

    plan = plans[0]
    validator = PackGrocerySkillPlanValidator(env, registry, agent_names)
    compiler = RRTSkillCompiler(env, FakeLegacyParser())
    manager = SkillFeedbackManager(validator, compiler, FakeGeometryFeedback())
    manager.update_obs(obs)
    ready, feedback = manager.give_feedback(plan)

    print("Canonical skill plan:")
    print(plan.get_action_desp())
    print("\nSemantic/geometric feedback:")
    print(feedback)
    if not ready:
        return 1
    print("\nSynthetic legacy plan:")
    print(plan.prepared_execution.metadata["synthetic_response"])
    return 0


def run_real_execute():
    try:
        from prompting.feedback import FeedbackManager
        from prompting.parser import LLMResponseParser
        from rocobench.envs import PackGroceryTask
        from rocobench.rrt_multi_arm import MultiArmRRT
        from rocobench.skills.executor import RRTSkillExecutor
    except Exception as exc:
        print("Cannot run --execute because simulator dependencies are unavailable:", exc)
        return 1

    env = PackGroceryTask(render_cameras=["teaser"], randomize_init=False, render_point_cloud=False)
    obs = env.get_obs()
    robots = env.get_sim_robots()
    agent_names = list(env.robot_name_map.values())
    registry = build_pack_grocery_skill_registry(agent_names)
    parser = SkillResponseParser(registry, agent_names)
    success, message, plans = parser.parse(obs, CANNED_RESPONSE)
    if not success:
        print("parse failed:", message)
        return 1

    legacy_parser = LLMResponseParser(env, "action_only", env.robot_name_map, ["NAME", "ACTION"])
    planner = MultiArmRRT(
        env.physics,
        robots=robots,
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    manager = SkillFeedbackManager(
        PackGrocerySkillPlanValidator(env, registry, agent_names),
        RRTSkillCompiler(env, legacy_parser),
        FeedbackManager(env, planner, "action_only", env.robot_name_map),
    )
    manager.update_obs(obs)
    ready, feedback = manager.give_feedback(plans[0])
    print(feedback)
    if not ready:
        return 1

    executor = RRTSkillExecutor(env, robots)
    result = executor.execute(plans[0], obs)
    print("Execution status:", result.status.value)
    print("Simulation steps:", result.num_sim_steps)
    return 0 if result.status == SkillExecutionStatus.SUCCESS else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run the real RRT-backed simulator path.")
    args = parser.parse_args()
    if args.execute:
        return run_real_execute()
    return run_fake_smoke()


if __name__ == "__main__":
    sys.exit(main())
