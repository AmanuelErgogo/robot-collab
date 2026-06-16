#!/usr/bin/env python
"""Run canned Phase 6 planner-integration demos without a live LLM.

The demo uses the real Phase 6 controller, skill parser, registry, validator,
router, budget, event log, feedback renderer, and recovery engine. It supplies a
small fake PackGrocery-like environment and fake executors so the scenarios are
fast, deterministic, and runnable in both the RoCo Python 3.8 environment and a
plain development environment.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prompting.skill_parser import SkillResponseParser
from rocobench.planning import (
    CannedPlanner,
    ExecutorRoute,
    ReplanningPolicy,
    SequentialPlanningController,
    SkillExecutorRouter,
)
from rocobench.skills import PackGrocerySkillPlanValidator, build_pack_grocery_skill_registry
from rocobench.skills.models import PreparedSkillExecution, SkillExecutionResult, SkillExecutionStatus


PLAN_APPLE_LEFT = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
"""

PLAN_APPLE_RIGHT = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_right)
NAME Bob ACTION WAIT()
"""

PLAN_ALL_WAIT = """EXECUTE
NAME Alice ACTION WAIT()
NAME Bob ACTION WAIT()
"""


class ObjectState(object):
    def __init__(self, name, xpos):
        self.name = name
        self.xpos = xpos
        self.contacts = set()


class RobotState(object):
    def __init__(self, contacts=None):
        self.contacts = set(contacts or [])


class DemoObservation(object):
    def __init__(self, positions, held):
        self.objects = {
            name: ObjectState(name, xpos)
            for name, xpos in positions.items()
        }
        self.ur5e_robotiq = RobotState([held["Alice"]] if held.get("Alice") else [])
        self.panda = RobotState([held["Bob"]] if held.get("Bob") else [])


class DemoPackEnv(object):
    item_names = ["apple", "banana", "milk"]
    bin_slot_xposes = {
        "bin_front_left": (0.0, 0.0, 0.0),
        "bin_front_right": (1.0, 0.0, 0.0),
        "bin_back_left": (0.0, 1.0, 0.0),
    }
    robot_name_map = {"ur5e_robotiq": "Alice", "panda": "Bob"}
    robot_name_map_inv = {"Alice": "ur5e_robotiq", "Bob": "panda"}

    def __init__(self):
        self.positions = {
            "apple": (0.0, 0.0, 0.1),
            "banana": (0.2, 0.0, 0.1),
            "milk": (0.4, 0.0, 0.1),
        }
        self.held = {"Alice": None, "Bob": None}
        self.occupancy = {slot_name: None for slot_name in self.bin_slot_xposes}
        self.restore_count = 0

    def get_obs(self):
        return DemoObservation(dict(self.positions), dict(self.held))

    def describe_obs(self, obs):
        del obs
        return "demo pack scene"

    def get_agent_held_object(self, obs, agent_name):
        del obs
        return self.held.get(agent_name)

    def get_slot_occupancy(self, obs):
        del obs
        return dict(self.occupancy)

    def get_packed_slot_for_object(self, obs, object_name):
        del obs
        for slot_name, occupant in self.occupancy.items():
            if occupant == object_name:
                return slot_name
        return None

    def snapshot(self):
        return {
            "positions": dict(self.positions),
            "held": dict(self.held),
            "occupancy": dict(self.occupancy),
        }

    def restore(self, snapshot):
        self.positions = dict(snapshot["positions"])
        self.held = dict(snapshot["held"])
        self.occupancy = dict(snapshot["occupancy"])
        self.restore_count += 1

    def get_state_digest(self):
        payload = {
            "positions": self.positions,
            "held": self.held,
            "occupancy": self.occupancy,
        }
        return json.dumps(payload, sort_keys=True)

    def place_from_plan(self, plan):
        for call in plan.calls:
            if call.skill_name != "WAIT":
                obj = call.arguments["object"]
                target = call.arguments["container"]
                self.occupancy[target] = obj
                self.positions[obj] = self.bin_slot_xposes[target]


class DemoRRTCompiler(object):
    def __init__(self):
        self.compile_count = 0

    def compile(self, plan, obs):
        del obs
        self.compile_count += 1
        return PreparedSkillExecution("rrt", plan.plan_id, ["compiled-demo-path-plan"])


class DemoExecutor(object):
    def __init__(self, env, outcomes):
        self.env = env
        self.outcomes = list(outcomes)
        self.calls = []

    def execute(self, plan, obs, artifact_dir=None):
        self.calls.append({"plan_id": plan.plan_id, "artifact_dir": artifact_dir})
        del obs
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = {"success": True, "is_success": True}
        mutate = outcome.get("mutate")
        if mutate:
            mutate(self.env)
        if outcome.get("success"):
            self.env.place_from_plan(plan)
            return SkillExecutionResult(
                success=True,
                status=SkillExecutionStatus.SUCCESS,
                reason="",
                num_sim_steps=int(outcome.get("steps", 1)),
                reward=1.0 if outcome.get("is_success", False) else 0.0,
                done=bool(outcome.get("is_success", False)),
                info={"is_success": bool(outcome.get("is_success", False))},
                metadata={"learned_success": bool(outcome.get("learned_success", False))},
            )
        return SkillExecutionResult(
            success=False,
            status=outcome.get("status", SkillExecutionStatus.EXECUTION_FAILED),
            reason=outcome.get("reason", outcome.get("failure_code", "failed")),
            num_sim_steps=int(outcome.get("steps", 1)),
            reward=0.0,
            done=False,
            info={"is_success": False},
            metadata={
                "failure_code": outcome.get("failure_code"),
                "progress_stage": outcome.get("progress_stage", "not_started"),
            },
        )


def corrupt_apple_position(env):
    env.positions["apple"] = (5.0, 5.0, 0.1)


def slipped_apple_position(env):
    env.positions["apple"] = (9.0, 9.0, 0.1)


def scenario_definition(name):
    if name == "success":
        return {
            "planner_responses": [PLAN_APPLE_LEFT],
            "learned_outcomes": [{"success": True, "is_success": True, "learned_success": True}],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=4),
        }
    if name == "invalid_replan":
        return {
            "preload": lambda env: env.occupancy.update({"bin_front_left": "banana"}),
            "planner_responses": [PLAN_APPLE_LEFT, PLAN_APPLE_RIGHT],
            "learned_outcomes": [{"success": True, "is_success": True, "learned_success": True}],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=3),
        }
    if name == "slippage_replan":
        return {
            "planner_responses": [PLAN_APPLE_LEFT, PLAN_APPLE_RIGHT],
            "learned_outcomes": [
                {"success": False, "failure_code": "SLIPPAGE", "reason": "demo slippage", "progress_stage": "transporting", "mutate": slipped_apple_position},
                {"success": True, "is_success": True, "learned_success": True},
            ],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=4),
        }
    if name == "inference_fallback":
        return {
            "planner_responses": [PLAN_APPLE_LEFT],
            "learned_outcomes": [
                {"success": False, "failure_code": "POLICY_INFERENCE_FAILURE", "reason": "demo inference failure", "mutate": corrupt_apple_position}
            ],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=3, max_fallbacks=1),
        }
    if name == "repeated_loop":
        return {
            "planner_responses": [PLAN_APPLE_LEFT],
            "learned_outcomes": [
                {"success": False, "failure_code": "MISSED_GRASP", "reason": "demo missed grasp"},
                {"success": False, "failure_code": "MISSED_GRASP", "reason": "demo missed grasp again"},
            ],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=3, max_retries_per_skill=3, max_repeated_failures=1, max_fallbacks=1),
        }
    if name == "budget_exhaustion":
        return {
            "planner_responses": [PLAN_ALL_WAIT],
            "learned_outcomes": [{"success": True, "is_success": True, "learned_success": True}],
            "rrt_outcomes": [{"success": True, "is_success": True}],
            "policy": ReplanningPolicy(max_plan_rounds=1),
        }
    raise ValueError("Unknown scenario: {}".format(name))


def build_controller(env, definition):
    agent_names = ["Alice", "Bob"]
    registry = build_pack_grocery_skill_registry(agent_names)
    parser = SkillResponseParser(registry, agent_names)
    validator = PackGrocerySkillPlanValidator(env, registry, agent_names)
    router = SkillExecutorRouter(
        routes=[
            ExecutorRoute(
                agent="Alice",
                skill="PUT_OBJECT_IN_CONTAINER",
                primary="learned",
                fallback="rrt",
                policy_id="demo_internal_policy_id",
            ),
            ExecutorRoute(
                agent="Bob",
                skill="PUT_OBJECT_IN_CONTAINER",
                primary="rrt",
            ),
        ],
        default_backend="rrt",
    )
    compiler = DemoRRTCompiler()
    learned = DemoExecutor(env, definition["learned_outcomes"])
    rrt = DemoExecutor(env, definition["rrt_outcomes"])
    controller = SequentialPlanningController(
        env=env,
        agent_names=agent_names,
        parser=parser,
        validator=validator,
        router=router,
        policy=definition["policy"],
        executors={"learned": learned, "rrt": rrt},
        rrt_compiler=compiler,
        registry=registry,
    )
    return controller, learned, rrt, compiler


def write_demo_artifacts(output_dir, scenario, result, planner):
    scenario_dir = os.path.join(output_dir, scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    result.events.write_jsonl(os.path.join(scenario_dir, "events.jsonl"))
    with open(os.path.join(scenario_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    with open(os.path.join(scenario_dir, "feedback.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(result.feedback_history))
        if result.feedback_history:
            f.write("\n")
    with open(os.path.join(scenario_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for index, prompt in enumerate(planner.prompts):
            f.write("=== Prompt {} ===\n{}\n\n".format(index, prompt))
    return scenario_dir


def run_scenario(name, output_dir):
    definition = scenario_definition(name)
    env = DemoPackEnv()
    preload = definition.get("preload")
    if preload:
        preload(env)
    controller, learned, rrt, compiler = build_controller(env, definition)
    planner = CannedPlanner(definition["planner_responses"])
    result = controller.run(planner, initial_obs=env.get_obs(), run_id="phase6_demo_{}".format(name), artifact_dir=os.path.join(output_dir, name, "executor_artifacts"))
    scenario_dir = write_demo_artifacts(output_dir, name, result, planner)
    return {
        "scenario": name,
        "success": bool(result.success),
        "reason": result.reason,
        "event_types": result.events.event_types(),
        "metrics": result.metrics.to_dict(),
        "budget": result.budget.to_dict(),
        "feedback_count": len(result.feedback_history),
        "learned_calls": len(learned.calls),
        "rrt_calls": len(rrt.calls),
        "rrt_compiles": compiler.compile_count,
        "restore_count": env.restore_count,
        "artifact_dir": scenario_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run canned Phase 6 planner demos.")
    parser.add_argument(
        "--scenario",
        choices=["success", "invalid_replan", "slippage_replan", "inference_fallback", "repeated_loop", "budget_exhaustion", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="artifacts/planning/phase6_demo")
    args = parser.parse_args()

    scenarios = [args.scenario]
    if args.scenario == "all":
        scenarios = ["success", "invalid_replan", "slippage_replan", "inference_fallback", "repeated_loop", "budget_exhaustion"]

    summaries = []
    for scenario in scenarios:
        summary = run_scenario(scenario, args.output_dir)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    overall = all(item["success"] for item in summaries if item["scenario"] != "budget_exhaustion")
    budget_exhaustion_ok = all(item["success"] is False for item in summaries if item["scenario"] == "budget_exhaustion")
    return 0 if overall and budget_exhaustion_ok else 1


if __name__ == "__main__":
    sys.exit(main())
