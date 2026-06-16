#!/usr/bin/env python
"""Run a real simulator-backed Phase 6 PackGrocery demo.

This demo uses the actual PackGroceryTask simulator, actual RRT compiler, actual
RRT executor, and actual env.step execution. The high-level planner response is
canned so the demo remains deterministic and does not require a live LLM.

It is intentionally RRT-only by default because the local ACT debug checkpoint
is not a reliable manipulation policy in this workspace. Learned backend wiring
still belongs behind the Phase 5 ``SkillExecutor`` contract.
"""

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Legacy RRT debugging paths contain ``breakpoint()`` calls on some collision
# failures. A demo should fail with structured output, not stop in pdb.
os.environ.setdefault("PYTHONBREAKPOINT", "0")

from prompting.feedback import FeedbackManager
from prompting.parser import LLMResponseParser
from prompting.skill_parser import SkillResponseParser
from rocobench.envs import PackGroceryTask
from rocobench.planning import (
    CannedPlanner,
    ExecutorRoute,
    ReplanningPolicy,
    SequentialPlanningController,
    SkillExecutorRouter,
)
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.skills import (
    PackGrocerySkillPlanValidator,
    RRTSkillCompiler,
    RRTSkillExecutor,
    build_pack_grocery_skill_registry,
)


VALID_PLAN = """EXECUTE
NAME Alice ACTION PUT_OBJECT_IN_CONTAINER(object=apple, container=bin_front_left)
NAME Bob ACTION WAIT()
"""

ALL_WAIT_PLAN = """EXECUTE
NAME Alice ACTION WAIT()
NAME Bob ACTION WAIT()
"""


def _json_safe(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_real_controller(env, max_plan_rounds):
    obs = env.get_obs()
    agent_names = list(env.robot_name_map.values())
    registry = build_pack_grocery_skill_registry(agent_names)
    parser = SkillResponseParser(registry, agent_names)
    validator = PackGrocerySkillPlanValidator(env, registry, agent_names)
    legacy_parser = LLMResponseParser(
        env,
        "action_only",
        env.robot_name_map,
        ["NAME", "ACTION"],
        use_prepick=env.use_prepick,
        use_preplace=env.use_preplace,
    )
    rrt_planner = MultiArmRRT(
        env.physics,
        robots=env.get_sim_robots(),
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    geometric_feedback = FeedbackManager(
        env=env,
        planner=rrt_planner,
        llm_output_mode="action_only",
        robot_name_map=env.robot_name_map,
        step_std_threshold=env.waypoint_std_threshold,
    )
    rrt_compiler = RRTSkillCompiler(env, legacy_parser)
    router = SkillExecutorRouter(
        routes=[
            ExecutorRoute(
                agent="Alice",
                skill="PUT_OBJECT_IN_CONTAINER",
                primary="rrt",
                fallback="unsupported",
            ),
            ExecutorRoute(
                agent="Bob",
                skill="PUT_OBJECT_IN_CONTAINER",
                primary="rrt",
                fallback="unsupported",
            ),
        ],
        default_backend="rrt",
    )
    controller = SequentialPlanningController(
        env=env,
        agent_names=agent_names,
        parser=parser,
        validator=validator,
        router=router,
        policy=ReplanningPolicy(max_plan_rounds=max_plan_rounds, max_fallbacks=0),
        executors={
            "rrt": RRTSkillExecutor(
                env=env,
                robots=env.get_sim_robots(),
                max_sim_steps=5000,
            )
        },
        rrt_compiler=rrt_compiler,
        registry=registry,
    )
    return controller, obs, geometric_feedback


def _execution_success(_env):
    def _check(obs, result):
        del obs
        return bool(result.success)

    return _check


def _skill_postcondition(env, object_name, target_slot):
    def _check(obs, result):
        if not result.success:
            return False
        current_obs = env.get_obs() if hasattr(env, "get_obs") else obs
        return env.get_packed_slot_for_object(current_obs, object_name) == target_slot

    return _check


def _write_artifacts(output_dir, result, scenario, initial_obs, final_obs):
    os.makedirs(output_dir, exist_ok=True)
    result.events.write_jsonl(os.path.join(output_dir, "events.jsonl"))
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(result.to_dict()), f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "feedback.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(result.feedback_history))
        if result.feedback_history:
            f.write("\n")
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "success": bool(result.success),
                "reason": result.reason,
                "event_types": result.events.event_types(),
                "metrics": result.metrics.to_dict(),
                "initial_has_objects": hasattr(initial_obs, "objects"),
                "final_has_objects": hasattr(final_obs, "objects"),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def run_demo(args):
    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    env = PackGroceryTask(
        render_cameras=["teaser"],
        randomize_init=False,
        render_point_cloud=False,
    )
    env.seed(np_seed=int(args.seed))
    env.reset(reload=True)
    controller, initial_obs, geometric_feedback = _build_real_controller(env, args.max_plan_rounds)
    del geometric_feedback  # The controller compiles and executes RRT directly.

    responses = [VALID_PLAN]
    if args.scenario == "invalid_then_success":
        responses = [ALL_WAIT_PLAN, VALID_PLAN]
    elif args.scenario == "budget_exhaustion":
        responses = [ALL_WAIT_PLAN]

    planner = CannedPlanner(responses)
    if args.success_check == "postcondition":
        task_success_fn = _skill_postcondition(env, "apple", "bin_front_left")
    else:
        task_success_fn = _execution_success(env)
    result = controller.run(
        planner,
        initial_obs=initial_obs,
        run_id="phase6_real_pack_{}".format(args.scenario),
        task_success_fn=task_success_fn,
        artifact_dir=os.path.join(args.output_dir, "executor_artifacts"),
    )
    final_obs = env.get_obs()
    packed_slot = env.get_packed_slot_for_object(final_obs, "apple")
    postcondition_met = packed_slot == "bin_front_left"
    _write_artifacts(args.output_dir, result, args.scenario, initial_obs, final_obs)

    summary = {
        "scenario": args.scenario,
        "success": bool(result.success),
        "success_check": args.success_check,
        "reason": result.reason,
        "apple_packed_slot": packed_slot,
        "apple_postcondition_met": bool(postcondition_met),
        "event_types": result.events.event_types(),
        "metrics": result.metrics.to_dict(),
        "artifact_dir": args.output_dir,
    }
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.scenario == "budget_exhaustion":
        return 0 if not result.success and result.events.event_types()[-1] == "BUDGET_EXHAUSTED" else 1
    if args.success_check == "postcondition":
        return 0 if result.success and postcondition_met else 1
    return 0 if result.success else 1


def main():
    parser = argparse.ArgumentParser(description="Run real simulator-backed Phase 6 PackGrocery demo.")
    parser.add_argument(
        "--scenario",
        choices=["success", "invalid_then_success", "budget_exhaustion"],
        default="success",
    )
    parser.add_argument("--output-dir", default="artifacts/planning/phase6_real_pack")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plan-rounds", type=int, default=3)
    parser.add_argument("--success-check", choices=["execution", "postcondition"], default="execution")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
