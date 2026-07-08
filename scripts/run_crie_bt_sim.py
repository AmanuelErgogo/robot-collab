#!/usr/bin/env python
"""Run CRIE-BT against existing RoCoBench MuJoCo task simulators."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Legacy RRT debugging paths may contain ``breakpoint()`` calls on failures.
os.environ.setdefault("PYTHONBREAKPOINT", "0")

from rocobench.crie_bt.controllers import build_controller
from rocobench.crie_bt.legacy_tasks import (
    FakeLegacyUncertaintyReporter,
    LegacyActionPlanner,
    LegacyPromptPlanner,
    LegacyTaskRRTExecutorAdapter,
    agent_names_for_env,
    available_legacy_task_ids,
    legacy_task_spec,
    make_legacy_task_env,
    make_wait_response,
    split_legacy_responses,
    supported_uncertainty_profiles as supported_legacy_uncertainty_profiles,
)
from rocobench.crie_bt.roco_adapters import (
    FakePackGroceryUncertaintyReporter,
    PackGroceryCRIEPlanner,
    PackGroceryRRTExecutorAdapter,
    pack_grocery_task_spec,
    parse_object_targets,
    supported_uncertainty_profiles as supported_pack_uncertainty_profiles,
)
from rocobench.crie_bt.status import ExecutionMode


TASK_GOALS = {
    "pack": "Pack all groceries into the bin.",
    "sort": "Sort the block into the panel assigned by the task.",
    "sweep": "Sweep the cubes into the dustpan.",
    "sandwich": "Assemble the sandwich in the required order.",
    "rope": "Move the rope to the target zone.",
    "cabinet": "Complete the cabinet manipulation task.",
}


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _planner_prompter(planner):
    return getattr(planner, "prompter", None)


def _planner_llm_latencies(planner) -> List[float]:
    prompter = _planner_prompter(planner)
    if prompter is None:
        return []
    return list(getattr(prompter, "_crie_bt_llm_call_latencies_s", []) or [])


def _planner_llm_usage(planner) -> List[Dict[str, Any]]:
    prompter = _planner_prompter(planner)
    if prompter is None:
        return []
    return list(getattr(prompter, "_crie_bt_llm_usage", []) or [])


def _token_total(usages: List[Dict[str, Any]], key: str) -> int:
    total = 0
    for usage in usages:
        value = usage.get(key)
        if value is None:
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _paper_method(mode: str, planner_mode: str) -> str:
    if mode == ExecutionMode.BT_MEDIATED.value and planner_mode == "dialog":
        return "CRIE-BT-Dialog"
    if mode == ExecutionMode.VLM_SARM_MONITOR_PLANNER.value and planner_mode == "dialog":
        return "VLM/SARM-Monitor-Planner-Dialog"
    if mode == ExecutionMode.BT_MEDIATED.value and planner_mode == "chat":
        return "CRIE-BT-Cent"
    if mode == ExecutionMode.VLM_SARM_MONITOR_PLANNER.value and planner_mode == "chat":
        return "VLM/SARM-Monitor-Planner-Cent"
    # Open-loop classifications retained for legacy artifacts but not part of
    # the primary evaluation methods exposed by the runner.
    return mode


def _modes(value: str) -> List[str]:
    if value == "all":
        return [ExecutionMode.DIRECT_FEEDBACK.value, ExecutionMode.BT_MEDIATED.value, ExecutionMode.VLM_SARM_MONITOR_PLANNER.value]
    return [value]


def _tasks(value: str) -> List[str]:
    if value == "all":
        return list(available_legacy_task_ids())
    return [value]


def _build_pack_env(seed: int):
    from rocobench.envs import PackGroceryTask

    env = PackGroceryTask(
        render_cameras=["teaser"],
        randomize_init=False,
        render_point_cloud=False,
    )
    env.seed(np_seed=int(seed))
    env.reset(reload=True)
    return env


def _read_legacy_responses(args) -> List[str]:
    if args.legacy_response_file:
        with open(args.legacy_response_file, "r", encoding="utf-8") as f:
            return split_legacy_responses(f.read())
    if args.legacy_response:
        return split_legacy_responses(args.legacy_response)
    return []


def _select_adapter(task_id: str, args) -> str:
    if args.planner_mode in ("plan", "chat", "dialog"):
        if args.adapter == "typed_pack":
            raise ValueError("--planner-mode {} requires the legacy action-plan adapter.".format(args.planner_mode))
        return "legacy"
    if args.adapter == "typed_pack":
        if task_id != "pack":
            raise ValueError("--adapter typed_pack is only valid for --task pack.")
        return "typed_pack"
    if args.adapter == "legacy":
        return "legacy"
    if task_id == "pack" and not (args.legacy_response or args.legacy_response_file):
        return "typed_pack"
    return "legacy"


def _build_pack_executor(env, args):
    from prompting.parser import LLMResponseParser
    from rocobench.skills import PackGrocerySkillPlanValidator, RRTSkillCompiler, RRTSkillExecutor, build_pack_grocery_skill_registry

    agent_names = list(env.robot_name_map.values())
    registry = build_pack_grocery_skill_registry(agent_names)
    validator = PackGrocerySkillPlanValidator(env, registry, agent_names)
    legacy_parser = LLMResponseParser(
        env,
        "action_only",
        env.robot_name_map,
        ["NAME", "ACTION"],
        use_prepick=env.use_prepick,
        use_preplace=env.use_preplace,
    )
    compiler = RRTSkillCompiler(env, legacy_parser)
    legacy_executor = RRTSkillExecutor(
        env=env,
        robots=env.get_sim_robots(),
        max_sim_steps=int(args.max_sim_steps),
    )
    return PackGroceryRRTExecutorAdapter(
        env=env,
        agent_names=agent_names,
        validator=validator,
        compiler=compiler,
        executor=legacy_executor,
        artifact_dir=args.artifact_dir,
        uncertainty_reporter=FakePackGroceryUncertaintyReporter(args.uncertainty_profile),
    )


def _build_legacy_prompt_planner(env, task_id: str, args):
    # Prompter/planner construction is shared with the pipeline via roco_runtime
    # so there is a single source for the real-LLM wiring.
    from rocobench.crie_bt.roco_runtime import build_legacy_prompt_planner

    save_dir = args.prompt_artifact_dir
    if not save_dir and args.artifact_dir:
        save_dir = os.path.join(args.artifact_dir, "planner_prompts")
    if not save_dir:
        # Default: store prompts alongside the output JSONL so they are easy to find.
        save_dir = os.path.splitext(os.path.abspath(args.output))[0] + "_prompts"
    return build_legacy_prompt_planner(
        env,
        task_id,
        args.planner_mode,
        llm_source=args.llm_source,
        api_key_path=args.api_key_path,
        save_dir=save_dir,
        llm_output_mode=args.llm_output_mode,
        direct_waypoints=args.direct_waypoints,
        max_failed_waypoints=args.max_failed_waypoints,
        max_tokens=args.max_tokens,
        num_replans=args.num_replans,
        temperature=args.temperature,
        max_calls_per_round=args.max_calls_per_round,
        use_history=(not args.no_history),
        use_feedback=(not args.no_feedback),
        plan_horizon=args.plan_horizon,
    )


def _build_legacy_executor(env, task_id: str, args):
    from rocobench.crie_bt.roco_runtime import build_legacy_rrt_executor

    return build_legacy_rrt_executor(
        env,
        task_id,
        max_sim_steps=args.max_sim_steps,
        artifact_dir=args.artifact_dir,
        uncertainty_profile=args.uncertainty_profile,
    )


def _build_env(task_id: str, adapter: str, seed: int):
    if task_id == "pack" and adapter == "typed_pack":
        return _build_pack_env(seed)
    return make_legacy_task_env(task_id, seed=seed)


def _build_planner(env, task_id: str, adapter: str, args):
    if task_id == "pack" and adapter == "typed_pack":
        object_targets = parse_object_targets(args.object_targets)
        return PackGroceryCRIEPlanner(env, object_targets=object_targets, active_agent=args.agent)
    if args.planner_mode in ("plan", "chat", "dialog"):
        return _build_legacy_prompt_planner(env, task_id, args)
    agent_names = agent_names_for_env(env)
    responses = _read_legacy_responses(args) or [make_wait_response(agent_names)]
    return LegacyActionPlanner(task_id, responses=responses, agent_names=agent_names)


def _build_executor(env, task_id: str, adapter: str, args):
    if task_id == "pack" and adapter == "typed_pack":
        return _build_pack_executor(env, args)
    return _build_legacy_executor(env, task_id, args)


def _task_goal(task_id: str, args) -> str:
    return args.task_goal or TASK_GOALS.get(task_id, "Complete the {} RoCoBench task.".format(task_id))


def _describe_obs(env, obs) -> str:
    if obs is not None and hasattr(env, "describe_obs"):
        return env.describe_obs(obs)
    return ""


def _sim_success(env, obs, fallback: bool) -> bool:
    if obs is not None and hasattr(env, "get_reward_done"):
        try:
            return bool(env.get_reward_done(obs)[1])
        except Exception:
            return bool(fallback)
    return bool(fallback)


def _task_spec(env, task_id: str, adapter: str) -> Dict[str, Any]:
    if task_id == "pack" and adapter == "typed_pack":
        return pack_grocery_task_spec(env)
    return legacy_task_spec(env, task_id)


def run(args) -> List[Dict[str, Any]]:
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    with open(output_path, "w", encoding="utf-8") as f:
        for episode in range(int(args.episodes)):
            for task_id in _tasks(args.task):
                for mode in _modes(args.mode):
                    adapter = _select_adapter(task_id, args)
                    env = _build_env(task_id, adapter, int(args.seed) + episode)
                    obs = env.get_obs() if hasattr(env, "get_obs") else None
                    planner = _build_planner(env, task_id, adapter, args)
                    executor = _build_executor(env, task_id, adapter, args)
                    controller = build_controller(
                        mode,
                        planner,
                        executor,
                        uncertainty_mode=args.uncertainty,
                        max_retries=int(args.max_retries),
                    )
                    started = time.perf_counter()
                    row = controller.run_episode(env, _task_goal(task_id, args), int(args.max_steps))
                    wall_time_s = time.perf_counter() - started
                    final_obs = env.get_obs() if hasattr(env, "get_obs") else None
                    llm_usage = _planner_llm_usage(planner)
                    llm_latencies = _planner_llm_latencies(planner)
                    row["episode"] = episode
                    row["task_id"] = task_id
                    row["task_name"] = env.__class__.__name__
                    row["adapter"] = adapter
                    row["planner_mode"] = args.planner_mode
                    row["paper_method"] = _paper_method(mode, args.planner_mode)
                    row["uncertainty_mode"] = args.uncertainty
                    row["uncertainty_profile"] = args.uncertainty_profile
                    row["task_spec"] = _task_spec(env, task_id, adapter)
                    row["initial_scene"] = _describe_obs(env, obs)
                    row["final_scene"] = _describe_obs(env, final_obs)
                    row["sim_success"] = _sim_success(env, final_obs, row.get("success", False))
                    row["wall_time_s"] = round(wall_time_s, 6)
                    row["llm_call_latencies_s"] = llm_latencies
                    row["llm_usage"] = llm_usage
                    row["llm_prompt_tokens"] = _token_total(llm_usage, "prompt_tokens")
                    row["llm_completion_tokens"] = _token_total(llm_usage, "completion_tokens")
                    row["llm_total_tokens"] = _token_total(llm_usage, "total_tokens")
                    row["skipped"] = False
                    f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
                    rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    task_choices = ["all"] + list(available_legacy_task_ids())
    uncertainty_profiles = sorted(set(supported_pack_uncertainty_profiles()).union(supported_legacy_uncertainty_profiles()))
    parser.add_argument("--task", choices=task_choices, default="pack")
    parser.add_argument(
        "--adapter",
        choices=["auto", "typed_pack", "legacy"],
        default="auto",
        help="auto uses the typed PackGrocery adapter for pack and the legacy action-plan adapter otherwise.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["legacy_action", "plan", "chat", "dialog"],
        default="legacy_action",
        help="legacy_action uses provided EXECUTE blocks; plan/chat/dialog use existing RoCoBench LLM prompters.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct_feedback", "bt_mediated", "vlm_sarm_monitor_planner", "all"],
        default="bt_mediated",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/crie_bt/pack_grocery_sim.jsonl")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--agent", default="Alice")
    parser.add_argument("--task-goal", default="")
    parser.add_argument(
        "--legacy-response",
        default="",
        help="One or more raw RoCoBench EXECUTE/NAME/ACTION blocks. Literal \\n sequences are accepted.",
    )
    parser.add_argument(
        "--legacy-response-file",
        default="",
        help="File containing one or more raw RoCoBench EXECUTE/NAME/ACTION blocks for the legacy adapter.",
    )
    parser.add_argument(
        "--object-targets",
        default="",
        help="Typed PackGrocery only: comma-separated object:container overrides.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-sim-steps", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--plan-horizon", type=int, default=1,
                        help="Number of steps to request from the LLM in one call (>1 = multi-step plan).")
    parser.add_argument("--llm-output-mode", choices=["action_only", "action_and_path"], default="action_only")
    parser.add_argument("--llm-source", default="gpt-4")
    parser.add_argument("--api-key-path", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-replans", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-calls-per-round", type=int, default=10)
    parser.add_argument("--direct-waypoints", type=int, default=0)
    parser.add_argument("--max-failed-waypoints", type=int, default=1)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--prompt-artifact-dir", default="")
    parser.add_argument(
        "--uncertainty",
        choices=["none", "heuristic", "ensemble_variance", "policy_metadata"],
        default="policy_metadata",
        help="CRIE-BT uncertainty estimator mode. policy_metadata consumes fake confidence/entropy.",
    )
    parser.add_argument("--uncertainty-profile", choices=uncertainty_profiles, default="nominal")
    args = parser.parse_args(argv)
    rows = run(args)
    successes = sum(1 for row in rows if row.get("success"))
    print(json.dumps({"output": args.output, "episodes": len(rows), "successes": successes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
