#!/usr/bin/env python
"""Run a real simulator-backed Phase 7 synchronized stepping demo.

The demo uses the actual PackGroceryTask simulator, actual Phase 7 scheduler,
and actual env.step calls. It intentionally uses hold-position action providers
instead of learned multi-agent policies, because learned concurrent manipulation
is still feature-gated and should not be claimed as passing from this smoke.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("PYTHONBREAKPOINT", "0")

from rocobench.envs import PackGroceryTask
from rocobench.multi_agent import (
    AgentActionFragment,
    MultiAgentScheduler,
    SchedulerConfig,
    SynchronizedExecutor,
)
from rocobench.skills.models import SkillCall, SkillPlan
from rocobench.skills.pack_grocery import PUT_OBJECT_IN_CONTAINER


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _put(agent_name, object_name, container_name):
    raw = "PUT_OBJECT_IN_CONTAINER(object={}, container={})".format(object_name, container_name)
    return SkillCall(
        agent_name=agent_name,
        skill_name=PUT_OBJECT_IN_CONTAINER,
        arguments={"object": object_name, "container": container_name},
        raw_action=raw,
    )


def _build_demo_plan():
    return SkillPlan(
        [
            _put("Alice", "apple", "bin_front_left"),
            _put("Bob", "banana", "bin_front_right"),
        ],
        "EXECUTE",
    )


class RealHoldProvider(object):
    """Return current robot joint state as a hold-position action fragment."""

    def __init__(self, env):
        self.env = env
        self.reset_count = 0
        self.calls = 0

    def reset(self):
        self.reset_count += 1

    def next_action(self, obs, agent_name):
        del obs
        self.calls += 1
        robot = self.env.robots[agent_name]
        qpos = self.env.physics.data.qpos
        ctrl = self.env.physics.data.ctrl
        ctrl_idxs = list(robot.joint_idxs_in_ctrl) + [robot.grasp_idx]
        ctrl_vals = [qpos[idx] for idx in robot.joint_idxs_in_qpos] + [ctrl[robot.grasp_idx]]
        return AgentActionFragment(
            agent_name=agent_name,
            ctrl_idxs=np.asarray(ctrl_idxs, dtype=np.int32),
            ctrl_vals=np.asarray(ctrl_vals, dtype=np.float32),
            qpos_idxs=np.asarray(robot.joint_idxs_in_qpos, dtype=np.int32),
            qpos_target=np.asarray(qpos[robot.joint_idxs_in_qpos], dtype=np.float32),
            metadata={"provider": "real_hold"},
        )


def _write_artifacts(output_dir, result, initial_qpos, final_qpos, provider_stats, ticks):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(result.to_dict()), f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "schedule.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(result.schedule.to_dict()), f, indent=2, sort_keys=True)
    summary = {
        "demo": "phase7_real_hold",
        "real_simulator": True,
        "learned_multi_agent_policy": False,
        "ticks_requested": int(ticks),
        "success": bool(result.success),
        "central_status": result.central_status,
        "central_steps": int(result.metrics.central_steps),
        "stop_all_events": int(result.metrics.stop_all_events),
        "schedule_mode": result.schedule.mode,
        "rule_id": result.schedule.rule_id,
        "independent_env_step_allowed": False,
        "central_joint_stepper_required": True,
        "qpos_delta_norm": float(np.linalg.norm(final_qpos - initial_qpos)),
        "provider_stats": provider_stats,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, sort_keys=True)


def run_demo(args):
    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    env = PackGroceryTask(
        render_cameras=["teaser"],
        image_hw=(64, 64),
        render_freq=1000,
        randomize_init=False,
        render_point_cloud=False,
    )
    env.seed(np_seed=int(args.seed))
    initial_obs = env.reset(reload=True)
    if initial_obs is None:
        initial_obs = env.get_obs()

    plan = _build_demo_plan()
    scheduler = MultiAgentScheduler(SchedulerConfig(enable_concurrency=True))
    schedule = scheduler.schedule(plan, env=env, obs=initial_obs)
    if not schedule.accepted:
        raise RuntimeError("Demo schedule was rejected: {}".format(schedule.reason))

    agent_order = tuple(env.robot_name_map.values())
    providers = {agent: RealHoldProvider(env) for agent in agent_order}
    initial_qpos = env.physics.data.qpos.copy()
    result = SynchronizedExecutor(
        env=env,
        agent_order=agent_order,
        action_providers=providers,
    ).run(schedule, initial_obs=initial_obs, max_joint_steps=int(args.ticks))
    final_qpos = env.physics.data.qpos.copy()
    provider_stats = {
        agent: {"calls": provider.calls, "resets": provider.reset_count}
        for agent, provider in providers.items()
    }
    _write_artifacts(args.output_dir, result, initial_qpos, final_qpos, provider_stats, args.ticks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=3, help="Number of synchronized joint ticks to run.")
    parser.add_argument("--seed", type=int, default=0, help="Simulator seed.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/multi_agent/phase7_real_hold",
        help="Directory for result.json, schedule.json, and summary.json.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    args = parser.parse_args()
    result = run_demo(args)
    print(json.dumps(_json_safe(result.to_dict()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
