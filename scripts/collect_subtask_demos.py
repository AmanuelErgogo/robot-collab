"""collect_subtask_demos.py — collect demonstration data for training subtask ACT policies.

Uses the existing scripted system (LLM + RRT + weld) to roll out complete task
episodes and saves each subtask's (observation, action) pairs as a dataset suitable
for training Action Chunking with Transformers (ACT) policies.

Each demo episode is saved under:
    <output_dir>/<task>/<skill_name>/<episode_NNNN>/
        ├── observations.npz   # joint qpos, EE poses, object positions per step
        ├── actions.npz        # ctrl values per step
        └── metadata.json      # skill call, success, task reset seed

Usage
-----
    python scripts/collect_subtask_demos.py \\
        --task pack \\
        --num_episodes 50 \\
        --output_dir data/subtask_demos \\
        --skills PICK PLACE

    # Collect for sandwich task
    python scripts/collect_subtask_demos.py \\
        --task sandwich \\
        --num_episodes 30 \\
        --output_dir data/subtask_demos \\
        --skills PICK STACK_ON
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Demo recording helpers
# ---------------------------------------------------------------------------

class SubtaskDemoRecorder:
    """
    Wraps the scripted RRT executor to intercept per-step (obs, action) pairs
    and segment them into per-subtask episodes.
    """

    def __init__(self, output_dir: str, task_name: str, skills_to_collect: List[str],
                 env=None, image_cameras: List[str] = None, image_size: tuple = (128, 128)):
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.skills_to_collect = [s.upper() for s in skills_to_collect]
        self._current_skill: Optional[str] = None
        self._obs_buffer: List[Dict] = []
        self._action_buffer: List[Dict] = []
        self._image_buffer: List[Dict] = []   # per-step {cam_name: np.uint8 (H,W,3)}
        self._episode_counts: Dict[str, int] = {}
        self._env = env
        self._image_cameras: List[str] = image_cameras or []
        self._image_size: tuple = image_size

    def start_skill(self, skill_name: str, call_args: Dict):
        """Called before each subtask begins."""
        self._current_skill = skill_name.upper()
        self._obs_buffer = []
        self._action_buffer = []
        self._image_buffer = []
        self._call_args = call_args

    def record_step(self, obs, sim_action):
        """Called after each env.step() during a subtask."""
        if self._current_skill not in self.skills_to_collect:
            return
        obs_dict = _flatten_obs(obs)
        ctrl_array = np.zeros(64, dtype=np.float32)
        for idx, val in zip(sim_action.ctrl_idxs, sim_action.ctrl_vals):
            if 0 <= int(idx) < len(ctrl_array):
                ctrl_array[int(idx)] = float(val)
        # eq_active (weld/gripper) commands
        eq_idxs = np.array(sim_action.eq_active_idxs or [], dtype=np.int32)
        eq_vals = np.array(sim_action.eq_active_vals or [], dtype=np.int32)
        self._obs_buffer.append(obs_dict)
        self._action_buffer.append({
            "ctrl": ctrl_array,
            "eq_active_idxs": eq_idxs,
            "eq_active_vals": eq_vals,
        })
        # Capture camera images for VLA training
        step_images = {}
        if self._env is not None and self._image_cameras:
            for cam in self._image_cameras:
                try:
                    img = self._env.physics.render(
                        height=self._image_size[0],
                        width=self._image_size[1],
                        camera_id=cam,
                    )
                    step_images[cam] = img.astype(np.uint8)
                except Exception:
                    pass
        self._image_buffer.append(step_images)

    def finish_skill(self, skill_name: str, success: bool):
        """Called after the subtask completes. Saves episode if collecting this skill."""
        if self._current_skill not in self.skills_to_collect:
            return
        if not self._obs_buffer:
            return

        ep_idx = self._episode_counts.get(self._current_skill, 0)
        ep_dir = self.output_dir / self.task_name / self._current_skill / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Stack observation fields
        stacked_obs = {}
        for key in self._obs_buffer[0]:
            try:
                stacked_obs[key] = np.stack([o[key] for o in self._obs_buffer], axis=0)
            except Exception:
                pass
        np.savez_compressed(ep_dir / "observations.npz", **stacked_obs)

        # Actions: ctrl array + per-step eq_active (weld/gripper) commands
        ctrl_stack = np.stack([a["ctrl"] for a in self._action_buffer], axis=0)
        # Save eq_active as a variable-length list encoded per-step
        # Steps with no weld command store empty arrays; others store (idxs, vals).
        # We persist them as object arrays so load knows the shape.
        eq_idxs_list = [a["eq_active_idxs"] for a in self._action_buffer]
        eq_vals_list = [a["eq_active_vals"] for a in self._action_buffer]
        # Flatten to (T, max_welds) padding with -1 for absent entries
        max_w = max((len(x) for x in eq_idxs_list), default=0)
        if max_w > 0:
            eq_idxs_arr = np.full((len(eq_idxs_list), max_w), -1, dtype=np.int32)
            eq_vals_arr = np.full((len(eq_vals_list), max_w),  0, dtype=np.int32)
            for t, (idxs, vals) in enumerate(zip(eq_idxs_list, eq_vals_list)):
                if len(idxs) > 0:
                    eq_idxs_arr[t, :len(idxs)] = idxs
                    eq_vals_arr[t, :len(vals)] = vals
        else:
            T = len(eq_idxs_list)
            eq_idxs_arr = np.full((T, 1), -1, dtype=np.int32)
            eq_vals_arr = np.zeros((T, 1), dtype=np.int32)

        np.savez_compressed(ep_dir / "actions.npz",
                            ctrl=ctrl_stack,
                            eq_active_idxs=eq_idxs_arr,
                            eq_active_vals=eq_vals_arr)

        # Save images per camera as (T, H, W, 3) uint8 arrays
        if self._image_buffer and self._image_cameras:
            img_dict = {}
            for cam in self._image_cameras:
                frames = [step.get(cam) for step in self._image_buffer]
                if any(f is not None for f in frames):
                    H, W = self._image_size
                    arr = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
                    for t, f in enumerate(frames):
                        if f is not None:
                            arr[t] = f
                    img_dict[f"image_{cam}"] = arr
            if img_dict:
                np.savez_compressed(ep_dir / "images.npz", **img_dict)

        meta = {
            "skill_name": self._current_skill,
            "call_args": self._call_args,
            "success": bool(success),
            "num_steps": len(self._action_buffer),
            "task": self.task_name,
            "episode": ep_idx,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._episode_counts[self._current_skill] = ep_idx + 1
        status = "✓" if success else "✗"
        print(f"  {status} {self._current_skill} ep={ep_idx:04d} "
              f"steps={len(self._action_buffer)} → {ep_dir}")

        self._obs_buffer = []
        self._action_buffer = []


def _flatten_obs(obs) -> Dict[str, np.ndarray]:
    """Flatten EnvState into a flat dict of numpy arrays."""
    result = {}
    for robot_name in ("ur5e_suction", "ur5e_robotiq", "panda", "humanoid"):
        agent = getattr(obs, robot_name, None)
        if agent is not None:
            result[f"{robot_name}_ee_xpos"] = np.array(agent.ee_xpos, dtype=np.float32)
            result[f"{robot_name}_ee_xquat"] = np.array(agent.ee_xquat, dtype=np.float32)
            result[f"{robot_name}_qpos"] = np.array(agent.qpos, dtype=np.float32)
    for obj_name, obj_state in (obs.objects or {}).items():
        result[f"obj_{obj_name}_xpos"] = np.array(obj_state.xpos, dtype=np.float32)
        result[f"obj_{obj_name}_xquat"] = np.array(obj_state.xquat, dtype=np.float32)
    return result


# ---------------------------------------------------------------------------
# Task env factory
# ---------------------------------------------------------------------------

TASK_MAP = {
    "pack": "rocobench.envs.task_pack:PackGroceryTask",
    "sandwich": "rocobench.envs.task_sandwich:MakeSandwichTask",
    "cabinet": "rocobench.envs.task_cabinet:OpenCabinetTask",
    "sort": "rocobench.envs.task_sort:SortTask",
    "rope": "rocobench.envs.task_rope:RopeTask",
    "sweep": "rocobench.envs.task_sweep:SweepTask",
}

DEFAULT_SKILLS_PER_TASK = {
    "pack": ["PICK", "PLACE", "PICK_AND_PLACE"],
    "sandwich": ["PICK", "STACK_ON"],
    "cabinet": ["PICK", "PLACE", "OPEN_CABINET"],
    "sort": ["PICK", "PLACE"],
    "rope": ["PICK", "PLACE"],
    "sweep": ["SWEEP"],
}


def _load_task_env(task_name: str, render: bool = False):
    entry = TASK_MAP.get(task_name)
    if entry is None:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {sorted(TASK_MAP)}")
    module_path, class_name = entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    # Try increasingly simple constructors
    for kwargs in [{"render_cameras": ["teaser"]}, {"render": render}, {}]:
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    return cls()


# ---------------------------------------------------------------------------
# Scripted rollout runner
# ---------------------------------------------------------------------------

def run_scripted_episode(env, recorder: SubtaskDemoRecorder, verbose: bool = False):
    """
    Run one full scripted episode using the existing PlannedPathPolicy.

    This drives the env through each subtask using the scripted planner,
    recording (obs, action) pairs along the way.

    Returns True if the episode completed successfully.
    """
    from rocobench.policy import PlannedPathPolicy
    from prompting.parser import LLMResponseParser

    obs = env.reset()
    env.sample_initial_scene()
    obs = env.get_obs()

    # Build a simple scripted plan for demonstration
    # In practice you would use the full dialog/prompter pipeline
    robots = env.get_sim_robots()
    agent_names = list(robots.keys())

    # Scripted subtask sequence varies by task — use the env's built-in demo
    if hasattr(env, "get_demo_subtask_sequence"):
        subtask_sequence = env.get_demo_subtask_sequence(obs)
    else:
        subtask_sequence = _build_default_sequence(env, obs, agent_names)

    if not subtask_sequence:
        print("  [WARNING] No subtask sequence available for this task.")
        return False

    for skill_name, call_args, path_plan in subtask_sequence:
        recorder.start_skill(skill_name, call_args)
        success = False
        try:
            policy = PlannedPathPolicy(
                physics=env.physics,
                robots=robots,
                path_plan=path_plan,
                graspable_object_names=env.get_graspable_objects(),
                allowed_collision_pairs=env.get_allowed_collision_pairs(),
            )
            ok, reason = policy.plan(env)
            if not ok:
                if verbose:
                    print(f"  RRT planning failed for {skill_name}: {reason}")
                recorder.finish_skill(skill_name, success=False)
                return False

            while not policy.plan_exhausted:
                sim_action = policy.act(obs, env.physics)
                recorder.record_step(obs, sim_action)
                obs, reward, done, info = env.step(sim_action)
            success = True
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            print(f"  [ERROR] {skill_name}: {exc}")
        finally:
            recorder.finish_skill(skill_name, success=success)

        if not success:
            return False

    return True


def _build_default_sequence(env, obs, agent_names):
    """
    Fallback: build pick-then-place plan for the first graspable object.

    Returns a list of (skill_name, call_args, LLMPathPlan) tuples.
    """
    from rocobench.subtask_plan import LLMPathPlan

    graspable = env.get_graspable_objects()
    if graspable is None:
        return []

    # Flatten dict-of-lists format used by some tasks
    items = []
    if isinstance(graspable, dict):
        for v in graspable.values():
            items.extend(v if isinstance(v, (list, tuple)) else [v])
    else:
        items = list(graspable)

    if not items:
        return []

    obj_name = items[0]
    agent_name = agent_names[0]
    # Resolve hardware body name (obs uses hardware names, not display names)
    name_map_inv = getattr(env, "robot_name_map_inv", {})
    hw_name = name_map_inv.get(agent_name, agent_name)
    agent_state = getattr(obs, hw_name, None)
    if agent_state is None:
        return []

    site_name = env.get_grasp_site(obj_name)
    obj_state = obs.objects.get(obj_name)
    if site_name is None or obj_state is None:
        return []

    sites = obj_state.sites
    site_obj = sites[site_name] if isinstance(sites, dict) else next(
        (s for s in sites if getattr(s, "name", None) == site_name), None)
    if site_obj is None:
        return []

    pick_pos = site_obj.xpos.copy()
    pick_quat = site_obj.xquat.copy()
    pick_target = np.concatenate([pick_pos, pick_quat])

    # Build ee_targets using hardware names for obs lookup but display names as keys
    ee_targets_all = {}
    for a in agent_names:
        hw = name_map_inv.get(a, a)
        a_state = getattr(obs, hw, None)
        if a_state is not None:
            ee_targets_all[a] = np.array(a_state.ee_pose)
    ee_targets_all[agent_name] = pick_target

    # Build hover → pre-grasp waypoints for a more realistic trajectory
    hover_pos = pick_pos.copy()
    hover_pos[2] = max(pick_pos[2] + 0.12, 0.35)   # 12 cm above
    hover_pose = np.concatenate([hover_pos, pick_quat])
    pre_grasp_pos = pick_pos.copy()
    pre_grasp_pos[2] = pick_pos[2] + 0.015
    pre_grasp_pose = np.concatenate([pre_grasp_pos, pick_quat])

    # All agents must have the same waypoint count — pad idle agents with their current pose
    n_waypoints = 2
    ee_waypoints_all = {}
    for a in agent_names:
        hw = name_map_inv.get(a, a)
        a_state = getattr(obs, hw, None)
        if a == agent_name:
            ee_waypoints_all[a] = [hover_pose, pre_grasp_pose]
        elif a_state is not None:
            idle_pose = np.array(a_state.ee_pose)
            ee_waypoints_all[a] = [idle_pose] * n_waypoints
        else:
            ee_waypoints_all[a] = [ee_targets_all.get(a, np.zeros(7))] * n_waypoints

    pick_plan = LLMPathPlan(
        agent_names=agent_names,
        ee_targets=ee_targets_all,
        ee_waypoints=ee_waypoints_all,
        tograsp={agent_name: (obj_name, site_name, 1),
                 **{a: None for a in agent_names if a != agent_name}},
        inhand={a: None for a in agent_names},
        parsed_proposal="scripted:pick",
        action_strs={a: "" for a in agent_names},
        return_home=None,
    )
    return [("PICK", {"object": obj_name}, pick_plan)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASK_MAP),
                   help="Task environment to collect from.")
    p.add_argument("--num_episodes", type=int, default=10,
                   help="Number of scripted rollout episodes.")
    p.add_argument("--output_dir", default="data/subtask_demos",
                   help="Root directory for saving demonstrations.")
    p.add_argument("--skills", nargs="+", default=None,
                   help="Skills to collect. Defaults to the recommended set for the task.")
    p.add_argument("--render", action="store_true", help="Enable MuJoCo renderer.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    skills = args.skills or DEFAULT_SKILLS_PER_TASK.get(args.task, ["PICK", "PLACE"])
    print(f"\n=== Collecting subtask demos ===")
    print(f"  task      : {args.task}")
    print(f"  skills    : {skills}")
    print(f"  episodes  : {args.num_episodes}")
    print(f"  output    : {args.output_dir}")
    print()

    env = _load_task_env(args.task, render=args.render)
    recorder = SubtaskDemoRecorder(args.output_dir, args.task, skills)

    success_count = 0
    for ep in range(args.num_episodes):
        print(f"Episode {ep + 1}/{args.num_episodes}")
        try:
            ok = run_scripted_episode(env, recorder, verbose=args.verbose)
            if ok:
                success_count += 1
        except Exception as exc:
            print(f"  [ERROR] Episode {ep + 1} crashed: {exc}")
            if args.verbose:
                traceback.print_exc()

    print(f"\nDone. {success_count}/{args.num_episodes} successful episodes.")
    print(f"Dataset saved to: {args.output_dir}/{args.task}/")

    # Print per-skill counts
    for skill in skills:
        skill_dir = Path(args.output_dir) / args.task / skill.upper()
        if skill_dir.exists():
            n = len(list(skill_dir.iterdir()))
            print(f"  {skill}: {n} episodes")


if __name__ == "__main__":
    main()
