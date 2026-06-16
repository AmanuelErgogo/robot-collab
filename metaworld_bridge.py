"""Persistent helper process for running MetaWorld under Python 3.10+.

This script is intentionally isolated from the rest of the repo because
MetaWorld 3.x requires a newer Python environment than the original RoCo stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

from metaworld_integration.config import (
    TASK_NAME_TO_ID,
    get_policy_class_name,
    get_task_description,
)


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _patch_metaworld_render_metadata():
    """Backfill Gymnasium's newer rgbd render mode into MetaWorld's metadata.

    MetaWorld 3.0.0 still defines render modes as
    ["human", "rgb_array", "depth_array"], while recent Gymnasium MuJoCo
    environments assert the presence of "rgbd_tuple" as well. Patching the
    class metadata is enough to make the environments instantiate cleanly.
    """
    import metaworld.sawyer_xyz_env as sawyer_xyz_env

    render_modes = list(sawyer_xyz_env.SawyerMocapBase.metadata.get("render_modes", []))
    if "rgbd_tuple" not in render_modes:
        render_modes.append("rgbd_tuple")
        sawyer_xyz_env.SawyerMocapBase.metadata["render_modes"] = render_modes
        sawyer_xyz_env.SawyerXYZEnv.metadata["render_modes"] = list(render_modes)


class MetaWorldSession:
    def __init__(self, task_name, camera_name, width, height, repo_dir=None):
        if repo_dir:
            sys.path.insert(0, repo_dir)

        import imageio.v2 as imageio
        import metaworld
        import metaworld.policies as policies

        _patch_metaworld_render_metadata()

        self.imageio = imageio
        self.metaworld = metaworld
        self.policies = policies
        self.task_name = task_name
        self.task_description = get_task_description(task_name)
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.step_index = 0
        self.last_obs = None
        self.last_info = {}

        benchmark = metaworld.MT1(task_name, seed=42)
        env_cls = benchmark.train_classes[task_name]
        self.env = env_cls(
            render_mode="rgb_array",
            camera_name=camera_name,
            width=width,
            height=height,
        )
        self.env.set_task(benchmark.train_tasks[0])
        self.env._freeze_rand_vec = False

        policy_class_name = get_policy_class_name(task_name)
        self.expert_policy = getattr(policies, policy_class_name)()

    def _extract_summary(self, raw_obs, reward=None, terminated=False, truncated=False, info=None):
        info = info or {}
        tcp_center = None
        if hasattr(self.env, "tcp_center"):
            tcp_center = np.asarray(self.env.tcp_center, dtype=np.float32)

        object_position = None
        if hasattr(self.env, "_get_pos_objects"):
            try:
                object_position = np.asarray(self.env._get_pos_objects(), dtype=np.float32)
            except Exception:
                object_position = None

        target_position = None
        if getattr(self.env, "_target_pos", None) is not None:
            target_position = np.asarray(self.env._target_pos, dtype=np.float32)

        gripper_open = None
        agent_pos = None
        if raw_obs is not None:
            raw_obs = np.asarray(raw_obs, dtype=np.float32)
            if raw_obs.shape[0] >= 4:
                agent_pos = raw_obs[:4]
                gripper_open = float(raw_obs[3])

        hand_to_target = None
        if tcp_center is not None and target_position is not None:
            hand_to_target = float(np.linalg.norm(tcp_center - target_position))

        hand_to_object = None
        if tcp_center is not None and object_position is not None:
            hand_to_object = float(np.linalg.norm(tcp_center - object_position))

        object_to_target = None
        if object_position is not None and target_position is not None:
            object_to_target = float(np.linalg.norm(object_position - target_position))

        success = bool(info.get("success", 0))
        return {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "task_id": TASK_NAME_TO_ID[self.task_name],
            "step_index": int(self.step_index),
            "raw_obs": _to_jsonable(raw_obs),
            "agent_pos": _to_jsonable(agent_pos),
            "tcp_center": _to_jsonable(tcp_center),
            "gripper_open": gripper_open,
            "object_position": _to_jsonable(object_position),
            "target_position": _to_jsonable(target_position),
            "hand_to_target_distance": hand_to_target,
            "hand_to_object_distance": hand_to_object,
            "object_to_target_distance": object_to_target,
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
            "info": _to_jsonable(info),
        }

    def reset(self, seed=None):
        self.step_index = 0
        raw_obs, info = self.env.reset(seed=seed)
        self.last_obs = raw_obs
        self.last_info = info
        return self._extract_summary(raw_obs, reward=None, terminated=False, truncated=False, info=info)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_index += 1
        self.last_obs = raw_obs
        self.last_info = info
        summary = self._extract_summary(
            raw_obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        return summary

    def expert_action(self):
        if self.last_obs is None:
            raise RuntimeError("Environment must be reset before requesting an expert action.")
        action = np.asarray(self.expert_policy.get_action(self.last_obs), dtype=np.float32)
        return {"action": _to_jsonable(action)}

    def save_frame(self, path):
        frame = self.env.render()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.imageio.imwrite(path, frame)
        return {"path": path}

    def close(self):
        self.env.close()
        return {"closed": True}


def _send(ok, data=None, error=None):
    payload = {"ok": bool(ok)}
    if ok:
        payload["data"] = data or {}
    else:
        payload["error"] = error or "Unknown error."
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--camera-name", default="corner2")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--repo-dir", default=None)
    args = parser.parse_args()

    try:
        session = MetaWorldSession(
            task_name=args.task,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            repo_dir=args.repo_dir,
        )
        _send(
            True,
            data={
                "task_name": args.task,
                "task_description": session.task_description,
                "camera_name": args.camera_name,
                "width": args.width,
                "height": args.height,
            },
        )
    except Exception:
        _send(False, error=traceback.format_exc())
        return

    command_map = {
        "reset": session.reset,
        "step": session.step,
        "expert_action": session.expert_action,
        "save_frame": session.save_frame,
        "close": session.close,
    }

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            command = payload["command"]
            kwargs = payload.get("kwargs", {})
            if command not in command_map:
                raise ValueError("Unknown command '{}'.".format(command))
            result = command_map[command](**kwargs)
            _send(True, data=_to_jsonable(result))
            if command == "close":
                break
        except Exception:
            _send(False, error=traceback.format_exc())


if __name__ == "__main__":
    main()
