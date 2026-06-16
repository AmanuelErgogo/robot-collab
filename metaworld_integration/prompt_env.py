"""Prompt-friendly MetaWorld environment wrapper."""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from .config import task_one_hot


def _fmt_vec(vec):
    if vec is None:
        return "n/a"
    return "({})".format(", ".join("{:.3f}".format(float(value)) for value in vec))


class MetaWorldPromptEnv:
    """Wrap the bridge client with prompt, feedback, and export helpers."""

    def __init__(self, bridge_client, max_steps=200):
        self.bridge = bridge_client
        self.max_steps = max_steps

    def reset(self, seed=None):
        return self.bridge.reset(seed=seed)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        return self.bridge.step(action.tolist())

    def expert_action(self):
        return np.asarray(self.bridge.expert_action(), dtype=np.float32)

    def save_frame(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.bridge.save_frame(path)
        return path

    def build_system_prompt(self):
        return (
            "You control a single Sawyer arm inside a MetaWorld manipulation task.\n"
            "At each turn, output exactly one 4D control command.\n"
            "The action format is:\n"
            "EXECUTE\n"
            "ACTION [dx, dy, dz, gripper]\n"
            "Use floats in [-1, 1]. The first three values move the end effector. "
            "The last value controls the gripper: negative tends to close, positive tends to open.\n"
            "Use concise reasoning mentally, then only emit the required action block."
        )

    def describe_state(self, snapshot):
        lines = [
            "[Task]",
            "- Name: {}".format(snapshot["task_name"]),
            "- Description: {}".format(snapshot["task_description"]),
            "[Current State]",
            "- Step: {} / {}".format(snapshot["step_index"], self.max_steps),
            "- End effector: {}".format(_fmt_vec(snapshot.get("tcp_center"))),
            "- Gripper open value: {}".format(snapshot.get("gripper_open")),
            "- Object position: {}".format(_fmt_vec(snapshot.get("object_position"))),
            "- Target position: {}".format(_fmt_vec(snapshot.get("target_position"))),
            "- Hand to target distance: {}".format(snapshot.get("hand_to_target_distance")),
            "- Hand to object distance: {}".format(snapshot.get("hand_to_object_distance")),
            "- Object to target distance: {}".format(snapshot.get("object_to_target_distance")),
            "- Previous reward: {}".format(snapshot.get("reward")),
            "- Success: {}".format(snapshot.get("success")),
        ]
        return "\n".join(lines)

    def build_user_prompt(self, snapshot, feedback_text=""):
        prompt = self.describe_state(snapshot) + "\n"
        if len(feedback_text.strip()) > 0:
            prompt += "\n[Feedback]\n{}\n".format(feedback_text.strip())
        prompt += (
            "\nChoose the next control to make progress on the task. "
            "Prefer small, stable motions over large jumps.\n"
            "Respond with only:\nEXECUTE\nACTION [dx, dy, dz, gripper]\n"
        )
        return prompt

    def build_feedback(self, previous_snapshot, action, current_snapshot):
        if current_snapshot.get("success", False):
            return "The last action succeeded. Keep stable or finish."

        prev_reward = previous_snapshot.get("reward")
        curr_reward = current_snapshot.get("reward")
        feedback_bits = []
        if prev_reward is not None and curr_reward is not None:
            if curr_reward > prev_reward:
                feedback_bits.append(
                    "Reward improved from {:.4f} to {:.4f}.".format(prev_reward, curr_reward)
                )
            elif curr_reward < prev_reward:
                feedback_bits.append(
                    "Reward dropped from {:.4f} to {:.4f}.".format(prev_reward, curr_reward)
                )
            else:
                feedback_bits.append("Reward did not change.")

        obj_dist = current_snapshot.get("object_to_target_distance")
        hand_dist = current_snapshot.get("hand_to_object_distance")
        if obj_dist is not None:
            feedback_bits.append("Object-to-target distance is now {}.".format(obj_dist))
        if hand_dist is not None:
            feedback_bits.append("Hand-to-object distance is now {}.".format(hand_dist))
        feedback_bits.append(
            "Previous action was [{:.3f}, {:.3f}, {:.3f}, {:.3f}].".format(*np.asarray(action).tolist())
        )
        return " ".join(feedback_bits)

    def to_lerobot_record(
        self,
        episode_index,
        step_index,
        snapshot,
        action,
        outcome_snapshot,
        image_path=None,
        prompt_text="",
        response_text="",
    ):
        return {
            "episode_index": int(episode_index),
            "step_index": int(step_index),
            "task": snapshot["task_name"],
            "task_description": snapshot["task_description"],
            "task_id": snapshot["task_id"],
            "task_one_hot": task_one_hot(snapshot["task_name"]),
            "observation.state": snapshot["agent_pos"],
            "observation.image_path": image_path,
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "reward": outcome_snapshot.get("reward"),
            "terminated": outcome_snapshot.get("terminated"),
            "truncated": outcome_snapshot.get("truncated"),
            "success": outcome_snapshot.get("success"),
            "prompt": prompt_text,
            "response": response_text,
        }
