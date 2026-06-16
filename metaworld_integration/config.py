"""Static task metadata for MetaWorld integration.

The task descriptions and difficulty splits mirror the public MetaWorld/LeRobot
task catalog so the prompts stay human-readable and line up with the
single-arm benchmark naming that downstream tooling expects.
"""

from __future__ import annotations

from typing import Dict, List


TASK_DESCRIPTIONS = {
    "assembly-v3": "Pick up a nut and place it onto a peg",
    "basketball-v3": "Dunk the basketball into the basket",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin",
    "box-close-v3": "Grasp the cover and close the box with it",
    "button-press-topdown-v3": "Press a button from the top",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top",
    "button-press-v3": "Press a button",
    "button-press-wall-v3": "Bypass a wall and press a button",
    "coffee-button-v3": "Push a button on the coffee machine",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "coffee-push-v3": "Push a mug under the coffee machine spout",
    "dial-turn-v3": "Rotate a dial clockwise",
    "disassemble-v3": "Lift a nut off a peg",
    "door-close-v3": "Close an open door",
    "door-lock-v3": "Lock the door by pushing the lock button",
    "door-open-v3": "Open a door by pulling the handle",
    "door-unlock-v3": "Unlock the door by pulling the lock button",
    "drawer-close-v3": "Push a drawer closed",
    "drawer-open-v3": "Pull a drawer open",
    "faucet-close-v3": "Rotate a faucet handle clockwise to close it",
    "faucet-open-v3": "Rotate a faucet handle counter-clockwise to open it",
    "hammer-v3": "Hammer a screw into the wall",
    "hand-insert-v3": "Insert the gripper into a hole",
    "handle-press-side-v3": "Press a handle down from the side",
    "handle-press-v3": "Press a handle down",
    "handle-pull-side-v3": "Pull a handle up from the side",
    "handle-pull-v3": "Pull a handle up",
    "lever-pull-v3": "Pull a lever down",
    "peg-insert-side-v3": "Pick up a peg and insert it into the side hole",
    "peg-unplug-side-v3": "Remove a peg from the side hole",
    "pick-out-of-hole-v3": "Pick an object out of a hole",
    "pick-place-v3": "Pick up a puck and move it to the goal",
    "pick-place-wall-v3": "Pick up a puck, move it around a wall, and place it at the goal",
    "plate-slide-back-side-v3": "Pull a plate toward the robot from the side",
    "plate-slide-back-v3": "Pull a plate toward the robot",
    "plate-slide-side-v3": "Slide a plate into the side cabinet opening",
    "plate-slide-v3": "Slide a plate into the cabinet opening",
    "push-back-v3": "Push an object to the back target",
    "push-v3": "Push an object to a target location",
    "push-wall-v3": "Push an object around a wall to the target",
    "reach-v3": "Move the gripper to the target position",
    "reach-wall-v3": "Move the gripper around a wall to the target position",
    "shelf-place-v3": "Place a puck onto a shelf",
    "soccer-v3": "Push a soccer ball into the goal",
    "stick-pull-v3": "Use a stick to pull an object closer",
    "stick-push-v3": "Use a stick to push an object away",
    "sweep-into-v3": "Sweep a puck into the goal",
    "sweep-v3": "Sweep a puck across the table",
    "window-open-v3": "Slide a window open",
    "window-close-v3": "Slide a window closed",
}

TASK_ORDER = tuple(TASK_DESCRIPTIONS.keys())
TASK_NAME_TO_ID = {task_name: idx for idx, task_name in enumerate(TASK_ORDER)}

DIFFICULTY_TO_TASKS = {
    "easy": [
        "button-press-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "dial-turn-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "handle-press-v3",
        "handle-press-side-v3",
        "handle-pull-v3",
        "handle-pull-side-v3",
        "lever-pull-v3",
        "plate-slide-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "plate-slide-side-v3",
        "reach-v3",
        "reach-wall-v3",
        "window-close-v3",
        "window-open-v3",
        "peg-unplug-side-v3",
    ],
    "medium": [
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "hammer-v3",
        "peg-insert-side-v3",
        "push-wall-v3",
        "soccer-v3",
        "sweep-v3",
        "sweep-into-v3",
    ],
    "hard": [
        "assembly-v3",
        "hand-insert-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "push-v3",
        "push-back-v3",
    ],
    "very_hard": [
        "shelf-place-v3",
        "disassemble-v3",
        "stick-pull-v3",
        "stick-push-v3",
        "pick-place-wall-v3",
    ],
}


def resolve_task_spec(task_spec):
    """Resolve comma-separated task names and difficulty groups."""
    tokens = [token.strip() for token in str(task_spec).split(",") if token.strip()]
    if len(tokens) == 0:
        raise ValueError("task_spec must include at least one MetaWorld task or difficulty group.")

    resolved = []
    seen = set()
    for token in tokens:
        expanded = DIFFICULTY_TO_TASKS.get(token, [token])
        for task_name in expanded:
            if task_name not in TASK_NAME_TO_ID:
                raise ValueError(
                    "Unknown MetaWorld task '{}' . Known tasks include: {}".format(
                        task_name,
                        ", ".join(list(TASK_ORDER)[:8]) + ", ...",
                    )
                )
            if task_name not in seen:
                seen.add(task_name)
                resolved.append(task_name)
    return resolved


def task_one_hot(task_name):
    """Return a LeRobot-friendly one-hot encoding for the task."""
    if task_name not in TASK_NAME_TO_ID:
        raise ValueError("Unknown MetaWorld task '{}'.".format(task_name))
    one_hot = [0] * len(TASK_ORDER)
    one_hot[TASK_NAME_TO_ID[task_name]] = 1
    return one_hot


def get_task_description(task_name):
    if task_name not in TASK_DESCRIPTIONS:
        raise ValueError("Unknown MetaWorld task '{}'.".format(task_name))
    return TASK_DESCRIPTIONS[task_name]


def get_policy_class_name(task_name):
    """Infer the upstream expert-policy class name from a task string."""
    if task_name not in TASK_NAME_TO_ID:
        raise ValueError("Unknown MetaWorld task '{}'.".format(task_name))
    stem = task_name
    if stem.endswith("-v3"):
        stem = stem[:-3]
    camel = "".join(part.capitalize() for part in stem.split("-"))
    return "Sawyer{}V3Policy".format(camel)


def list_supported_tasks():
    return list(TASK_ORDER)
