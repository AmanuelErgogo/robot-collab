"""Top-level RoCo exports.

Heavy simulator modules are imported lazily so lightweight packages such as
``rocobench.skills`` can be tested without MuJoCo/dm_control installed.
"""

import importlib


def __getattr__(name):
    env_exports = {
        "CabinetTask",
        "EnvState",
        "MakeSandwichTask",
        "MoveRopeTask",
        "MujocoSimEnv",
        "ObjectState",
        "PackGroceryTask",
        "RobotState",
        "SimAction",
        "SimRobot",
        "SimSaveData",
        "SortOneBlockTask",
        "SweepTask",
    }
    if name in env_exports:
        envs = importlib.import_module("rocobench.envs")
        return getattr(envs, name)
    if name == "PlannedPathPolicy":
        from rocobench.policy import PlannedPathPolicy

        return PlannedPathPolicy
    if name == "LLMPathPlan":
        from rocobench.subtask_plan import LLMPathPlan

        return LLMPathPlan
    if name == "MultiArmRRT":
        from rocobench.rrt_multi_arm import MultiArmRRT

        return MultiArmRRT
    raise AttributeError(name)


__all__ = [
    "CabinetTask",
    "EnvState",
    "LLMPathPlan",
    "MakeSandwichTask",
    "MoveRopeTask",
    "MujocoSimEnv",
    "MultiArmRRT",
    "ObjectState",
    "PackGroceryTask",
    "PlannedPathPolicy",
    "RobotState",
    "SimAction",
    "SimRobot",
    "SimSaveData",
    "SortOneBlockTask",
    "SweepTask",
]
