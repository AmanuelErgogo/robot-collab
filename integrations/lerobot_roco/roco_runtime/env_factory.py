"""Factory for supported RoCoBench tasks."""

from typing import Any

from .config import RoCoBridgeServerConfig


def create_roco_env(config: RoCoBridgeServerConfig) -> Any:
    if config.task != "pack":
        raise ValueError("Phase 0 bridge currently supports only task=pack.")

    from rocobench.envs.task_pack import PackGroceryTask

    env = PackGroceryTask(
        image_hw=(config.image_height, config.image_width),
        np_seed=config.seed,
    )
    return env
