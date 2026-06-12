"""EnvHub-shaped entry point for future LeRobot integration."""

from typing import Any, Optional

from .env import RoCoGymEnv


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: Optional[Any] = None):
    if n_envs != 1:
        raise NotImplementedError("Phase 0 supports only n_envs=1.")
    if use_async_envs:
        raise NotImplementedError("Phase 0 does not support async/vector environments.")
    kwargs = {}
    if cfg is not None:
        if isinstance(cfg, dict):
            kwargs.update(cfg)
        else:
            for name in ["endpoint", "active_agent", "render_mode", "request_timeout_ms", "max_episode_steps"]:
                if hasattr(cfg, name):
                    kwargs[name] = getattr(cfg, name)
    return RoCoGymEnv(**kwargs)
