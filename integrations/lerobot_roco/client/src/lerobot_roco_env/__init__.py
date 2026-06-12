"""Gymnasium client for the RoCo Phase 0 bridge."""

__all__ = ["RoCoGymEnv", "make_env"]


def __getattr__(name):
    if name == "RoCoGymEnv":
        from .env import RoCoGymEnv

        return RoCoGymEnv
    if name == "make_env":
        from .envhub import make_env

        return make_env
    raise AttributeError(name)
