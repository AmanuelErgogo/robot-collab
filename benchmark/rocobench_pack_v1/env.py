"""EnvHub-shaped entry point for RoCoBench-Pack-Skills-v1."""

import os
import sys
from typing import Any, Optional

from .version import REPO_ROOT


def _ensure_client_src() -> None:
    client_src = os.path.join(REPO_ROOT, "integrations", "lerobot_roco", "client", "src")
    if client_src not in sys.path:
        sys.path.insert(0, client_src)


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: Optional[Any] = None):
    """Create the bridge-backed Gym environment.

    The Python 3.8 RoCo simulator server must already be running. This function
    only creates the Python 3.12+/Gym/LeRobot client-side environment.
    """

    _ensure_client_src()
    from lerobot_roco_env.envhub import make_env as make_roco_env

    return make_roco_env(n_envs=n_envs, use_async_envs=use_async_envs, cfg=cfg)

