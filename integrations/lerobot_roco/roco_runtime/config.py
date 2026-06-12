"""RoCo bridge server configuration."""

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional
from urllib.parse import urlparse


def default_camera_aliases(active_agent: str) -> Dict[str, str]:
    aliases = {"front": "teaser"}
    if active_agent == "Alice":
        aliases["active_agent"] = "face_ur5e"
    elif active_agent == "Bob":
        aliases["active_agent"] = "face_panda"
    return aliases


def endpoint_host(endpoint: str) -> Optional[str]:
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp":
        return None
    if parsed.hostname is not None:
        return parsed.hostname
    rest = endpoint.split("tcp://", 1)[-1]
    return rest.rsplit(":", 1)[0]


def is_loopback_endpoint(endpoint: str) -> bool:
    host = endpoint_host(endpoint)
    return host in {"127.0.0.1", "localhost", "::1"}


@dataclass
class RoCoBridgeServerConfig:
    endpoint: str = "tcp://127.0.0.1:5557"
    task: str = "pack"
    active_agent: str = "Alice"
    seed: int = 0
    image_height: int = 256
    image_width: int = 256
    camera_aliases: Optional[Mapping[str, str]] = None
    max_episode_steps: int = 300
    request_timeout_ms: int = 30000
    max_payload_bytes: int = 64 * 1024 * 1024
    enable_debug_commands: bool = False
    allow_remote_shutdown: bool = False
    session_token: Optional[str] = None
    unsafe_bind_non_loopback: bool = False
    headless: bool = False
    live_view: bool = False
    live_view_camera: str = "front"
    request_log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.task != "pack":
            raise ValueError("Phase 0 bridge currently supports only task=pack.")
        if self.active_agent not in {"Alice", "Bob"}:
            raise ValueError("active_agent must be Alice or Bob.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive.")
        if not is_loopback_endpoint(self.endpoint) and not self.unsafe_bind_non_loopback:
            raise ValueError("Refusing to bind non-loopback endpoint without unsafe override.")
        if self.camera_aliases is None:
            self.camera_aliases = default_camera_aliases(self.active_agent)
        if self.headless:
            os.environ.setdefault("MUJOCO_GL", "egl")
