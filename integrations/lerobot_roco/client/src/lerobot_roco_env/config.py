"""Client configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoCoGymConfig:
    endpoint: str = "tcp://127.0.0.1:5557"
    active_agent: str = "Alice"
    render_mode: Optional[str] = "rgb_array"
    request_timeout_ms: int = 30000
    max_episode_steps: Optional[int] = None
    action_out_of_bounds: str = "reject"

    def __post_init__(self) -> None:
        if self.active_agent not in {"Alice", "Bob"}:
            raise ValueError("active_agent must be Alice or Bob.")
        if self.render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode='rgb_array' is supported.")
        if self.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive.")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if self.action_out_of_bounds not in {"reject", "clip"}:
            raise ValueError("action_out_of_bounds must be 'reject' or 'clip'.")
