"""CLI entrypoint for the RoCo bridge server."""

import argparse
import logging
from typing import Iterable, Optional

from .config import RoCoBridgeServerConfig, default_camera_aliases
from .server import RoCoBridgeServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the RoCo Phase 0 bridge server.")
    parser.add_argument("--task", default="pack")
    parser.add_argument("--active-agent", default="Alice", choices=["Alice", "Bob"])
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--cameras", default=None, help="Comma-separated alias=actual camera mappings.")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--request-log-level", default="INFO")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--live-view", action="store_true", help="Open an OpenCV window and show rendered frames on reset/step/render.")
    parser.add_argument("--live-view-camera", default="front", help="Camera alias to show in --live-view, usually front or active_agent.")
    parser.add_argument("--enable-debug-commands", action="store_true")
    parser.add_argument("--allow-remote-shutdown", action="store_true")
    parser.add_argument("--session-token", default=None)
    parser.add_argument("--unsafe-bind-non-loopback", action="store_true")
    return parser


def parse_camera_aliases(value: Optional[str], active_agent: str):
    if value is None:
        return default_camera_aliases(active_agent)
    aliases = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Camera mapping must use alias=actual syntax.")
        alias, camera = item.split("=", 1)
        aliases[alias.strip()] = camera.strip()
    return aliases


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = RoCoBridgeServerConfig(
        endpoint=args.endpoint,
        task=args.task,
        active_agent=args.active_agent,
        seed=args.seed,
        image_height=args.image_height,
        image_width=args.image_width,
        camera_aliases=parse_camera_aliases(args.cameras, args.active_agent),
        max_episode_steps=args.max_episode_steps,
        request_log_level=args.request_log_level,
        headless=args.headless,
        live_view=args.live_view,
        live_view_camera=args.live_view_camera,
        enable_debug_commands=args.enable_debug_commands,
        allow_remote_shutdown=args.allow_remote_shutdown,
        session_token=args.session_token,
        unsafe_bind_non_loopback=args.unsafe_bind_non_loopback,
    )
    logging.basicConfig(level=getattr(logging, config.request_log_level.upper(), logging.INFO))
    server = RoCoBridgeServer(config)
    server.serve_forever()


if __name__ == "__main__":
    main()
