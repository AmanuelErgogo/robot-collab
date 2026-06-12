"""Shared protocol utilities for the RoCo bridge.

The common package must stay Python 3.8-compatible and must not import MuJoCo,
Gymnasium, LeRobot, or rocobench.
"""

from .errors import (
    ErrorCode,
    RoCoActionError,
    RoCoBridgeError,
    RoCoConnectionError,
    RoCoEpisodeError,
    RoCoObservationError,
    RoCoProtocolError,
)
from .protocol import PROTOCOL_VERSION, make_request
from .types import ArraySpec, CameraSpec, RoCoEnvSpec

__all__ = [
    "ArraySpec",
    "CameraSpec",
    "ErrorCode",
    "PROTOCOL_VERSION",
    "RoCoActionError",
    "RoCoBridgeError",
    "RoCoConnectionError",
    "RoCoEnvSpec",
    "RoCoEpisodeError",
    "RoCoObservationError",
    "RoCoProtocolError",
    "make_request",
]
