"""Stable error codes and typed exceptions for the RoCo bridge."""

from typing import Any, Dict, Optional


class ErrorCode:
    UNSUPPORTED_PROTOCOL = "UNSUPPORTED_PROTOCOL"
    UNKNOWN_COMMAND = "UNKNOWN_COMMAND"
    INVALID_REQUEST = "INVALID_REQUEST"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
    SERVER_NOT_READY = "SERVER_NOT_READY"
    EPISODE_NOT_ACTIVE = "EPISODE_NOT_ACTIVE"
    EPISODE_ALREADY_ACTIVE = "EPISODE_ALREADY_ACTIVE"
    EPISODE_TERMINATED = "EPISODE_TERMINATED"
    EPISODE_ID_MISMATCH = "EPISODE_ID_MISMATCH"
    STEP_INDEX_MISMATCH = "STEP_INDEX_MISMATCH"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    UNKNOWN_AGENT = "UNKNOWN_AGENT"
    INVALID_ACTION_SHAPE = "INVALID_ACTION_SHAPE"
    INVALID_ACTION_DTYPE = "INVALID_ACTION_DTYPE"
    NONFINITE_ACTION = "NONFINITE_ACTION"
    ACTION_OUT_OF_BOUNDS = "ACTION_OUT_OF_BOUNDS"
    INVALID_CONTROL_INDEX = "INVALID_CONTROL_INDEX"
    OBSERVATION_SHAPE_MISMATCH = "OBSERVATION_SHAPE_MISMATCH"
    RENDER_FAILED = "RENDER_FAILED"
    RESET_FAILED = "RESET_FAILED"
    STEP_FAILED = "STEP_FAILED"
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    SERVER_DISCONNECTED = "SERVER_DISCONNECTED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DEBUG_COMMAND_DISABLED = "DEBUG_COMMAND_DISABLED"
    UNAUTHORIZED_SHUTDOWN = "UNAUTHORIZED_SHUTDOWN"


class RoCoBridgeError(Exception):
    """Base exception carrying a stable bridge error code."""

    default_code = ErrorCode.INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_error_payload(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class RoCoConnectionError(RoCoBridgeError):
    default_code = ErrorCode.SERVER_DISCONNECTED


class RoCoProtocolError(RoCoBridgeError):
    default_code = ErrorCode.INVALID_REQUEST


class RoCoEpisodeError(RoCoBridgeError):
    default_code = ErrorCode.EPISODE_NOT_ACTIVE


class RoCoActionError(RoCoBridgeError):
    default_code = ErrorCode.INVALID_ACTION_SHAPE


class RoCoObservationError(RoCoBridgeError):
    default_code = ErrorCode.OBSERVATION_SHAPE_MISMATCH


def exception_from_error_payload(error: Dict[str, Any]) -> RoCoBridgeError:
    code = str(error.get("code", ErrorCode.INTERNAL_SERVER_ERROR))
    message = str(error.get("message", code))
    details = error.get("details", {})
    retryable = bool(error.get("retryable", False))

    cls = RoCoBridgeError
    if code in {ErrorCode.CONNECTION_TIMEOUT, ErrorCode.SERVER_DISCONNECTED}:
        cls = RoCoConnectionError
    elif code in {
        ErrorCode.UNSUPPORTED_PROTOCOL,
        ErrorCode.UNKNOWN_COMMAND,
        ErrorCode.INVALID_REQUEST,
        ErrorCode.PAYLOAD_TOO_LARGE,
        ErrorCode.SERIALIZATION_ERROR,
    }:
        cls = RoCoProtocolError
    elif code in {
        ErrorCode.EPISODE_NOT_ACTIVE,
        ErrorCode.EPISODE_ALREADY_ACTIVE,
        ErrorCode.EPISODE_TERMINATED,
        ErrorCode.EPISODE_ID_MISMATCH,
        ErrorCode.STEP_INDEX_MISMATCH,
    }:
        cls = RoCoEpisodeError
    elif code in {
        ErrorCode.INVALID_ACTION_SHAPE,
        ErrorCode.INVALID_ACTION_DTYPE,
        ErrorCode.NONFINITE_ACTION,
        ErrorCode.ACTION_OUT_OF_BOUNDS,
        ErrorCode.INVALID_CONTROL_INDEX,
    }:
        cls = RoCoActionError
    elif code in {ErrorCode.OBSERVATION_SHAPE_MISMATCH, ErrorCode.RENDER_FAILED}:
        cls = RoCoObservationError
    return cls(message=message, code=code, details=details, retryable=retryable)
