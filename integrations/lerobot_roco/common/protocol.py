"""Versioned request/response envelope validation."""

import time
import uuid
from typing import Any, Dict, Iterable, Optional

from .errors import ErrorCode, RoCoBridgeError, RoCoProtocolError
from .serialization import MAX_PAYLOAD_BYTES, estimate_payload_size

PROTOCOL_VERSION = "0.1"

COMMANDS = {
    "PING",
    "HELLO",
    "GET_SPEC",
    "RESET",
    "STEP",
    "STEP_NATIVE_ACTION",
    "RENDER",
    "GET_STATE_DIGEST",
    "CLOSE_EPISODE",
    "SHUTDOWN",
}


def new_request_id() -> str:
    return str(uuid.uuid4())


def make_request(command: str, payload: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id or new_request_id(),
        "command": command,
        "payload": payload or {},
    }


def make_success_response(request: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": str(request.get("request_id", "")),
        "ok": True,
        "payload": payload or {},
    }


def make_error_response(
    request: Optional[Dict[str, Any]],
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
) -> Dict[str, Any]:
    request = request or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": str(request.get("request_id", "")),
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "retryable": retryable,
        },
    }


def make_exception_response(request: Optional[Dict[str, Any]], exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, RoCoBridgeError):
        return make_error_response(
            request=request,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            retryable=exc.retryable,
        )
    return make_error_response(
        request=request,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="Internal server error.",
        details={},
        retryable=False,
    )


def validate_request_envelope(
    request: Dict[str, Any],
    supported_versions: Optional[Iterable[str]] = None,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    allow_debug_commands: bool = False,
) -> None:
    if not isinstance(request, dict):
        raise RoCoProtocolError("Request must be a map.", code=ErrorCode.INVALID_REQUEST)
    supported = set(supported_versions or [PROTOCOL_VERSION])
    version = request.get("protocol_version")
    if version not in supported:
        raise RoCoProtocolError(
            "Unsupported protocol version.",
            code=ErrorCode.UNSUPPORTED_PROTOCOL,
            details={"received": version, "supported": sorted(supported)},
        )
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RoCoProtocolError("Request ID must be a non-empty string.", code=ErrorCode.INVALID_REQUEST)
    command = request.get("command")
    if command not in COMMANDS:
        raise RoCoProtocolError(
            "Unknown command.",
            code=ErrorCode.UNKNOWN_COMMAND,
            details={"command": command},
        )
    if command == "STEP_NATIVE_ACTION" and not allow_debug_commands:
        raise RoCoProtocolError("Debug command is disabled.", code=ErrorCode.DEBUG_COMMAND_DISABLED)
    payload = request.get("payload")
    if payload is None:
        request["payload"] = {}
        payload = request["payload"]
    if not isinstance(payload, dict):
        raise RoCoProtocolError("Payload must be a map.", code=ErrorCode.INVALID_REQUEST)
    estimated = estimate_payload_size(payload)
    if estimated > max_payload_bytes:
        raise RoCoProtocolError(
            "Payload exceeds maximum size.",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            details={"estimated_bytes": estimated, "max_payload_bytes": int(max_payload_bytes)},
        )


def validate_response_envelope(response: Dict[str, Any], request_id: str) -> None:
    if not isinstance(response, dict):
        raise RoCoProtocolError("Response must be a map.", code=ErrorCode.INVALID_REQUEST)
    version = response.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise RoCoProtocolError(
            "Unsupported response protocol version.",
            code=ErrorCode.UNSUPPORTED_PROTOCOL,
            details={"received": version, "supported": [PROTOCOL_VERSION]},
        )
    if response.get("request_id") != request_id:
        raise RoCoProtocolError(
            "Response request ID does not match request.",
            code=ErrorCode.INVALID_REQUEST,
            details={"expected": request_id, "received": response.get("request_id")},
        )
    if response.get("ok") not in (True, False):
        raise RoCoProtocolError("Response ok field must be boolean.", code=ErrorCode.INVALID_REQUEST)
    if response["ok"]:
        if not isinstance(response.get("payload", {}), dict):
            raise RoCoProtocolError("Successful response payload must be a map.", code=ErrorCode.INVALID_REQUEST)
    else:
        error = response.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("code"), str):
            raise RoCoProtocolError("Error response must include an error code.", code=ErrorCode.INVALID_REQUEST)


def ping_payload() -> Dict[str, Any]:
    return {"server_time": time.time(), "status": "ready"}
