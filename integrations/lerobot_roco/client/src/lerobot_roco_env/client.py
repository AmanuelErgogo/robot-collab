"""Remote protocol client for the RoCo bridge."""

from typing import Any, Dict, Optional

from integrations.lerobot_roco.common.errors import (
    ErrorCode,
    RoCoConnectionError,
    exception_from_error_payload,
)
from integrations.lerobot_roco.common.protocol import (
    PROTOCOL_VERSION,
    make_request,
    validate_response_envelope,
)
from integrations.lerobot_roco.common.serialization import MAX_PAYLOAD_BYTES, pack_message, unpack_message
from integrations.lerobot_roco.common.types import RoCoEnvSpec


class RemoteRoCoClient:
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        request_timeout_ms: int = 30000,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    ) -> None:
        self.endpoint = endpoint
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_payload_bytes = int(max_payload_bytes)
        self._context = None
        self._socket = None
        self._closed = False

    def _ensure_socket(self) -> Any:
        if self._closed:
            raise RoCoConnectionError("Client is closed.", code=ErrorCode.SERVER_DISCONNECTED)
        if self._socket is not None:
            return self._socket
        try:
            import zmq
        except ImportError as exc:
            raise RoCoConnectionError("pyzmq is required for bridge transport.", code=ErrorCode.SERVER_DISCONNECTED) from exc
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        return self._socket

    def request(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            import zmq
        except ImportError:
            zmq = None
        request = make_request(command, payload or {})
        socket = self._ensure_socket()
        try:
            socket.send(pack_message(request, max_payload_bytes=self.max_payload_bytes))
            data = socket.recv()
        except Exception as exc:
            if zmq is not None and isinstance(exc, zmq.Again):
                raise RoCoConnectionError(
                    "Timed out waiting for RoCo bridge response.",
                    code=ErrorCode.CONNECTION_TIMEOUT,
                    retryable=True,
                ) from exc
            raise RoCoConnectionError("RoCo bridge disconnected.", code=ErrorCode.SERVER_DISCONNECTED, retryable=True) from exc
        response = unpack_message(data, max_payload_bytes=self.max_payload_bytes)
        validate_response_envelope(response, request_id=request["request_id"])
        if not response["ok"]:
            raise exception_from_error_payload(response["error"])
        return response.get("payload", {})

    def ping(self) -> Dict[str, Any]:
        return self.request("PING")

    def hello(self) -> Dict[str, Any]:
        return self.request(
            "HELLO",
            {
                "client_name": "lerobot_roco_env",
                "client_version": "0.1.0",
                "supported_protocol_versions": [PROTOCOL_VERSION],
            },
        )

    def get_spec(self) -> RoCoEnvSpec:
        payload = self.request("GET_SPEC")
        return RoCoEnvSpec.from_dict(payload["spec"])

    def reset(self, seed: Optional[int], active_agent: str) -> Dict[str, Any]:
        return self.request("RESET", {"seed": seed, "options": {"active_agent": active_agent}})

    def step(self, episode_id: str, step_index: int, action: Any) -> Dict[str, Any]:
        return self.request("STEP", {"episode_id": episode_id, "step_index": int(step_index), "action": action})

    def step_native_action(self, episode_id: str, step_index: int, sim_action: Dict[str, Any]) -> Dict[str, Any]:
        return self.request(
            "STEP_NATIVE_ACTION",
            {"episode_id": episode_id, "step_index": int(step_index), "sim_action": sim_action},
        )

    def render(self) -> Any:
        return self.request("RENDER")["image"]

    def get_state_digest(self) -> Dict[str, Any]:
        return self.request("GET_STATE_DIGEST")

    def close_episode(self) -> None:
        try:
            self.request("CLOSE_EPISODE")
        except Exception:
            return

    def shutdown(self, session_token: Optional[str] = None) -> None:
        payload = {}
        if session_token is not None:
            payload["session_token"] = session_token
        self.request("SHUTDOWN", payload)

    def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None
