import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoBridgeError, RoCoProtocolError
from integrations.lerobot_roco.common.protocol import make_success_response
from lerobot_roco_env import client as client_module
from lerobot_roco_env.client import RemoteRoCoClient

pytestmark = pytest.mark.client


class FakeSocket:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.sent = None

    def send(self, data):
        self.sent = data

    def recv(self):
        return b"response"


class FakeClient(RemoteRoCoClient):
    def __init__(self, response_factory):
        super().__init__()
        self.socket = FakeSocket(response_factory)
        self.response_factory = response_factory

    def _ensure_socket(self):
        return self.socket


def test_remote_client_validates_success_response(monkeypatch):
    captured = {}

    def fake_pack(request, max_payload_bytes):
        captured["request"] = request
        return b"request"

    def fake_unpack(data, max_payload_bytes):
        return make_success_response(captured["request"], {"pong": True})

    monkeypatch.setattr(client_module, "pack_message", fake_pack)
    monkeypatch.setattr(client_module, "unpack_message", fake_unpack)
    client = FakeClient(fake_unpack)
    assert client.request("PING") == {"pong": True}


def test_remote_client_maps_server_error(monkeypatch):
    captured = {}

    def fake_pack(request, max_payload_bytes):
        captured["request"] = request
        return b"request"

    def fake_unpack(data, max_payload_bytes):
        return {
            "protocol_version": "0.1",
            "request_id": captured["request"]["request_id"],
            "ok": False,
            "error": {
                "code": ErrorCode.ACTION_OUT_OF_BOUNDS,
                "message": "bad action",
                "details": {},
                "retryable": False,
            },
        }

    monkeypatch.setattr(client_module, "pack_message", fake_pack)
    monkeypatch.setattr(client_module, "unpack_message", fake_unpack)
    client = FakeClient(fake_unpack)
    with pytest.raises(RoCoBridgeError) as exc:
        client.request("STEP", {})
    assert exc.value.code == ErrorCode.ACTION_OUT_OF_BOUNDS


def test_remote_client_preserves_packing_errors(monkeypatch):
    def fake_pack(request, max_payload_bytes):
        raise RoCoProtocolError("msgpack is required for bridge transport.", code=ErrorCode.SERIALIZATION_ERROR)

    monkeypatch.setattr(client_module, "pack_message", fake_pack)
    client = FakeClient(lambda data, max_payload_bytes: None)

    with pytest.raises(RoCoProtocolError) as exc:
        client.request("PING")

    assert exc.value.code == ErrorCode.SERIALIZATION_ERROR
