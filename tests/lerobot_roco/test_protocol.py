import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoProtocolError
from integrations.lerobot_roco.common.protocol import (
    PROTOCOL_VERSION,
    make_error_response,
    make_request,
    make_success_response,
    validate_request_envelope,
    validate_response_envelope,
)


def test_valid_request_and_response_envelopes():
    request = make_request("PING")
    validate_request_envelope(request)
    response = make_success_response(request, {"ok": "yes"})
    validate_response_envelope(response, request["request_id"])


def test_rejects_protocol_mismatch():
    request = make_request("PING")
    request["protocol_version"] = "9.9"
    with pytest.raises(RoCoProtocolError) as exc:
        validate_request_envelope(request)
    assert exc.value.code == ErrorCode.UNSUPPORTED_PROTOCOL


def test_rejects_unknown_command():
    request = make_request("PING")
    request["command"] = "NOPE"
    with pytest.raises(RoCoProtocolError) as exc:
        validate_request_envelope(request)
    assert exc.value.code == ErrorCode.UNKNOWN_COMMAND


def test_rejects_debug_command_when_disabled():
    request = make_request("STEP_NATIVE_ACTION")
    with pytest.raises(RoCoProtocolError) as exc:
        validate_request_envelope(request, allow_debug_commands=False)
    assert exc.value.code == ErrorCode.DEBUG_COMMAND_DISABLED


def test_response_request_id_must_match():
    request = make_request("PING")
    response = make_success_response(request, {})
    with pytest.raises(RoCoProtocolError):
        validate_response_envelope(response, "different")


def test_error_response_shape_is_stable():
    request = make_request("PING")
    response = make_error_response(request, ErrorCode.INVALID_REQUEST, "bad")
    validate_response_envelope(response, request["request_id"])
    assert response["error"]["code"] == ErrorCode.INVALID_REQUEST
    assert response["error"]["retryable"] is False
