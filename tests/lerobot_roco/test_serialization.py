import numpy as np
import pytest

from integrations.lerobot_roco.common.errors import ErrorCode, RoCoProtocolError
from integrations.lerobot_roco.common.serialization import decode_ndarray, decode_value, encode_ndarray, encode_value


def test_ndarray_round_trip_preserves_dtype_shape_and_values():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    payload = encode_ndarray(arr)
    decoded = decode_ndarray(payload)
    assert decoded.dtype == np.float32
    assert decoded.shape == (3, 2)
    np.testing.assert_array_equal(decoded, np.ascontiguousarray(arr))
    assert decoded.flags.c_contiguous


def test_nested_value_round_trip_with_uint8_image():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = encode_value({"pixels": {"front": image}, "agent_pos": np.ones(2, dtype=np.float32)})
    decoded = decode_value(payload)
    assert decoded["pixels"]["front"].dtype == np.uint8
    assert decoded["pixels"]["front"].shape == (4, 5, 3)
    assert decoded["agent_pos"].dtype == np.float32


def test_rejects_object_dtype():
    with pytest.raises(RoCoProtocolError) as exc:
        encode_ndarray(np.asarray([object()], dtype=object))
    assert exc.value.code == ErrorCode.SERIALIZATION_ERROR


def test_rejects_mismatched_byte_count():
    payload = encode_ndarray(np.arange(3, dtype=np.int32))
    payload["data"] = payload["data"][:-1]
    with pytest.raises(RoCoProtocolError) as exc:
        decode_ndarray(payload)
    assert exc.value.code == ErrorCode.SERIALIZATION_ERROR


def test_rejects_oversized_array():
    arr = np.zeros((16,), dtype=np.uint8)
    with pytest.raises(RoCoProtocolError) as exc:
        encode_ndarray(arr, max_payload_bytes=8)
    assert exc.value.code == ErrorCode.PAYLOAD_TOO_LARGE
