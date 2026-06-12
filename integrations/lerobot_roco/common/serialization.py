"""Safe explicit serialization for NumPy arrays and protocol payloads."""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np

from .errors import ErrorCode, RoCoProtocolError

MAX_NDIMS = 8
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _validate_shape(shape: Any, max_ndims: int) -> tuple:
    if not isinstance(shape, (list, tuple)):
        raise RoCoProtocolError("Array shape must be a list.", code=ErrorCode.SERIALIZATION_ERROR)
    if len(shape) > max_ndims:
        raise RoCoProtocolError(
            "Array has too many dimensions.",
            code=ErrorCode.SERIALIZATION_ERROR,
            details={"ndim": len(shape), "max_ndims": max_ndims},
        )
    out = []
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise RoCoProtocolError(
                "Array shape contains an invalid dimension.",
                code=ErrorCode.SERIALIZATION_ERROR,
                details={"shape": list(shape)},
            )
        out.append(int(dim))
    return tuple(out)


def _expected_nbytes(dtype: np.dtype, shape: tuple) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return int(count * dtype.itemsize)


def encode_ndarray(
    array: np.ndarray,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    max_ndims: int = MAX_NDIMS,
) -> Dict[str, Any]:
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise RoCoProtocolError("Object dtype arrays are not allowed.", code=ErrorCode.SERIALIZATION_ERROR)
    if arr.ndim > max_ndims:
        raise RoCoProtocolError(
            "Array has too many dimensions.",
            code=ErrorCode.SERIALIZATION_ERROR,
            details={"ndim": arr.ndim, "max_ndims": max_ndims},
        )
    contiguous = np.ascontiguousarray(arr)
    if contiguous.nbytes > max_payload_bytes:
        raise RoCoProtocolError(
            "Array payload is too large.",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            details={"nbytes": int(contiguous.nbytes), "max_payload_bytes": int(max_payload_bytes)},
        )
    return {
        "__ndarray__": True,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(order="C"),
    }


def decode_ndarray(
    payload: Dict[str, Any],
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    max_ndims: int = MAX_NDIMS,
) -> np.ndarray:
    if not isinstance(payload, dict) or payload.get("__ndarray__") is not True:
        raise RoCoProtocolError("Invalid ndarray payload.", code=ErrorCode.SERIALIZATION_ERROR)
    try:
        dtype = np.dtype(payload["dtype"])
    except Exception as exc:
        raise RoCoProtocolError(
            "Invalid ndarray dtype.",
            code=ErrorCode.SERIALIZATION_ERROR,
            details={"dtype": payload.get("dtype")},
        ) from exc
    if dtype.hasobject:
        raise RoCoProtocolError("Object dtype arrays are not allowed.", code=ErrorCode.SERIALIZATION_ERROR)
    shape = _validate_shape(payload.get("shape"), max_ndims=max_ndims)
    data = payload.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise RoCoProtocolError("Array data must be bytes.", code=ErrorCode.SERIALIZATION_ERROR)
    expected = _expected_nbytes(dtype, shape)
    actual = len(data)
    if actual != expected:
        raise RoCoProtocolError(
            "Array byte length does not match dtype and shape.",
            code=ErrorCode.SERIALIZATION_ERROR,
            details={"expected_nbytes": expected, "actual_nbytes": actual},
        )
    if actual > max_payload_bytes:
        raise RoCoProtocolError(
            "Array payload is too large.",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            details={"nbytes": actual, "max_payload_bytes": int(max_payload_bytes)},
        )
    return np.frombuffer(bytes(data), dtype=dtype).reshape(shape).copy()


def encode_value(value: Any, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> Any:
    if isinstance(value, np.ndarray):
        return encode_ndarray(value, max_payload_bytes=max_payload_bytes)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return encode_value(asdict(value), max_payload_bytes=max_payload_bytes)
    if isinstance(value, dict):
        return {str(k): encode_value(v, max_payload_bytes=max_payload_bytes) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v, max_payload_bytes=max_payload_bytes) for v in value]
    return value


def decode_value(value: Any, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> Any:
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            return decode_ndarray(value, max_payload_bytes=max_payload_bytes)
        return {k: decode_value(v, max_payload_bytes=max_payload_bytes) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v, max_payload_bytes=max_payload_bytes) for v in value]
    return value


def estimate_payload_size(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            total += len(str(key).encode("utf-8"))
            total += estimate_payload_size(item)
        return total
    if isinstance(value, (list, tuple)):
        return sum(estimate_payload_size(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 8


def pack_message(message: Dict[str, Any], max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> bytes:
    try:
        import msgpack
    except ImportError as exc:
        raise RoCoProtocolError(
            "msgpack is required for bridge transport.",
            code=ErrorCode.SERIALIZATION_ERROR,
        ) from exc
    encoded = encode_value(message, max_payload_bytes=max_payload_bytes)
    packed = msgpack.packb(encoded, use_bin_type=True)
    if len(packed) > max_payload_bytes:
        raise RoCoProtocolError(
            "Packed message exceeds maximum payload size.",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            details={"nbytes": len(packed), "max_payload_bytes": int(max_payload_bytes)},
        )
    return packed


def unpack_message(data: bytes, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> Dict[str, Any]:
    if len(data) > max_payload_bytes:
        raise RoCoProtocolError(
            "Packed message exceeds maximum payload size.",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            details={"nbytes": len(data), "max_payload_bytes": int(max_payload_bytes)},
        )
    try:
        import msgpack
    except ImportError as exc:
        raise RoCoProtocolError(
            "msgpack is required for bridge transport.",
            code=ErrorCode.SERIALIZATION_ERROR,
        ) from exc
    try:
        decoded = msgpack.unpackb(data, raw=False)
    except Exception as exc:
        raise RoCoProtocolError("Could not unpack message.", code=ErrorCode.SERIALIZATION_ERROR) from exc
    value = decode_value(decoded, max_payload_bytes=max_payload_bytes)
    if not isinstance(value, dict):
        raise RoCoProtocolError("Protocol message must be a map.", code=ErrorCode.SERIALIZATION_ERROR)
    return value
