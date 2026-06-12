from integrations.lerobot_roco.common.errors import ErrorCode, RoCoActionError, exception_from_error_payload


def test_error_payload_maps_to_typed_exception():
    exc = exception_from_error_payload(
        {
            "code": ErrorCode.ACTION_OUT_OF_BOUNDS,
            "message": "outside",
            "details": {"field": "action"},
            "retryable": False,
        }
    )
    assert isinstance(exc, RoCoActionError)
    assert exc.code == ErrorCode.ACTION_OUT_OF_BOUNDS
    assert exc.details == {"field": "action"}
