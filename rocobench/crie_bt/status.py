"""Shared CRIE-BT status enumerations.

The CRIE-BT package is intentionally Python 3.8-compatible because it lives
inside the RoCo runtime package.
"""

from enum import Enum


class ExecutionMode(Enum):
    OPEN_LOOP = "open_loop"
    DIRECT_FEEDBACK = "direct_feedback"
    BT_MEDIATED = "bt_mediated"
    VLM_SARM_MONITOR_PLANNER = "vlm_sarm_monitor_planner"


class BTStatus(Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class RuntimeDecision(Enum):
    CONTINUE = "CONTINUE"
    LOCAL_RETRY = "LOCAL_RETRY"
    EXPLAIN = "EXPLAIN"
    REQUEST_HUMAN_INPUT = "REQUEST_HUMAN_INPUT"
    REQUEST_REPLAN = "REQUEST_REPLAN"
    ABORT = "ABORT"


class FailureCode(Enum):
    NONE = "NONE"
    MISSED_GRASP = "MISSED_GRASP"
    SLIPPAGE = "SLIPPAGE"
    NO_PROGRESS = "NO_PROGRESS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
    TARGET_OCCUPIED = "TARGET_OCCUPIED"
    WRONG_OBJECT = "WRONG_OBJECT"
    WRONG_TARGET = "WRONG_TARGET"
    TIMEOUT = "TIMEOUT"
    SAFETY_CONFLICT = "SAFETY_CONFLICT"
    UNKNOWN = "UNKNOWN"


class ProgressStage(Enum):
    NOT_STARTED = "NOT_STARTED"
    APPROACHING_OBJECT = "APPROACHING_OBJECT"
    NEAR_OBJECT = "NEAR_OBJECT"
    GRASP_ATTEMPT = "GRASP_ATTEMPT"
    OBJECT_GRASPED = "OBJECT_GRASPED"
    TRANSPORTING = "TRANSPORTING"
    NEAR_TARGET = "NEAR_TARGET"
    RELEASING = "RELEASING"
    OBJECT_IN_TARGET = "OBJECT_IN_TARGET"
    STABLE_SUCCESS = "STABLE_SUCCESS"
    STUCK = "STUCK"
    FAILED = "FAILED"


def enum_value(value):
    """Return a JSON-friendly enum value."""
    if isinstance(value, Enum):
        return value.value
    return value


def coerce_enum(enum_cls, value, default=None):
    """Coerce strings or enum instances to ``enum_cls``."""
    if isinstance(value, enum_cls):
        return value
    if value is None:
        if default is not None:
            return default
        raise ValueError("Missing enum value for {}".format(enum_cls.__name__))
    text = str(value)
    for item in enum_cls:
        if text == item.value or text == item.name:
            return item
    if default is not None:
        return default
    raise ValueError("Unknown {} value: {}".format(enum_cls.__name__, value))
